"""Speaker-independent k-fold cross-validation over the RAVDESS actors.

`main.py train` / `test` use one actor split (19 training / 2 validation / 3 test actors), so the test accuracy
depends on which 3 actors happened to be drawn. Cross-validation tests every actor exactly once:

  * the actors are dealt into k folds (default 8: 3 test actors per fold, the size of that split);
  * fold i tests on its own actors, validates on the 2 actors that follow them in the fold order (early stopping
    and checkpoint selection) and trains on the other 19;
  * every fold is trained like `main.py train` (same model, AudioSet starting point, early stopping, 8 best
    checkpoints by validation accuracy) and averaged like `main.py weight_average`;
  * the averaged model is tested on the fold's test actors (simulated rooms, like `main.py test`) and, to show
    overfitting, on its validation and training actors too (eval mode, no augmentation). Training accuracy minus
    test accuracy is the generalisation gap; the loss curves are in fold<i>/logs/version_*/metrics.csv;
  * with --ace, also on the test actors convolved with the 14 real ACE RIRs, like `main.py test_ace`.

The folds re-partition the rows of the existing dataset (config.dataset_path, built by prepare_data.py --rev):
each utterance keeps the simulated room it was given there, only its split changes. The dataset is not modified.

    python cross_validation.py --plan         # print the folds
    python cross_validation.py                # every fold (about an hour each on an M4 Max), then the summary
    python cross_validation.py --ace          # ... and test every fold on the ACE rooms too (+ a few minutes each)
    python cross_validation.py --folds 0 3    # only these folds. On a machine with several GPUs, run folds side by
                                              # side: CUDA_VISIBLE_DEVICES=1 python cross_validation.py --folds 1
    python cross_validation.py --summary      # the summary of the folds finished so far
    python cross_validation.py --actors 1-6 --k 3 --max-epoch 1 --out /tmp/cv_check   # quick pipeline check

A fold runs on one device (one GPU, the Mac GPU or the CPU); per-step batch = config.batch_size //
config.accumulate_grad_batches. Running the script again skips finished folds and restarts an unfinished one from
its beginning; --ace on a finished fold only adds its ACE results.
Outputs, in <workspace>/results_.../<exp_name>_cv<k>/ unless --out is given: folds.json, config.json;
fold<i>/ with result.json, predictions.csv (+ predictions_ace.csv), model_avg.ckpt (252 MB) and logs/;
summary.csv, per_actor.csv and, with --ace, ace.csv.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1") # as main.py: run ops the Apple-silicon GPU lacks on the CPU
import argparse
import contextlib
import gc
import glob
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score, classification_report
from confidence_intervals import evaluate_with_conf_int

import config
import main
import prepare_data as prep
from data_generator import Emotion_DataLoader
from sed_model import SEDWrapper
from utils import dump_config

HEAD = ["sed_model.head.weight", "sed_model.head.bias"] # not loaded for testing, as in main.test() (unused with tscam)
NOT_PRETRAINED = HEAD + ["sed_model.tscam_conv.weight", "sed_model.tscam_conv.bias"] # as in main.train()
CI_ALPHA = 25 # 75% confidence intervals, as in sed_model.py


# ----------------------------------------------------------------------------------------------------- the folds
def make_folds(actors, k=8, n_val=2, seed=0):
    """k folds in which every actor is tested exactly once.

    The actors are shuffled (seeded) with men and women alternating -- RAVDESS actors with odd numbers are male,
    even ones female -- and cut into k consecutive groups, so every fold has both. Fold i tests on group i,
    validates on the n_val actors that follow it in that order (wrapping around) and trains on the rest."""
    actors = sorted({int(a) for a in actors})
    if not 2 <= k <= len(actors):
        raise ValueError("k must be between 2 and the number of actors (%d), got %d" % (len(actors), k))
    rng = np.random.RandomState(seed)
    men = [int(a) for a in rng.permutation([a for a in actors if a % 2 == 1])]
    women = [int(a) for a in rng.permutation([a for a in actors if a % 2 == 0])]
    order = [a for pair in itertools.zip_longest(men, women) for a in pair if a is not None]
    folds = []
    for group in np.array_split(np.arange(len(order)), k):
        test = [order[i] for i in group]
        val = [order[(int(group[-1]) + 1 + j) % len(order)] for j in range(n_val)]
        train = [a for a in order if a not in test and a not in val]
        if set(val) & set(test) or not train:
            raise ValueError("%d actors are too few for k=%d with %d validation actors" % (len(actors), k, n_val))
        folds.append({"train": sorted(train), "eval": sorted(val), "test": sorted(test)})
    return folds


def print_plan(folds, pool):
    clips = {}
    for row in pool:
        clips[row_actor(row)] = clips.get(row_actor(row), 0) + 1
    count = lambda actors: sum(clips[a] for a in actors)
    print("%d folds over %d actors, %d clips (odd actor numbers are men, even women):" % (len(folds), len(clips), len(pool)))
    for i, fold in enumerate(folds):
        print("  fold %d: test %-9s (%3d clips) | validation %-6s (%3d) | training %2d actors (%4d)" % (
            i, ",".join(map(str, fold["test"])), count(fold["test"]), ",".join(map(str, fold["eval"])),
            count(fold["eval"]), len(fold["train"]), count(fold["train"])))


# ---------------------------------------------------------------------------------------------------------- data
def row_name(row):
    return row.get("wav_name", row.get("name", ""))


def row_actor(row):
    """RAVDESS actor of a dataset row: its "actor" field (prepare_data.py), else the last field of the file name
    (03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav)."""
    if "actor" in row:
        return int(row["actor"])
    return int(os.path.splitext(os.path.basename(row_name(row)))[0].split("-")[-1])


def load_pool(dataset_path, actors=None):
    """All rows of the dataset's train, eval and test files, i.e. every utterance with its simulated room."""
    rows = []
    for name in list(main.TRAIN_FILES["RAVDESS"]) + [main.TEST_FILES["RAVDESS"]]:
        path = os.path.join(dataset_path, name)
        if not os.path.exists(path):
            raise SystemExit("%s not found -- build the dataset with prepare_data.py --rev" % path)
        rows += [item[0] for item in np.load(path, allow_pickle=True)]
    if actors is not None:
        rows = [r for r in rows if row_actor(r) in actors]
    names = [row_name(r) for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("%s: some utterances are in more than one split file" % dataset_path)
    return sorted(rows, key=row_name)


def as_dataset(rows):
    """Rows as the (N, 1) object array Emotion_DataLoader reads (dataset[i][0] is a row), as prepare_data saves them."""
    arr = np.empty((len(rows), 1), dtype=object)
    for i, row in enumerate(rows):
        arr[i, 0] = row
    return arr


def ace_sets(actors):
    """(mic position, room, rows) for every ACE Mobile RIR: the actors' clean utterances convolved with it, built in
    memory exactly as prepare_data.build_ace_sets() writes the test_ace sets."""
    records = prep.list_ravdess(actors=actors)
    for ace_config, position in prep.ACE_POSITIONS.items():
        for ace_room, room in prep.ACE_ROOMS.items():
            found = sorted(glob.glob(os.path.join(prep.ACE_RIR_DIR, "Mobile", ace_room, ace_config, "*RIR*.wav")))
            if not found:
                raise FileNotFoundError("no ACE RIR for %s/%s in %s -- run prepare_data.py --download-ace"
                                        % (ace_room, ace_config, prep.ACE_RIR_DIR))
            yield position, room, [prep._row((r, "ace", "test", 0, found[0])) for r in records]


# -------------------------------------------------------------------------------------------------------- model
class FoldModel(SEDWrapper):
    """SEDWrapper that keeps the per-clip outputs of its last test run, for per-actor accuracy."""
    last_pred = None

    def on_test_epoch_end(self):
        if self._test_outputs:
            self.last_pred = torch.cat([pred for pred, _ in self._test_outputs]).float().cpu().numpy()
        return super().on_test_epoch_end()


def build_model(dataset):
    """main.build_model() (the same network) as a FoldModel. `dataset` is only used while training."""
    return FoldModel(sed_model=main.build_model(dataset).sed_model, config=config, dataset=dataset)


@contextlib.contextmanager
def one_device():
    """Train and test on one device: no DDP, so test predictions stay in dataset order and the confidence intervals
    get computed (sed_model.py computes them only in single-device runs)."""
    resolve = main.resolve_devices
    main.resolve_devices = lambda: (resolve()[0], 1) # main.make_trainer() looks it up when called
    try:
        yield
    finally:
        main.resolve_devices = resolve


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def average_checkpoints(ckpt_dir, model_path):
    """main.weight_average() on one fold's checkpoints, written atomically to model_path."""
    saved = config.wa_folder, config.wa_model_path
    config.wa_folder, config.wa_model_path = ckpt_dir, model_path + ".tmp"
    try:
        main.weight_average()
    finally:
        config.wa_folder, config.wa_model_path = saved
    os.replace(model_path + ".tmp", model_path)


def train_fold(fold_dir, rows, keep_checkpoints=False):
    """Train like main.train() (rows["eval"] for early stopping and checkpoint selection), then average the best
    checkpoints like main.weight_average() into fold_dir/model_avg.ckpt. Returns a summary of the training."""
    ckpt_dir = os.path.join(fold_dir, "checkpoints")
    if os.path.isdir(ckpt_dir): # left by an interrupted run of this fold, which starts again
        print("removing the checkpoints of an unfinished run in", ckpt_dir)
        shutil.rmtree(ckpt_dir)
    pl.seed_everything(config.random_seed)
    train_set = Emotion_DataLoader(dataset=as_dataset(rows["train"]), config=config, eval_mode=False)
    val_set = Emotion_DataLoader(dataset=as_dataset(rows["eval"]), config=config, eval_mode=True)
    # main.train()'s callbacks, with weights-only checkpoints (252 MB instead of 746 MB: no optimizer state)
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, monitor="acc", mode="max", save_top_k=8,
                                 filename="l-{epoch:d}-{acc:.3f}", save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=config.es_patience, mode="min", verbose=True)
    trainer, _ = main.make_trainer(
        deterministic=False,
        default_root_dir=fold_dir,
        max_epochs=config.max_epoch,
        sync_batchnorm=True,
        callbacks=[checkpoint, early_stopping],
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=getattr(config, "accumulate_grad_batches", 1),
        logger=CSVLogger(save_dir=fold_dir, name="logs"), # the training / validation loss curves
        log_every_n_steps=1, # train_loss of every step (the default, every 50 steps, is about once an epoch here)
    )
    model = build_model(train_set)
    if config.resume_checkpoint is not None:
        main.load_weights(model, config.resume_checkpoint, drop=NOT_PRETRAINED)
    start = time.time()
    trainer.fit(model, main.data_prep(train_set, val_set, 1))
    info = {
        "epochs": int(trainer.current_epoch),
        "minutes": round((time.time() - start) / 60, 1),
        "best_val_acc": None if checkpoint.best_model_score is None else float(checkpoint.best_model_score),
        "averaged": {os.path.basename(p): float(s) for p, s in sorted(checkpoint.best_k_models.items())},
    }
    del trainer, model
    free_memory()
    average_checkpoints(ckpt_dir, os.path.join(fold_dir, "model_avg.ckpt"))
    if not keep_checkpoints:
        shutil.rmtree(ckpt_dir)
    return info


def test_rows(model, rows):
    """main.evaluate() on rows -> ({acc, CI_low, CI_high, clips}, predicted class of every row)."""
    dataset = Emotion_DataLoader(dataset=as_dataset(rows), config=config, eval_mode=True)
    model.last_pred = None
    res = main.evaluate(model, dataset)[0]
    if model.last_pred is None or len(model.last_pred) != len(rows):
        raise RuntimeError("expected one test prediction per clip (%d)" % len(rows))
    ci = [float(res[key]) if key in res else None for key in ("CI_low", "CI_high")]
    return {"acc": float(res["acc"]), "CI_low": ci[0], "CI_high": ci[1], "clips": len(rows)}, model.last_pred.argmax(1)


def predictions_frame(rows, pred, **columns):
    """One line per clip: the given constant columns, then wav_name, actor, target and the predicted class."""
    frame = pd.DataFrame({"wav_name": [row_name(r) for r in rows], "actor": [row_actor(r) for r in rows],
                          "target": [int(r["target"]) for r in rows], "pred": [int(p) for p in pred]})
    for i, (key, value) in enumerate(columns.items()):
        frame.insert(i, key, value)
    return frame


def evaluate_fold(index, fold_dir, split, rows, result, ace, result_file):
    """Add what `result` lacks: the averaged model's accuracy on the test, validation and training actors (test
    predictions in predictions.csv) and, with ace, on the 14 ACE rooms (predictions_ace.csv)."""
    model = build_model(None)
    main.load_weights(model, os.path.join(fold_dir, "model_avg.ckpt"), drop=HEAD)
    if "test" not in result:
        result["test"], pred = test_rows(model, rows["test"])
        predictions_frame(rows["test"], pred, fold=index).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        result["eval"], _ = test_rows(model, rows["eval"])
        result["train"], _ = test_rows(model, rows["train"])
        save_json(result, result_file)
    if ace and "ace" not in result:
        scores, frames = [], []
        for position, room, room_rows in ace_sets(split["test"]):
            metrics, pred = test_rows(model, room_rows)
            scores.append(dict(mic_position=position, room=room, **metrics))
            frames.append(predictions_frame(room_rows, pred, fold=index, mic_position=position, room=room))
        pd.concat(frames).to_csv(os.path.join(fold_dir, "predictions_ace.csv"), index=False)
        result["ace"] = scores
        save_json(result, result_file)
    del model
    free_memory()
    return result


def run_fold(index, split, pool, out_dir, ace=False, keep_checkpoints=False):
    """Train fold `index` (unless its averaged model exists) and test it; returns its result."""
    fold_dir = os.path.join(out_dir, "fold%d" % index)
    result_file = os.path.join(fold_dir, "result.json")
    result = load_json(result_file) or {"fold": index, "actors": split}
    if "test" in result and (not ace or "ace" in result):
        print("fold %d: done" % index)
        return result
    rows = {name: [r for r in pool if row_actor(r) in actors] for name, actors in split.items()}
    print("\n=== fold %d: test actors %s | validation %s | %d training actors" % (
        index, split["test"], split["eval"], len(split["train"])))
    os.makedirs(fold_dir, exist_ok=True)
    with one_device():
        if not os.path.exists(os.path.join(fold_dir, "model_avg.ckpt")):
            result["training"] = train_fold(fold_dir, rows, keep_checkpoints)
            save_json(result, result_file)
        return evaluate_fold(index, fold_dir, split, rows, result, ace, result_file)


# ------------------------------------------------------------------------------------------------------- summary
def pooled_accuracy(preds):
    """Accuracy over all the clips, with a 75% CI from bootstrapping actors (an actor's clips stay together)."""
    return evaluate_with_conf_int(preds["pred"].to_numpy(), accuracy_score, preds["target"].to_numpy(),
                                  conditions=preds["actor"].to_numpy(), alpha=CI_ALPHA)


def summarize(out_dir, folds):
    """Print the cross-validated results of the finished folds; write summary.csv, per_actor.csv (and ace.csv)."""
    done = [r for r in (load_json(os.path.join(out_dir, "fold%d" % i, "result.json")) for i in range(len(folds)))
            if r and "test" in r]
    if not done:
        print("no finished folds in", out_dir)
        return
    table = pd.DataFrame([{
        "fold": r["fold"],
        "test_actors": " ".join(str(a) for a in r["actors"]["test"]),
        "epochs": r.get("training", {}).get("epochs"),
        "minutes": r.get("training", {}).get("minutes"),
        "train_acc": r["train"]["acc"],
        "val_acc": r["eval"]["acc"],
        "test_acc": r["test"]["acc"],
        "gap": r["train"]["acc"] - r["test"]["acc"],
        "ace_acc": np.mean([a["acc"] for a in r["ace"]]) if "ace" in r else np.nan,
    } for r in done])
    table.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    preds = pd.concat([pd.read_csv(os.path.join(out_dir, "fold%d" % r["fold"], "predictions.csv")) for r in done])
    acc, (low, high) = pooled_accuracy(preds)
    per_actor = (preds.assign(correct=preds["pred"] == preds["target"]).groupby("actor")
                 .agg(fold=("fold", "first"), clips=("correct", "size"), acc=("correct", "mean")).reset_index())
    per_actor.insert(1, "gender", np.where(per_actor["actor"] % 2 == 1, "male", "female"))
    per_actor.to_csv(os.path.join(out_dir, "per_actor.csv"), index=False)

    show = table.copy()
    for column in ("train_acc", "val_acc", "test_acc", "gap", "ace_acc"):
        show[column] = show[column] * 100
    if show["ace_acc"].isna().all():
        show = show.drop(columns="ace_acc")
    print("\n" + "=" * 100)
    print("Cross-validation over actors: %d of %d folds finished (%s)" % (len(done), len(folds), out_dir))
    print("accuracies in %; gap = train_acc - test_acc (the averaged model, eval mode)")
    print(show.to_string(index=False, float_format=lambda x: "%.1f" % x, na_rep="-"))
    spread = " +- %.1f (std)" % (table["test_acc"].std() * 100) if len(table) > 1 else ""
    print("test accuracy: %.1f%% over %d clips of %d actors, 75%% CI %.1f-%.1f (bootstrapping actors) | per fold "
          "%.1f%s" % (acc * 100, len(preds), len(per_actor), low * 100, high * 100, table["test_acc"].mean() * 100, spread))
    print("train - test accuracy gap: %.1f points on average" % (table["gap"].mean() * 100))
    worst, best = per_actor.loc[per_actor["acc"].idxmin()], per_actor.loc[per_actor["acc"].idxmax()]
    print("per actor: from %.1f%% (actor %d) to %.1f%% (actor %d), see per_actor.csv"
          % (worst["acc"] * 100, worst["actor"], best["acc"] * 100, best["actor"]))
    print(classification_report(preds["target"], preds["pred"], labels=list(range(len(prep.EMOTIONS))),
                                target_names=prep.EMOTIONS, digits=3, zero_division=0))

    ace_files = [os.path.join(out_dir, "fold%d" % r["fold"], "predictions_ace.csv") for r in done if "ace" in r]
    if ace_files:
        ace = pd.concat([pd.read_csv(f) for f in ace_files])
        rooms = []
        for (position, room), d in ace.groupby(["mic_position", "room"], sort=False):
            room_acc, (room_low, room_high) = pooled_accuracy(d)
            rooms.append({"mic_position": position, "room": room, "acc": room_acc, "CI_low": room_low,
                          "CI_high": room_high, "clips": len(d), "actors": d["actor"].nunique()})
        rooms = pd.DataFrame(rooms)
        rooms.to_csv(os.path.join(out_dir, "ace.csv"), index=False)
        show = rooms.copy()
        for column in ("acc", "CI_low", "CI_high"):
            show[column] = show[column] * 100
        print("ACE rooms (the test actors of %d fold%s, 75%% CI bootstrapping actors):"
              % (len(ace_files), "" if len(ace_files) == 1 else "s"))
        print(show.to_string(index=False, float_format=lambda x: "%.1f" % x))
        print("mean over the %d room/position sets: %.1f%%" % (len(rooms), rooms["acc"].mean() * 100))


# -------------------------------------------------------------------------------------------------------- files
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    """Write JSON atomically (several processes may run folds of the same cross-validation)."""
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def run_settings(actors, args):
    """What the results depend on: running again in an existing output folder has to use the same."""
    return {
        "actors": actors, "k": args.k, "n_val": args.n_val, "seed": args.seed,
        "dataset": os.path.basename(os.path.normpath(config.dataset_path)),
        "pretrained": os.path.basename(config.resume_checkpoint) if config.resume_checkpoint else None,
        "audio": bool(config.audio), "video": bool(config.video), "num_mics": config.num_mics,
        "mode_type": config.mode_type, "batch_size": config.batch_size, "max_epoch": config.max_epoch,
        "random_seed": config.random_seed,
    }


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8, help="number of folds (default 8: 3 test actors per fold)")
    ap.add_argument("--n-val", type=int, default=2, help="validation actors per fold (default 2)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the actor shuffle (default 0)")
    ap.add_argument("--actors", help="use only these actors, e.g. 1-6 (default: all in the dataset)")
    ap.add_argument("--folds", type=int, nargs="+", help="run only these folds (default: all)")
    ap.add_argument("--ace", action="store_true", help="also test every fold on the 14 ACE room/position sets")
    ap.add_argument("--max-epoch", type=int, help="override config.max_epoch (e.g. 1 for a quick check)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep each fold's 8 best checkpoints (252 MB each) after averaging them")
    ap.add_argument("--out", help="output folder (default: <workspace>/results_.../<exp_name>_cv<k>)")
    ap.add_argument("--plan", action="store_true", help="print the folds and exit")
    ap.add_argument("--summary", action="store_true", help="only print the summary of the finished folds")
    args = ap.parse_args()

    if config.dataset_type != "RAVDESS":
        raise SystemExit("cross-validation over actors is implemented for RAVDESS (config.dataset_type = %r)"
                         % config.dataset_type)
    if args.max_epoch is not None:
        config.max_epoch = args.max_epoch
    out_dir = os.path.abspath(args.out or main.experiment_dir() + "_cv%d" % args.k)
    plan_file = os.path.join(out_dir, "folds.json")
    if args.summary:
        plan = load_json(plan_file)
        if plan is None:
            raise SystemExit("no cross-validation in %s (no folds.json)" % out_dir)
        summarize(out_dir, plan["folds"])
        return

    pool = load_pool(config.dataset_path, prep.parse_actors(args.actors) if args.actors else None)
    actors = sorted({row_actor(r) for r in pool})
    folds = make_folds(actors, args.k, args.n_val, args.seed)
    print_plan(folds, pool)
    if args.plan:
        return
    selected = list(range(len(folds))) if args.folds is None else args.folds
    if any(not 0 <= i < len(folds) for i in selected):
        raise SystemExit("--folds: the folds are numbered 0-%d" % (len(folds) - 1))
    if args.ace and not glob.glob(os.path.join(prep.ACE_RIR_DIR, "Mobile", "*", "*", "*RIR*.wav")):
        raise SystemExit("--ace needs the ACE RIRs in %s: run prepare_data.py --download-ace" % prep.ACE_RIR_DIR)
    if config.resume_checkpoint is not None and not os.path.exists(config.resume_checkpoint):
        raise SystemExit("config.resume_checkpoint not found: %s (prepare_data.py --checkpoint downloads it)"
                         % config.resume_checkpoint)

    plan = {"settings": run_settings(actors, args), "folds": folds}
    existing = load_json(plan_file)
    if existing is not None and existing["settings"] != plan["settings"]:
        changed = sorted(key for key in plan["settings"] if existing["settings"].get(key) != plan["settings"][key])
        raise SystemExit("%s holds a cross-validation with other settings (%s): use --out for a new one"
                         % (out_dir, ", ".join(changed)))
    os.makedirs(out_dir, exist_ok=True)
    save_json(plan, plan_file)
    dump_config(config, os.path.join(out_dir, "config"))
    for i in selected:
        run_fold(i, folds[i], pool, out_dir, args.ace, args.keep_checkpoints)
        summarize(out_dir, folds)


if __name__ == "__main__":
    cli()

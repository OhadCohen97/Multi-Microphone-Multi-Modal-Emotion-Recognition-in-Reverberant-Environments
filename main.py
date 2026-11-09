import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1") # run any op the Apple-silicon GPU lacks on the CPU
import numpy as np
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.callbacks import EarlyStopping
from utils import create_folder, dump_config
import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from data_generator import Emotion_DataLoader
from model.mmer import MMER
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PL_VERSION = tuple(int(v) for v in pl.__version__.split(".")[:2])

# file names inside config.dataset_path (train, eval) and of a test split
TRAIN_FILES = {
    "RAVDESS": ("ravdess_av_train.npy", "ravdess_av_eval.npy"), # 8 classes
    "IEMOCAP": ("IEMOCAP_train_BIUREVgen.npy", "IEMOCAP_eval_BIUREVgen.npy"), # 4 classes
    "CREMA-D": ("CREMAD_train_BIUREVgen.npy", "CREMAD_eval_BIUREVgen.npy"), # 6 classes
}
TEST_FILES = {"RAVDESS": "ravdess_av_test.npy", "IEMOCAP": "iemocap_test.npy", "CREMA-D": "cremad_test.npy"}


class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num
        # per-step batch: the effective batch split over devices and gradient-accumulation steps
        self.batch_size = max(1, config.batch_size // (device_num * getattr(config, "accumulate_grad_batches", 1)))

    def _loader(self, dataset):
        sampler = DistributedSampler(dataset, shuffle = False) if self.device_num > 1 else None
        return DataLoader(
            dataset = dataset,
            num_workers = config.num_workers,
            batch_size = self.batch_size,
            shuffle = False,
            sampler = sampler
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset)
    def val_dataloader(self):
        return self._loader(self.eval_dataset)
    def test_dataloader(self):
        return self._loader(self.eval_dataset)


def resolve_devices():
    """(accelerator, number of devices) for config.accelerator: "auto" | "gpu" | "mps" | "cpu"."""
    want = str(getattr(config, "accelerator", "auto")).lower()
    if want in ("auto", "gpu", "cuda") and torch.cuda.is_available():
        return "gpu", torch.cuda.device_count()
    mps = getattr(torch.backends, "mps", None)
    if want in ("auto", "mps") and mps is not None and mps.is_available():
        if PL_VERSION >= (1, 7):
            return "mps", 1
        print("pytorch-lightning %s has no Apple-GPU (MPS) support (needs >= 1.7): using the CPU" % pl.__version__)
    elif want not in ("auto", "cpu"):
        print("accelerator %r is not available here: using the CPU" % want)
    return "cpu", 1


def make_trainer(**kwargs):
    """pl.Trainer on the devices resolve_devices() picks. Works with pytorch-lightning 1.x (the Docker image)
    and 2.x (requirements-mac.txt). Returns (trainer, number of devices)."""
    accelerator, n = resolve_devices()
    distributed = n > 1
    if PL_VERSION >= (1, 5):
        kwargs.update(accelerator=accelerator, devices=n)
        kwargs["strategy"] = "ddp" if distributed else ("auto" if PL_VERSION >= (2, 0) else None)
    else:
        kwargs.update(gpus=n if accelerator == "gpu" else 0, accelerator="ddp" if distributed else None)
        if "enable_checkpointing" in kwargs:
            kwargs["checkpoint_callback"] = kwargs.pop("enable_checkpointing")
    kwargs["use_distributed_sampler" if PL_VERSION >= (2, 0) else "replace_sampler_ddp"] = False
    kwargs["sync_batchnorm"] = distributed and kwargs.get("sync_batchnorm", False)
    accumulate = kwargs.get("accumulate_grad_batches", 1)
    print("Trainer: %s x%d, %d sample(s) per device per step%s" % (accelerator, n,
          max(1, config.batch_size // (n * getattr(config, "accumulate_grad_batches", 1))),
          ", gradients accumulated over %d steps" % accumulate if accumulate > 1 else ""))
    return pl.Trainer(**kwargs), n


def load_checkpoint(path):
    """torch.load with the safe tensors-only unpickler first; full unpickling (which can run code from the file)
    only for checkpoints that need it, e.g. Lightning checkpoints storing callback objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location="cpu")
    except Exception: # pickle.UnpicklingError for non-tensor content
        print("%s stores Python objects besides tensors (pytorch-lightning 1.x checkpoints do): loading it with full "
              "unpickling, which can run code from the file -- fine for your own checkpoints, only use it with files "
              "you trust" % os.path.basename(path))
        return torch.load(path, map_location="cpu", weights_only=False)


def load_weights(model, path, drop=()):
    """Load a checkpoint into `model` with strict=False (as before) and print what did not match, so a wrong
    checkpoint or config no longer goes unnoticed."""
    if not os.path.exists(path):
        raise FileNotFoundError("checkpoint not found: %s" % path)
    state = load_checkpoint(path)
    state = state.get("state_dict", state)
    for key in drop:
        state.pop(key, None)
    result = model.load_state_dict(state, strict=False)

    def summary(keys):
        groups = {}
        for key in keys:
            group = ".".join(key.split(".")[:2])
            groups[group] = groups.get(group, 0) + 1
        return ", ".join("%s (%d)" % item for item in sorted(groups.items())) or "none"
    print("Loaded %s: %d tensors | missing from checkpoint: %s | unused: %s" % (
        os.path.basename(path), len(state) - len(result.unexpected_keys),
        summary(result.missing_keys), summary(result.unexpected_keys)))
    return result


def build_model(dataset):
    sed_model = MMER(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_mics=config.num_mics,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    return SEDWrapper(sed_model = sed_model, config = config, dataset = dataset)


def load_split(folder, filename, eval_mode):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found -- build it with prepare_data.py (see MER_mac_tutorial.ipynb)" % path)
    return Emotion_DataLoader(dataset = np.load(path, allow_pickle = True), config = config, eval_mode = eval_mode)


def experiment_dir():
    return os.path.join(config.workspace, "results_"+config.dataset_type+"_batch_size"+str(config.batch_size)+"_num_mics"+str(config.num_mics)+"_"+config.mode_type+"_seed"+str(config.random_seed), config.exp_name)


def weight_average():
    folder = config.wa_folder or os.path.join(experiment_dir(), "checkpoint", "checkpoints")
    model_files = sorted(f for f in os.listdir(folder) if f.endswith(".ckpt"))
    if not model_files:
        raise FileNotFoundError("no .ckpt files in %s" % folder)
    print("Averaging %d checkpoints from %s" % (len(model_files), folder))
    model_ckpt = []
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(folder, model_file)
        model_ckpt.append(load_checkpoint(model_file)["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched %s %s" % (model_ckpt_key.shape, model_ckpt[0][key].shape)
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)
    print("Saved", config.wa_model_path)

def esm_test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)

    if config.dataset_type == "RAVDESS":
        print("Test on RAVDESS")
        test_set = np.load(os.path.join(config.dataset_path, "ravdess_test.npy"), allow_pickle = True)
    elif config.dataset_type == "IEMOCAP":
        print("Test on IEMOCAP")
        test_set = np.load(os.path.join(config.dataset_path, "iemocap_test.npy"), allow_pickle = True)

    eval_dataset = Emotion_DataLoader(
            dataset = test_set,
            config = config,
            eval_mode = True
        )
    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=True,
        gpus = device_num,
        max_epochs = config.max_epoch,
        auto_lr_find = True,
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_models = []
    for esm_model_path in config.esm_model_pathes:
        sed_model = MMER(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        sed_wrapper = SEDWrapper(
            sed_model = sed_model,
            config = config,
            dataset = eval_dataset
        )
        ckpt = torch.load(esm_model_path, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        sed_wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        sed_models.append(sed_wrapper)

    model = Ensemble_SEDWrapper(
        sed_models = sed_models,
        config = config,
        dataset = eval_dataset
    )
    trainer.test(model, datamodule=audioset_data)


def evaluate(model, eval_dataset):
    """Run trainer.test of `model` on one test split; returns the logged metrics (acc, CI_low, CI_high)."""
    _, device_num = resolve_devices()
    data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer, _ = make_trainer(
        deterministic=True,
        max_epochs = config.max_epoch,
        sync_batchnorm = True,
        enable_checkpointing = False,
        num_sanity_val_steps = 0,
        gradient_clip_val=1.0,
        logger = False
    )
    return trainer.test(model, datamodule=data)


def test():
    print("Test on", config.dataset_type)
    eval_dataset = load_split(config.dataset_path, TEST_FILES[config.dataset_type], eval_mode = True)
    model = build_model(eval_dataset)
    if config.resume_checkpoint_test is not None:
        load_weights(model, config.resume_checkpoint_test, drop = ["sed_model.head.weight", "sed_model.head.bias"])
    return evaluate(model, eval_dataset)


def test_ace():
    # each model is evaluated on <ace_path>/<mic position>/<room>/<test file>, rooms and models from config.py
    rooms = [(mp, room) for mp in config.ace_mic_positions for room in config.ace_rooms]
    models = config.ace_models or {config.resume_checkpoint_test: config.num_mics}
    for model_path, spec in models.items():
        num_mics, mode_type = spec if isinstance(spec, (tuple, list)) else (spec, config.mode_type)
        config.num_mics, config.mode_type, config.resume_checkpoint_test = num_mics, mode_type, model_path
        print('MODEL: ', model_path, '| num_mics:', num_mics, '| mode:', mode_type)
        model = None
        rows = []
        for mp, room in rooms:
            eval_dataset = load_split(os.path.join(config.ace_path, mp, room), TEST_FILES[config.dataset_type], eval_mode = True)
            if model is None:
                model = build_model(eval_dataset)
                load_weights(model, model_path, drop = ["sed_model.head.weight", "sed_model.head.bias"])
            res = evaluate(model, eval_dataset)[0]
            rows.append({"mic_position": mp, "room": room, "acc": res["acc"],
                         "CI_low": res.get("CI_low"), "CI_high": res.get("CI_high")})
            print('NUM MICS: ', config.num_mics, '| ROOM: ', mp, room, '| acc: %.4f' % res["acc"])
        df = pd.DataFrame(rows)
        os.makedirs(config.ace_results_dir, exist_ok=True)
        name = os.path.basename(os.path.dirname(os.path.abspath(model_path))) + "_" + os.path.splitext(os.path.basename(model_path))[0]
        out = os.path.join(config.ace_results_dir, name + ".csv")
        df.to_csv(out, sep='|', index=False)
        print(df.to_string(index=False))
        print("Wrote", out)

def train():
    exp_dir = experiment_dir()
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    if config.debug == False and config.use_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger()
        # start a new wandb run to track this script
        wandb_proj_name = " " + config.mode_type
        if config.audio == True and config.num_mics == 1 and config.multimodal == False:
            wandb_proj_name = "Single-Channel Audio HTSAT " + config.mode_type
        elif config.audio == True and config.num_mics == 1 and config.video ==True:
            wandb_proj_name = "Multimodal Video & Single-Channel Audio " + config.mode_type
        elif config.audio == True and config.num_mics > 1 and config.video ==True:
            wandb_proj_name = "Multimodal Video & Multi-Channel REV Audio " + config.mode_type
        elif config.audio == True and config.num_mics > 1:
            wandb_proj_name = "Multi-Channel Audio " + config.mode_type
        elif config.video ==True and config.multimodal == False:
             wandb_proj_name = "Video pretrained " + config.dataset_type
        wandb.init(
            # set the wandb project where this run will be logged
            project="HTS-AT "+config.dataset_type+ wandb_proj_name, #+ " Simulated dataset", | change the project name as you like
            # track hyperparameters and run metadata
            config={
            "architecture": "AduioSet fine-tune" + wandb_proj_name+"_"+config.mode_type,
            "Fine-tuned checkpoint":config.resume_checkpoint,
            "pretrained": True,
            "video":config.video,
            "multimodal":config.multimodal,
            "audio":config.audio,
            "video_model":config.video_model,
            "dataset": config.dataset_type,
            "Type": config.mode_type,
            "num_mics": config.num_mics,
            "dataset path": config.dataset_path,
            "num_classes": config.classes_num,
            "lr_rate":config.lr_rate,
            "loss_type":config.loss_type,
            "time_drop": config.time_drop,
            "time_stripes": config.time_stripes,
            "freq_drop": config.freq_drop,
            "freq_stripes": config.freq_stripes,
            "clip_samples":config.clip_samples,
            "sec":config.sec,
            "mel bins":config.mel_bins,
            "win_length":config.window_size,
            "hopsize":config.hop_size,
            "n_fft":config.window_size,
            "lr_scheduler_epoch":config.lr_scheduler_epoch,
            "seed":config.random_seed,
            "epochs": config.max_epoch,
            "early_stopping_patience":config.es_patience,
            "batch_size":config.batch_size,
            "accumulate_grad_batches": getattr(config, "accumulate_grad_batches", 1),
            "accelerator": resolve_devices()[0],
            "learning_rate": config.learning_rate,
            "enable_tscam":config.enable_tscam,
            "htsat_window_size": config.htsat_window_size,
            "htsat_spec_size": config.htsat_spec_size,
            "htsat_patch_size": config.htsat_patch_size,
            "htsat_stride": config.htsat_stride,
            "htsat_num_head": config.htsat_num_head,
            "htsat_dim": config.htsat_dim,
            "htsat_depth": config.htsat_depth
            }
        )
        wandb.run.log_code(".")
    elif config.debug == False:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir = exp_dir, name = "logs") # metrics.csv per run
    else:
        logger = False

    train_file, eval_file = TRAIN_FILES[config.dataset_type]
    print("Using", config.dataset_type)
    dataset = load_split(config.dataset_path, train_file, eval_mode = False)
    eval_dataset = load_split(config.dataset_path, eval_file, eval_mode = True)

    # set exp folder
    if not config.debug:
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)

    _, device_num = resolve_devices()
    data = data_prep(dataset, eval_dataset, device_num)

    checkpoint_callback = ModelCheckpoint(
            dirpath = os.path.join(checkpoint_dir, "checkpoints"),
            monitor = "acc",
            filename='l-{epoch:d}-{acc:.3f}',
            save_top_k = 8,
            mode = "max"
        )
    early_stopping_callback = EarlyStopping(
            monitor = "val_loss",
            patience = config.es_patience, # 25 for SIM data RAVDESS, 12 for SIM data IEMOCAP
            verbose = True,
            mode = "min"
        )
    trainer, _ = make_trainer(
        deterministic=False,
        default_root_dir = checkpoint_dir,
        max_epochs = config.max_epoch,
        sync_batchnorm = True,
        callbacks = [checkpoint_callback,early_stopping_callback],
        num_sanity_val_steps = 0,
        gradient_clip_val=1.0,
        accumulate_grad_batches = getattr(config, "accumulate_grad_batches", 1),
        logger=logger
    )
    model = build_model(dataset)
    if config.resume_checkpoint is not None:
        if not os.path.exists(config.resume_checkpoint):
            raise FileNotFoundError("resume_checkpoint not found: %s -- download it with "
                                    "prepare_data.download_audioset_checkpoint(), or set it to None" % config.resume_checkpoint)
        load_weights(model, config.resume_checkpoint, drop = ["sed_model.head.weight", "sed_model.head.bias",
                                                              "sed_model.tscam_conv.weight", "sed_model.tscam_conv.bias"])
    elif config.swin_pretrain_path is not None: # train with pretrained model
        ckpt = load_checkpoint(config.swin_pretrain_path)
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(model.state_dict())

        for key in model_params:
            m_key = key.replace("sed_model.", "")
            if m_key in ckpt:
                if m_key == "patch_embed.proj.weight":
                    ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        model.load_state_dict(ckpt, strict = False)
        model_params = dict(model.named_parameters())
    trainer.fit(model, data)


def main():
    parser = argparse.ArgumentParser(description="HTS-AT")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_esm_test = subparsers.add_parser("esm_test")
    parser_wa = subparsers.add_parser("weight_average")
    parser_test_ace = subparsers.add_parser("test_ace")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(config.random_seed)

    if config.debug == True: # for debug and code check
        test_ace()
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "esm_test":
        esm_test()
    elif args.mode == "weight_average":
        weight_average()
    elif args.mode =="test_ace":
        test_ace()
    else:
        raise Exception("Error Mode!")


if __name__ == '__main__':
    main()

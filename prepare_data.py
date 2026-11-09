"""Build the RAVDESS audio-visual datasets that main.py trains and tests on -- on any machine, incl. a Mac.

This is prep_ravdess.ipynb (first paper) adapted to this repo's MER setup:
  * 8 emotion classes, target = RAVDESS emotion code - 1 (prep_ravdess merged neutral+calm into 7)
  * each row is [{"wav_name", "video_name", "waveform", "target", ...}] and a split is saved as an (N, 1) object
    array -- the layout prep_ravdess.ipynb wrote, hence `dataset[p][0]` in data_generator.py
  * actor split (paper Sec. IV-A): the int(n*0.8) / int(n*0.1) / rest arithmetic of prep_ravdess.ipynb applied
    to actors instead of files: 24 actors -> 19 / 2 / 3, no actor in two splits (3 test actors = 180 clips,
    the counts behind the paper's Fig. 4)
  * reverberant 3-mic audio (paper Sec. IV-A): every clean utterance gets its own simulated room, RIRs from
    pyroomacoustics on the CPU instead of gpuRIR (CUDA only), plus AR(1) noise at 20 dB SNR
  * ACE test sets for `main.py test_ace`: test utterances convolved with the real ACE "Mobile" phone RIRs
  * video: each clip's 8 sampled frames pre-extracted into data_generator's cache

Everything:   python prepare_data.py --download --checkpoint --rev --video-cache [--download-ace]
Quick check:  python prepare_data.py --actors 1-3 --download --checkpoint --rev --video-cache
Paths come from config.py (data_root, dataset_path, video_path, ace_path, video_cache_dir, resume_checkpoint).
"""
import argparse
import functools
import glob
import hashlib
import json
import multiprocessing
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

import config

RAVDESS_RECORD = "1188976" # zenodo.org/records/1188976 (CC BY-NC-SA 4.0)
ACE_RECORD = "6257551" # zenodo.org/records/6257551, the ACE challenge corpus
ACE_MOBILE_ARCHIVE = "ACE_Corpus_RIRN_Mobile.tbz2" # 1.6 GB: RIRs + noise of the 3-mic mobile phone
AUDIOSET_CKPT_GDRIVE_ID = "1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv" # HTSAT_AudioSet_Saved_1.ckpt (htsat_esc_training.ipynb)

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"] # target = code - 1
SPLIT_FILES = {"train": "ravdess_av_train.npy", "eval": "ravdess_av_eval.npy", "test": "ravdess_av_test.npy"}
SPLIT_IDS = {"train": 0, "eval": 1, "test": 2}

# ACE corpus room folder -> room folder name used by config.ace_rooms; ACE room config "1"/"2" -> "mp1"/"mp2"
ACE_ROOMS = {
    "Lecture_Room_1": "Lecture_room_1_508", "Lecture_Room_2": "Lecture_room_2_403a", "Building_Lobby": "lobby",
    "Meeting_Room_2": "Meeting room_2_611", "Meeting_Room_1": "Meeting_room_1_503",
    "Office_1": "Office_1_502", "Office_2": "Office_2_803",
}
ACE_POSITIONS = {"1": "mp1", "2": "mp2"}

# room simulation, paper Sec. IV-A
SIM = dict(
    fs=16000, num_mics=3,
    room_size=(3.0, 8.0), max_aspect=1.6, room_height=2.9, # length & width ~ U(3, 8) m, ratio <= 1.6
    t60=(0.5, 0.85), # Sabine T60 target (gpuRIR's beta_SabineEstimation does the same)
    src_height=1.75, mic_height=1.6, wall_margin=0.5, # source >= 0.5 m from the walls
    min_dist=0.2, # source-mic distance ~ U(0.2, d_c), d_c = critical distance
    snr_db=20.0, noise_ar=0.9, # spatially-white AR(1) noise (coefficient not given in the paper)
    peak=0.9, # every multi-channel example is peak-normalised to this
)

RAVDESS_DIR = os.path.join(config.data_root, "RAVDESS")
RAW_DIR = os.path.join(RAVDESS_DIR, "raw")
AUDIO_DIR = os.path.join(RAVDESS_DIR, "audio")
VIDEO_DIR = config.video_path.rstrip(os.sep)
CLEAN_DATASET_PATH = os.path.join(config.data_root, "RAVDESS_AV_actor_split_clean")
ACE_RIR_DIR = os.path.join(config.data_root, "ACE", "rirs")


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


# ------------------------------------------------------------------------------------------------- downloads
def _zenodo_files(record):
    with urllib.request.urlopen("https://zenodo.org/api/records/%s" % record, timeout=60) as r:
        return {f["key"]: f for f in json.load(r)["files"]}


def _download_stream(url, tmp, size, bar):
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def _download_ranges(url, tmp, size, connections, bar, chunk=16 << 20):
    """Fetch [0, size) as parallel HTTP range requests into tmp. Returns False if the server ignores ranges."""
    with open(tmp, "wb") as f:
        f.truncate(size)

    def fetch(start):
        end = min(start + chunk, size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        return False
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError("short read")
                with open(tmp, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                bar.update(len(data))
                return True
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt) # transient errors / rate limiting (HTTP 429)

    with ThreadPoolExecutor(max_workers=connections) as ex:
        return all(ex.map(fetch, range(0, size, chunk)))


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, size=None, md5=None, connections=8):
    """Download url to dest (skipped if it exists with the right size); checks the md5 when given.
    Files over 64 MB use `connections` parallel range requests: Zenodo serves ~1.6 MB/s per connection
    (8 connections measured at 8.8 MB/s)."""
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with tqdm(total=size or 0, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
        if not (size and size > (64 << 20) and connections > 1 and _download_ranges(url, tmp, size, connections, bar)):
            bar.reset()
            _download_stream(url, tmp, size, bar)
    if size is not None and os.path.getsize(tmp) != size:
        raise IOError("incomplete download of %s" % dest)
    if md5 and _md5(tmp) != md5:
        os.remove(tmp)
        raise IOError("md5 mismatch for %s" % dest)
    os.replace(tmp, dest)
    return dest


def _zenodo_download(files, key, folder):
    meta = files[key]
    md5 = meta.get("checksum", "").split(":", 1)[-1] or None
    return download(meta["links"]["self"], os.path.join(folder, key), meta["size"], md5)


def parse_actors(spec):
    """'1-24' / '1,2,5' / [1, 2] -> sorted list of actor ids."""
    if isinstance(spec, (list, tuple, range)):
        return sorted(int(a) for a in spec)
    actors = set()
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        actors.update(range(int(lo), int(hi or lo) + 1))
    return sorted(actors)


def download_ravdess(actors=range(1, 25), audio_dir=AUDIO_DIR, video_dir=VIDEO_DIR, raw_dir=RAW_DIR, keep_zips=False):
    """Fetch RAVDESS speech from Zenodo: the audio zip (208 MB, all actors) and one ~550 MB video zip per actor.
    Keeps the audio-only .wav (03-01-*) and full audio-video .mp4 (01-01-*) files of `actors` in
    <audio_dir>/Actor_XX and <video_dir>/Actor_XX. Video zips are deleted after extraction unless keep_zips."""
    actors = parse_actors(actors)
    files = _zenodo_files(RAVDESS_RECORD)

    def have(folder, actor, pattern):
        return len(glob.glob(os.path.join(folder, "Actor_%02d" % actor, pattern))) >= 60

    def extract(zip_path, folder, prefix, suffix, wanted):
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if not (base.startswith(prefix) and base.endswith(suffix)):
                    continue
                actor = int(os.path.splitext(base)[0].split("-")[-1])
                if actor in wanted:
                    dest = os.path.join(folder, "Actor_%02d" % actor, base)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    missing = [a for a in actors if not have(audio_dir, a, "03-01-*.wav")]
    if missing:
        zip_path = _zenodo_download(files, "Audio_Speech_Actors_01-24.zip", raw_dir)
        extract(zip_path, audio_dir, "03-01-", ".wav", set(missing))
    for actor in actors:
        if have(video_dir, actor, "01-01-*.mp4"):
            continue
        zip_path = _zenodo_download(files, "Video_Speech_Actor_%02d.zip" % actor, raw_dir)
        extract(zip_path, video_dir, "01-01-", ".mp4", {actor})
        if not keep_zips:
            os.remove(zip_path)
    print("RAVDESS actors %s: audio in %s, video in %s" % (actors, audio_dir, video_dir))
    return audio_dir, video_dir


def download_audioset_checkpoint(dest=config.resume_checkpoint):
    """HTSAT_AudioSet_Saved_1.ckpt (121 MB) from the HTS-AT authors' Google Drive, the file their
    htsat_esc_training.ipynb fine-tunes from. (The original runs used Saved_3, in the paper's Dropbox.)"""
    if not os.path.exists(dest):
        import gdown
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        gdown.download(id=AUDIOSET_CKPT_GDRIVE_ID, output=dest, quiet=False)
    print("AudioSet checkpoint:", dest)
    return dest


def download_ace_rirs(rir_dir=ACE_RIR_DIR, raw_dir=os.path.join(config.data_root, "ACE", "raw"), keep_archive=False):
    """Fetch the ACE 'Mobile' archive (1.6 GB, mostly noise recordings) and extract only its RIR files
    to <rir_dir>/Mobile/<room>/<config>/. Returns rir_dir."""
    if glob.glob(os.path.join(rir_dir, "Mobile", "*", "*", "*RIR*.wav")):
        return rir_dir
    archive = _zenodo_download(_zenodo_files(ACE_RECORD), ACE_MOBILE_ARCHIVE, raw_dir)
    extract_ace_rirs(archive, rir_dir)
    if not keep_archive:
        os.remove(archive)
    return rir_dir


def extract_ace_rirs(archive, rir_dir=ACE_RIR_DIR):
    """Stream through an ACE_Corpus_RIRN_*.tbz2 and write only the *_RIR.wav members."""
    count = 0
    with tarfile.open(archive, "r|bz2") as tf:
        for member in tf:
            base = os.path.basename(member.name)
            if member.isfile() and base.lower().endswith(".wav") and "_RIR" in base:
                dest = os.path.join(rir_dir, *member.name.split("/"))
                if not os.path.abspath(dest).startswith(os.path.abspath(rir_dir) + os.sep):
                    continue # never write outside rir_dir
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with tf.extractfile(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    print("extracted %d ACE RIR files to %s" % (count, rir_dir))
    return rir_dir


# --------------------------------------------------------------------------------------- listing and the split
def list_ravdess(audio_dir=AUDIO_DIR, video_dir=VIDEO_DIR, actors=None):
    """One record per speech utterance: wav path, wav_name, video_name (relative to config.video_path), target."""
    records, missing_video = [], []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "Actor_*", "03-01-*.wav"))):
        name = os.path.basename(wav)
        modality, channel, emotion, intensity, statement, repetition, actor = os.path.splitext(name)[0].split("-")
        actor = int(actor)
        if actors is not None and actor not in actors:
            continue
        video_name = "Actor_%02d/01-%s.mp4" % (actor, os.path.splitext(name)[0][3:])
        if video_dir is not None and not os.path.exists(os.path.join(video_dir, video_name)):
            missing_video.append(video_name)
            continue
        records.append(dict(wav_path=wav, wav_name=name, video_name=video_name, target=int(emotion) - 1, actor=actor))
    if missing_video:
        print("skipped %d utterances without a video, e.g. %s" % (len(missing_video), missing_video[0]))
    return records


def actor_split(actors, seed=0, train=0.8, val=0.1):
    """prep_ravdess.ipynb's int(n*0.8) / int(n*0.1) / rest, over a seeded permutation of the actors."""
    actors = sorted(set(actors))
    order = [int(a) for a in np.random.RandomState(seed).permutation(actors)]
    n_train, n_val = int(len(actors) * train), int(len(actors) * val)
    if len(actors) >= 3: # tiny subsets: keep one actor each for validation and test
        n_val = max(n_val, 1)
        n_train = min(n_train, len(actors) - n_val - 1)
    return {"train": sorted(order[:n_train]), "eval": sorted(order[n_train:n_train + n_val]),
            "test": sorted(order[n_train + n_val:])}


# ---------------------------------------------------------------------------------------------- audio synthesis
def critical_distance(volume, t60):
    return 0.057 * np.sqrt(volume / t60)


def sample_room(rng, p=SIM):
    """Room, T60, source and mic positions drawn as described in the paper (Sec. IV-A)."""
    while True:
        length, width = rng.uniform(*p["room_size"], size=2)
        if max(length, width) / min(length, width) <= p["max_aspect"]:
            break
    room = np.array([length, width, p["room_height"]])
    t60 = rng.uniform(*p["t60"])
    d_c = critical_distance(np.prod(room), t60)
    m = p["wall_margin"]
    src = np.array([rng.uniform(m, length - m), rng.uniform(m, width - m), p["src_height"]])
    dz = p["src_height"] - p["mic_height"]
    mics = []
    while len(mics) < p["num_mics"]: # each mic at its own distance ~ U(0.2, d_c) and direction
        dist = rng.uniform(p["min_dist"], max(d_c, p["min_dist"] + 1e-3))
        r, phi = np.sqrt(max(dist ** 2 - dz ** 2, 0.0)), rng.uniform(0, 2 * np.pi)
        x, y = src[0] + r * np.cos(phi), src[1] + r * np.sin(phi)
        if 0.1 <= x <= length - 0.1 and 0.1 <= y <= width - 0.1:
            mics.append([x, y, p["mic_height"]])
    return dict(room=room, t60=t60, critical_distance=d_c, src=src, mics=np.array(mics))


def simulate_rirs(geometry, fs=SIM["fs"]):
    """(num_mics, n) RIRs of a shoebox room with the image-source method (pyroomacoustics)."""
    import pyroomacoustics as pra
    e_absorption, max_order = pra.inverse_sabine(geometry["t60"], geometry["room"])
    room = pra.ShoeBox(geometry["room"], fs=fs, materials=pra.Material(e_absorption), max_order=max_order,
                       air_absorption=False)
    room.add_source(geometry["src"])
    room.add_microphone_array(geometry["mics"].T)
    room.compute_rir()
    n = max(len(room.rir[m][0]) for m in range(len(room.rir)))
    return np.stack([np.pad(room.rir[m][0], (0, n - len(room.rir[m][0]))) for m in range(len(room.rir))])


def trim_to_direct_path(rirs, margin=16):
    """Drop the samples before the earliest direct-path peak (same shift on every channel)."""
    onset = int(min(np.argmax(np.abs(h)) for h in rirs))
    return rirs[:, max(0, onset - margin):]


def ar1_noise(rng, n, coef):
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -coef], rng.standard_normal(n))


def reverberate(clean, rirs, rng=None, snr_db=None, noise_ar=SIM["noise_ar"], peak=SIM["peak"]):
    """Convolve a clean utterance with each RIR (kept at the clean length), add noise at snr_db per channel,
    peak-normalise. Returns (channels, samples) float32."""
    from scipy.signal import fftconvolve
    rev = np.stack([fftconvolve(clean, h)[: len(clean)] for h in trim_to_direct_path(rirs)])
    if snr_db is not None:
        for c in range(len(rev)):
            noise = ar1_noise(rng, rev.shape[1], noise_ar)
            rev[c] += noise * np.sqrt(np.mean(rev[c] ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    scale = np.abs(rev).max()
    return (rev * (peak / scale) if scale > 0 else rev).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _load_clean(wav_path):
    import librosa
    y, _ = librosa.load(wav_path, sr=SIM["fs"], mono=True) # 48 kHz -> 16 kHz (soxr), as prep_ravdess.ipynb
    return y


@functools.lru_cache(maxsize=4)
def load_ace_rir(rir_file):
    import librosa
    rir, _ = librosa.load(rir_file, sr=SIM["fs"], mono=False) # ACE RIRs are 48 kHz
    return np.atleast_2d(rir)


def _row(task):
    """Worker: one dataset row. task = (record, kind, split, seed, rir_file)."""
    rec, kind, split, seed, rir_file = task
    clean = _load_clean(rec["wav_path"])
    row = {"wav_name": rec["wav_name"], "video_name": rec["video_name"], "target": rec["target"], "actor": rec["actor"]}
    if kind == "clean":
        row["waveform"] = clean[np.newaxis].astype(np.float32)
    elif kind == "rev":
        rng = np.random.default_rng([seed, SPLIT_IDS[split], zlib.crc32(rec["wav_name"].encode())])
        geometry = sample_room(rng)
        row["waveform"] = reverberate(clean, simulate_rirs(geometry), rng, SIM["snr_db"])
        row.update(t60=float(geometry["t60"]), room=geometry["room"].tolist())
    elif kind == "ace":
        row["waveform"] = reverberate(clean, load_ace_rir(rir_file)[: SIM["num_mics"]])
    else:
        raise ValueError(kind)
    return row


def _pmap(fn, tasks, workers, desc):
    if workers <= 1:
        return [fn(t) for t in tqdm(tasks, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(tqdm(ex.map(fn, tasks, chunksize=8), total=len(tasks), desc=desc))


def save_rows(rows, path):
    """Save rows as an (N, 1) object array: dataset[i][0] is the row dict (the layout data_generator reads)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.empty((len(rows), 1), dtype=object)
    for i, row in enumerate(rows):
        arr[i, 0] = row
    np.save(path, arr, allow_pickle=True)


def class_counts(rows):
    counts = np.bincount([r["target"] for r in rows], minlength=len(EMOTIONS))
    return ", ".join("%s %d" % (e, c) for e, c in zip(EMOTIONS, counts))


def build_dataset(records, split, out_dir=config.dataset_path, kind="rev", seed=0, workers=None):
    """Write <out_dir>/ravdess_av_{train,eval,test}.npy for an actor split (dict of split -> actors).
    kind: "rev" (3-mic simulated reverberation + noise) or "clean" (the clean utterance, 1 channel)."""
    workers = default_workers() if workers is None else workers
    for name, actors in split.items():
        subset = [r for r in records if r["actor"] in actors]
        rows = _pmap(_row, [(r, kind, name, seed, None) for r in subset], workers, "%s %s" % (kind, name))
        save_rows(rows, os.path.join(out_dir, SPLIT_FILES[name]))
        print("%-5s actors %-28s %4d clips | %s" % (name, actors, len(rows), class_counts(rows)))
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"split": split, "kind": kind, "seed": seed, "simulation": SIM if kind == "rev" else None}, f, indent=2)
    return out_dir


def build_ace_sets(test_records, rir_dir=ACE_RIR_DIR, out_root=config.ace_path, workers=None):
    """<out_root>/<mp1|mp2>/<room>/ravdess_av_test.npy: the test utterances convolved with each ACE Mobile RIR."""
    workers = default_workers() if workers is None else workers
    built = 0
    for ace_room, room in ACE_ROOMS.items():
        for ace_config, position in ACE_POSITIONS.items():
            found = sorted(glob.glob(os.path.join(rir_dir, "Mobile", ace_room, ace_config, "*RIR*.wav")))
            if not found:
                print("no ACE RIR for %s/%s under %s -- skipped" % (ace_room, ace_config, rir_dir))
                continue
            rows = _pmap(_row, [(r, "ace", "test", 0, found[0]) for r in test_records], workers,
                         "ACE %s %s" % (position, room))
            save_rows(rows, os.path.join(out_root, position, room, SPLIT_FILES["test"]))
            built += 1
    print("built %d ACE test sets in %s" % (built, out_root))
    return out_root


# --------------------------------------------------------------------------------------------------- video cache
def _cache_clip(task):
    from data_generator import load_video_clip
    video_file, cache_file, time_samples = task
    load_video_clip(video_file, time_samples, cache_file)


def build_video_cache(records, video_dir=VIDEO_DIR, cache_dir=None, time_samples=None, workers=None):
    """Decode every clip once into config.video_cache_dir (the frames the training loader would compute)."""
    from data_generator import video_cache_file
    cache_dir = cache_dir or config.video_cache_dir or os.path.join(RAVDESS_DIR, "video_cache")
    time_samples = time_samples or config.video_time_samples
    workers = default_workers() if workers is None else workers
    names = sorted({r["video_name"] for r in records})
    tasks = [(os.path.join(video_dir, n), video_cache_file(cache_dir, n, time_samples), time_samples) for n in names]
    tasks = [t for t in tasks if not os.path.exists(t[1])]
    if tasks:
        _pmap(_cache_clip, tasks, workers, "video cache")
    print("video cache: %d clips in %s" % (len(names), cache_dir))
    if config.video_cache_dir is None:
        print("(config.video_cache_dir is None on this machine: set it to that folder to train from the cache)")
    return cache_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actors", default="1-24", help="actor ids, e.g. 1-24 or 1,2,3")
    ap.add_argument("--download", action="store_true", help="fetch RAVDESS speech audio + video from Zenodo")
    ap.add_argument("--checkpoint", action="store_true", help="fetch HTSAT_AudioSet_Saved_1.ckpt")
    ap.add_argument("--rev", action="store_true", help="build the reverberant 3-mic splits (config.dataset_path)")
    ap.add_argument("--clean", action="store_true", help="also build clean 1-channel splits")
    ap.add_argument("--download-ace", action="store_true", help="fetch the ACE Mobile RIRs (1.6 GB archive)")
    ap.add_argument("--ace-archive", help="use an ACE_Corpus_RIRN_Mobile.tbz2 you already have")
    ap.add_argument("--video-cache", action="store_true", help="pre-extract the video frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    actors = parse_actors(args.actors)
    if args.checkpoint:
        download_audioset_checkpoint()
    if args.download:
        download_ravdess(actors)
    records = list_ravdess(actors=actors)
    split = actor_split(sorted({r["actor"] for r in records}), seed=args.seed)
    print("actor split:", split)
    if args.rev:
        build_dataset(records, split, config.dataset_path, "rev", args.seed, args.workers)
    if args.clean:
        build_dataset(records, split, CLEAN_DATASET_PATH, "clean", args.seed, args.workers)
    if args.ace_archive or args.download_ace:
        rir_dir = extract_ace_rirs(args.ace_archive) if args.ace_archive else download_ace_rirs()
        build_ace_sets([r for r in records if r["actor"] in split["test"]], rir_dir, config.ace_path, args.workers)
    if args.video_cache:
        build_video_cache(records, workers=args.workers)


if __name__ == "__main__":
    main()

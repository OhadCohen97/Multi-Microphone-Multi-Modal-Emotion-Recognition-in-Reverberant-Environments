import os
import sys

# ---------------------------------------------------------------------------------------------------------
# Where things live. Defaults are relative to this repo, in the layout prepare_data.py builds
# (see MER_mac_tutorial.ipynb), so the code runs as-is on any machine. On the BIU cluster, point these at the
# existing folders or set MER_WORKSPACE / MER_DATA_ROOT.
# ---------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
workspace = os.environ.get("MER_WORKSPACE", ROOT) # outputs: experiments (results_*), averaged checkpoints, CI results
data_root = os.environ.get("MER_DATA_ROOT", os.path.join(ROOT, "data")) # inputs: datasets, videos, pretrained ckpt

exp_name = "exp_finetune_actor_split" # the saved ckpt prefix name of the model

multimodal = True
if multimodal == True:
    audio = True
    video = True
else: #chose modality
    audio = True
    video = False
video_path = os.path.join(data_root, "RAVDESS", "videos") + os.sep # <video_path><video_name> is a sample's .mp4

num_mics = 3
if num_mics == 1:
    mode_type = 'avg_mel' # vanilla HTS-AT
else:
    mode_type = 'avg_mel' # "sum" | "avg_mel"

dataset_type = "RAVDESS" # "IEMOCAP" | "RAVDESS" | "CREMA-D"
loss_type ="clip_ce" # or "clip_bce" for binary cross-entropy

# weight average: folder whose checkpoints get averaged. None = the current experiment's checkpoints
# (<workspace>/results_.../<exp_name>/checkpoint/checkpoints). Runs made before this layout saved them under
# .../checkpoint/None/version_None/checkpoints -- set wa_folder to that folder for those.
wa_folder = None
# weight average output filename
wa_model_path = os.path.join(workspace, "HTSAT_" + dataset_type + "_Saved_x.ckpt")

# fine-tune from an HTS-AT AudioSet checkpoint, set None to train from scratch. The paper's runs used
# HTSAT_AudioSet_Saved_3.ckpt (in the paper's Dropbox), used when it is in model/. Otherwise
# HTSAT_AudioSet_Saved_1.ckpt (121 MB), which prepare_data.download_audioset_checkpoint() fetches from the
# HTS-AT authors' Google Drive.
_paper_checkpoint = os.path.join(ROOT, "model", "HTSAT_AudioSet_Saved_3.ckpt")
resume_checkpoint = _paper_checkpoint if os.path.exists(_paper_checkpoint) else \
    os.path.join(data_root, "checkpoints", "HTSAT_AudioSet_Saved_1.ckpt")
# evaluate a single model for testing (`test`, and `test_ace` when ace_models is None)
resume_checkpoint_test = wa_model_path
debug = False # True: no W&B and no results folder -- and main.py also runs test_ace() before the chosen mode

# ---------------------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------------------
accelerator = "auto" # "auto": CUDA GPUs if any, else the Apple-silicon GPU (MPS), else CPU | "gpu" | "mps" | "cpu"
use_wandb = True # False: log to CSV inside the experiment folder (no W&B account needed)

batch_size = 32 # effective batch per optimizer step = per-step batch x devices x accumulate_grad_batches
# An Apple-silicon GPU can't hold 32 video clips at once (~41 GB): run steps of 8 and accumulate the
# gradients of 4 of them, which keeps the effective batch at 32.
accumulate_grad_batches = 4 if sys.platform == "darwin" else 1
learning_rate = 1e-3 # 1e-4 also workable
max_epoch = 500
# macOS: 0, load in the main process. With the video cache an item takes ~5 ms, far less than a training step,
# while DataLoader workers have to be re-spawned (macOS can't fork them) and cost 20-40 s per epoch here.
num_workers = 8 if sys.platform != "darwin" else 0

lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

video_model = None # None for r2dplus, 'x3d_s' for facebookresearch/pytorchvideo model
video_augment = True
video_time_samples = 8 # frames sampled per clip (16 was used for clean data, 8 for REV data)
# Cache each clip's sampled + resized frames (uint8 .npy, 1.2 MB) so every .mp4 is decoded only once.
# Bit-identical to decoding every time; None = decode on every access (the original behaviour).
video_cache_dir = os.path.join(data_root, "RAVDESS", "video_cache") if sys.platform == "darwin" else None
confusion_matrix = False # for plotting confusion matrix in sed_model.py

if dataset_type == "RAVDESS":
    random_seed = 1 # 19970318 970131 12412 127777 1009 34047
    dataset_path = os.path.join(data_root, "RAVDESS_AV_actor_split_REV_1440_500_850") # the dataset path
    classes_num = 8 # 7 classes for the first work. change data accordingly
    sec = 4
    es_patience = 12
    time_drop = 64
    time_stripes = 4
    freq_drop = 8
    freq_stripes = 2
    htsat_depth = [2,2,6,2] # for htsat hyperparamater
elif dataset_type == "IEMOCAP":
    random_seed = 19970318 # 19970318 970131 12412 127777 1009 34047
    dataset_path = os.path.join(data_root, "generated_IEMOCAP_BIUREVgen_sim") # the dataset path
    classes_num = 4
    sec = 4
    es_patience = 25
    time_drop = 64
    time_stripes = 4
    freq_drop = 8
    freq_stripes = 2
    htsat_depth = [2,2,6,2] # for htsat hyperparamater
else: # CREMA-D
    random_seed = 127777 # 19970318 970131 12412 127777 1009 34047
    dataset_path = os.path.join(data_root, "generated_CREMAD_BIUREVgen_sim") # the dataset path
    classes_num = 6
    sec = 2.5
    es_patience = 25
    time_drop = 64
    time_stripes = 3
    freq_drop = 16
    freq_stripes = 2
    htsat_depth = [2,2,6,2] # for htsat hyperparamater

# ---------------------------------------------------------------------------------------------------------
# test_ace: the test actors convolved with the real RIRs of the ACE "Mobile" (3-mic phone) recordings.
# Layout <ace_path>/<mic position>/<room>/ravdess_av_test.npy (prepare_data.py builds it).
# ---------------------------------------------------------------------------------------------------------
ace_path = os.path.join(data_root, "ACE", "RAVDESS_actor_split")
ace_mic_positions = ["mp1", "mp2"]
ace_rooms = ["Lecture_room_1_508", "Lecture_room_2_403a", "lobby", "Meeting room_2_611",
             "Meeting_room_1_503", "Office_1_502", "Office_2_803"]
# models to evaluate: {checkpoint: num_mics} or {checkpoint: (num_mics, mode_type)}.
# None = {resume_checkpoint_test: num_mics} with the mode_type above.
ace_models = None
ace_results_dir = os.path.join(workspace, "CI_results")

# for signal processing
sample_rate = 16000
clip_samples = int(sample_rate * sec) # 4 sec for iemocap and ravdess, 2.5 sec for crema-d
window_size = 1024
hop_size = 160
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)

# for model's design
enable_tscam = True # enbale the token-semantic layer

# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96
# htsat_depth = [1,1,3,1]

swin_pretrain_path = None
# ".../swin_tiny_c24_patch4_window8_256.pth"

# for ensemble test

# ensemble_checkpoints = []
# ensemble_strides = []

# esm_model_pathes = [
#     ".../HTSAT_AudioSet_Saved_1.ckpt",
#     ".../HTSAT_AudioSet_Saved_2.ckpt",
#     ".../HTSAT_AudioSet_Saved_3.ckpt",
#     ".../HTSAT_AudioSet_Saved_4.ckpt",
#     ".../HTSAT_AudioSet_Saved_5.ckpt",
#     ".../HTSAT_AudioSet_Saved_6.ckpt"
# ]


fl_local = False # indicate if we need to use this dataset for the framewise detection

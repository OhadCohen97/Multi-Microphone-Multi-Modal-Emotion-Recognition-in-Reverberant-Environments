# Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments - Accepted to EUSIPCO 2025

🔥 **Official repository PyTorch-Lightning code for the paper:**  
**"Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments"** (EUSIPCO 2025)  
This work is an extension of our previous paper, [**"Multi-Microphone Speech Emotion Recognition using the Hierarchical Token-semantic Audio Transformer Architecture"**](https://ieeexplore.ieee.org/document/10887873) (ICASSP 2025). It adds multi-modal learning to further improve robustness.  
📄 [Read the Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/11226096) | [arXiv](https://arxiv.org/pdf/2409.09545)  

🍎 **New:** the code now also runs on Apple-silicon Macs (see [Running on macOS](#-running-on-macos-apple-silicon)). Note that the data pipeline used there processes the RIRs differently from the paper (see [Preprocess Data](#preprocess-data)), so its results are not directly comparable to the paper's.


---

## 🔍 Overview
Human emotions are conveyed through **speech and facial expressions**, making **multi-modal emotion recognition (MER)** crucial for robust emotion classification. This work introduces a **multi-microphone and multi-modal system** combining:

- **HTS-AT Transformer** for multi-channel audio processing
- **R(2+1)D ResNet CNN** for video-based emotion recognition
- **Late fusion concatenation** to combine audio & video features

### 🔮 Key Features
✔️ **Multi-Microphone Audio Processing**: Robust against reverberation  
✔️ **Multi-Modal Learning**: Combines speech and facial cues  
✔️ **Tested on RAVDESS convolved with Real-World RIRs** (ACE Database)  
✔️ **Pretrained Models Available** for Fine-Tuning and testing  
✔️ **Runs on CUDA GPUs and Apple-silicon Macs (MPS)**

---

## 📸 Model Architecture
Our approach consists of **two main components**:
1. **HTS-AT Transformer (Audio Modality)**:
   - Processes **multi-channel mel-spectrograms**
   - Uses **Patch-Embed Summation & Averaging strategies**
   - Extracts **deep features** for robust emotion classification
2. **R(2+1)D CNN (Video Modality)**:
   - Extracts spatiotemporal features from facial expressions
   - Pretrained on **Kinetics dataset**, fine-tuned for MER
3. **Feature Fusion & Classification**:
   - Late fusion via **concatenation of extracted embeddings**
   - Fully connected layers for final emotion classification

| Multi-channel Multi-modal Architecture | Multi-Microphone Audio | Video-Based Recognition |
|----------------------|------------------|----------------------|
| ![Model](images/MER.png) | ![The extended HTS-AT](images/audio.png) | ![R2+1D](images/video.png) |

---

## 🔧 Getting Started

### **1️⃣ Installation**
The base way to run the code is with Docker Container.
#### **Pull Docker Image**
```bash
docker pull ohadico97/mer:v1
```
After having the image, run:

```bash
docker run -it --gpus all --shm-size 20G -v $HOME:$HOME --name <container name> ohadico97/mer:v1
```
#### **Clone the Repository**
```bash
git clone https://github.com/OhadCohen97/Multi-Microphone-Multi-Modal-Emotion-Recognition-in-Reverberant-Environments.git
cd Multi-Microphone-Multi-Modal-Emotion-Recognition-in-Reverberant-Environments
```
#### **Use the Virtual Environment**
```bash
Python 3.8.13 ('base')
```

---

### **2️⃣ Dataset Setup**
The previous paper evaluated the models on three datasets: **RAVDESS**, **IEMOCAP**, and **CREMA-D**. In this work, we focus exclusively on the **RAVDESS** dataset. The **training and validation splits** are reverberated synthetically using the **'gpuRIR' Python library**, while the **test sets** are reverberated with real-world ACE RIRs recorded in various acoustic environments. You can choose which modality to fine-tune using the ```multimodal``` flag in ```config.py``` (see [Choosing the model](#choosing-the-model)).
- 🔗 [Dataset & Pre-trained Models](https://www.dropbox.com/scl/fo/px0lqliq9kc53tpv2smf2/AGnCptxxQBvGsigrih7urbU?rlkey=cxtlcus07oka82xvfruz8t5cw&st=2lm1tbt9&dl=0) has two things:
  - `HTSAT_AudioSet_Saved_3.ckpt`, the AudioSet checkpoint this work fine-tunes from. Put it in `model/`.
  - The first paper's clean audio splits of RAVDESS, IEMOCAP and CREMA-D.

**Note** that the first work fine-tuned on different amounts of samples and splits. Its RAVDESS splits are by file, not by actor: every actor appears in both training and test, and neutral and calm are merged into 7 classes. The Dropbox doesn't have this work's audio-visual, actor-split RAVDESS data or its ACE test sets. `prepare_data.py` builds both (see below).

`prepare_data.py` builds this layout from the public RAVDESS and ACE corpora (default paths in `config.py`, under `data/`):
```bash
data/
  │── RAVDESS_AV_actor_split_REV_1440_500_850/   # config.dataset_path
  │         │── ravdess_av_train.npy
  │         │── ravdess_av_eval.npy
  │         │── ravdess_av_test.npy
  │── RAVDESS_AV_actor_split_clean/               # --clean: the same splits, clean and single-channel
  │── ACE/RAVDESS_actor_split/                    # config.ace_path
  │         │── mp1/
  │         │     │── Lecture_room_1_508/ravdess_av_test.npy
  │         │     │── Lecture_room_2_403a/ravdess_av_test.npy
  │         │     │── lobby/ ... "Meeting room_2_611"/ Meeting_room_1_503/ Office_1_502/ Office_2_803/
  │         │── mp2/ ...
  │── RAVDESS/videos/Actor_XX/01-01-*.mp4         # config.video_path
  │── checkpoints/HTSAT_AudioSet_Saved_1.ckpt     # config.resume_checkpoint, unless model/HTSAT_AudioSet_Saved_3.ckpt exists
```
Each `.npy` is an `(N, 1)` object array; `dataset[i][0]` is a dict with `wav_name`, `video_name` (relative to `config.video_path`), `waveform` (`(mics, samples)`, 16 kHz), `target` (0-7) and `actor` (1-24). The reverberant sets also store each clip's simulated `t60` and `room`.

---

### **3️⃣ Training & Evaluation**
#### **Preprocess Data**
```bash
python3 prepare_data.py --download --checkpoint --rev --video-cache   # RAVDESS + AudioSet checkpoint, reverberant 3-mic splits
python3 prepare_data.py --clean                                       # optional: clean 1-channel splits (Clean single-ch. models)
python3 prepare_data.py --download-ace                                # optional: ACE "Mobile" RIRs (1.6 GB) -> test_ace sets
```
It adapts `prep_ravdess.ipynb` (first paper) to this work: 8 classes, actor split (19/2/3 actors), and one simulated room per utterance with the parameters of Sec. IV-A, using pyroomacoustics instead of gpuRIR so it also runs without CUDA. `MER_mac_tutorial.ipynb` runs the same steps, then training and testing, from a notebook.

**Different RIR processing.** The data it builds is close to the paper's, but its RIR processing differs, so results are not directly comparable to the paper's:
- **Training rooms:** simulated with pyroomacoustics instead of gpuRIR, with the room parameters of Sec. IV-A. Each microphone is placed on its own, at a random distance U(0.2, d<sub>c</sub>) and a random direction from the speaker. The paper doesn't give an array geometry.
- **Every RIR, simulated or ACE:** cut to start 1 ms before the direct-path peak (the same cut on every channel). The ACE RIRs are first resampled from 48 to 16 kHz.
- **After convolution:** the training clips get 20 dB noise, AR(1) with coefficient 0.9, which the paper doesn't give. Every clip is then peak-normalized to 0.9.
- **Actor split:** a seeded draw (`--seed 0`) with validation actors 1 and 4 and test actors 13, 16 and 22. It isn't the paper's split.
#### **Fine-tune the Models**
`config.resume_checkpoint` is the HTS-AT AudioSet checkpoint training starts from: the paper's `HTSAT_AudioSet_Saved_3.ckpt` (from the Dropbox) when it is in `model/`, otherwise `HTSAT_AudioSet_Saved_1.ckpt`, which `--checkpoint` downloads. `config.dataset_path` is the dataset to use.
To train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 main.py train
```
The 8 best checkpoints (validation accuracy) are saved in `results_.../<exp_name>/checkpoint/checkpoints/`. Average them into `config.wa_model_path`, the model `test` and `test_ace` evaluate by default:
```bash
python3 main.py weight_average
```
#### **Choosing the model**
Each model in the paper's Table 1 is one training run with these `config.py` settings:

| Table 1 model | `multimodal` | `audio` | `video` | `num_mics` | `mode_type` | data |
|---|---|---|---|---|---|---|
| Video | False | False | True | 3 | – | reverberant |
| Audio, Clean single-ch. | False | True | False | 1 | `avg_mel` | clean |
| Audio, Rev single-ch. | False | True | False | 1 | `avg_mel` | reverberant |
| Audio, Avg mel | False | True | False | 3 | `avg_mel` | reverberant |
| Audio, Sum PE | False | True | False | 3 | `sum` | reverberant |
| MER, Clean single-ch. | True | True | True | 1 | `avg_mel` | clean |
| MER, Rev single-ch. | True | True | True | 1 | `avg_mel` | reverberant |
| MER, Avg mel | True | True | True | 3 | `avg_mel` | reverberant |
| MER, Sum PE | True | True | True | 3 | `sum` | reverberant |

`audio`/`video` and `mode_type` are set in the `if`/`else` blocks below `multimodal` and `num_mics`. "Reverberant" is the default `dataset_path`; "clean" is `RAVDESS_AV_actor_split_clean`. Single-channel models use the first microphone.

Every run saves its checkpoints and results under the same file names in `MER_WORKSPACE` (default: this folder). Give each model its own folder, and use the same one for all four steps (`train`, `weight_average`, `test` and `test_ace`), e.g. `MER_WORKSPACE=runs/mer_sum python3 main.py train`. `MER_DATA_ROOT` (default: `data/`) is where the data is read from. Training logs to Weights & Biases; set `use_wandb = False` to log to CSV instead.
#### **Evaluate Performance**
On the test split of `config.dataset_path`:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py test
```
On every ACE room and mic position (`config.ace_rooms`, `config.ace_mic_positions`; `config.ace_models` evaluates several checkpoints). Accuracy and 75% confidence intervals are written to `config.ace_results_dir`. Run it on one GPU: the confidence intervals are only computed in single-device runs.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py test_ace
```
#### **Cross-validation over actors**
With only 3 test actors, the test accuracy depends on which actors were drawn. `cross_validation.py` splits the 24 actors into 8 folds, so each actor is tested once. Each fold has 3 test, 2 validation and 19 training actors, and trains the model set in `config.py` the same way `main.py train` does:
```bash
python3 cross_validation.py --plan      # print the folds
python3 cross_validation.py --ace       # train and test every fold, also on the ACE rooms (about 1 h per fold for MER on an M4 Max)
python3 cross_validation.py --summary   # summary of the folds finished so far
```
Running it again skips finished folds. Results per fold, per actor and (with `--ace`) per ACE room go to `<workspace>/results_.../<exp_name>_cv8/`.

### **🍎 Running on macOS (Apple silicon)**
The same code runs on a Mac, on its GPU (MPS). The model and training code are unchanged; only the data pipeline differs from the paper's (see the RIR processing notes in [Preprocess Data](#preprocess-data)). If you have the paper's datasets, point `config.dataset_path` and `config.ace_path` at them. Tested on an M4 Max (36 GB):
```bash
conda create -n mer python=3.12 -y && conda activate mer
pip install -r requirements-mac.txt
```
Then run the steps above without `CUDA_VISIBLE_DEVICES`, or open `MER_mac_tutorial.ipynb`. On a Mac:
- steps of 8 clips with gradients accumulated over 4 steps keep the effective batch of 32 (32 clips at once don't fit in memory); BatchNorm statistics are per 8 clips;
- R(2+1)D's convolutions run as equivalent 2-D convolutions on MPS (same weights, 2.6x faster);
- every clip's 8 video frames are decoded once into `config.video_cache_dir` (identical tensors), and data loads in the main process (`num_workers = 0`);
- `prepare_data.py` simulates the training rooms with pyroomacoustics (paper parameters in `prepare_data.SIM`); the actor split is a seeded 19/2/3 draw.

On an M4 Max, an epoch on the full dataset takes ~2 min for the multimodal and video-only models (0.66 s per 8-clip step) and ~20 s audio-only. With early stopping, a full training run took 37–75 min for MER, 45 min video-only and 7–10 min audio-only. The data needs ~10 GB (6.6 GB of video); each training run saves up to 8 checkpoints of 746 MB.

---

## 📊 Results

![results](images/results_mer.png)
---

## 🏆 Citation
If you use this work, please cite:
```bibtex
@inproceedings{cohen2025mer,
  title={Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environment},
  author={Cohen, Ohad and Hazan, Gershon and Gannot, Sharon},
  booktitle={2025 33rd European Signal Processing Conference (EUSIPCO)},
  pages={940--944},
  year={2025},
  doi={10.23919/EUSIPCO63237.2025.11226096}
}
```
and the first paper:
```bibtex
@inproceedings{cohen2025ser,
  title={Multi-Microphone Speech Emotion Recognition Using the Hierarchical Token-Semantic Audio Transformer Architecture},
  author={Cohen, Ohad and Hazan, Gershon and Gannot, Sharon},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  doi={10.1109/ICASSP49660.2025.10887873}
}
```

---

## 🌟 Acknowledgments
This research was supported by the **European Union’s Horizon 2020 Program** and the **Audition Project, Data Science Program, Israel**.

---

## 👤 Contact
For questions or collaborations, feel free to reach out:  
📧 **ohad.cohen@biu.ac.il**


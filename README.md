# Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments - Submitted to EUSIPCO 2025

🔥 **Official repository PyTorch-Lightning code for the paper:**  
**"Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments"**  
This work is an extension of our previous paper "Multi-Microphone Speech Emotion Recognition using the Hierarchical Token-semantic Audio Transformer
Architecture", incorporating multi-modal learning to further improve robustness.  
📄 [Read the Paper](https://arxiv.org/pdf/2409.09545)  


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
The previous paper evaluated the models on three datasets: **RAVDESS**, **IEMOCAP**, and **CREMA-D**. In this work, we focus exclusively on the **RAVDESS** dataset. The **training and validation splits** are reverberated synthetically using the **'gpuRIR' Python library**, while the **test sets** are reverberated with real-world ACE RIRs recorded in various acoustic environments. You can choose which modality to fine-tune using the ```multimodal``` flag in ```config.py```.
- 🔗 [Dataset & Pre-trained Models](https://www.dropbox.com/scl/fo/px0lqliq9kc53tpv2smf2/AGnCptxxQBvGsigrih7urbU?rlkey=cxtlcus07oka82xvfruz8t5cw&st=2lm1tbt9&dl=0)

**Note** that the first work fine-tuned on different amounts of samples and splits.

`prepare_data.py` builds this layout from the public RAVDESS and ACE corpora (default paths in `config.py`, under `data/`):
```bash
data/
  │── RAVDESS_AV_actor_split_REV_1440_500_850/   # config.dataset_path
  │         │── ravdess_av_train.npy
  │         │── ravdess_av_eval.npy
  │         │── ravdess_av_test.npy
  │── ACE/RAVDESS_actor_split/                    # config.ace_path
  │         │── mp1/
  │         │     │── Lecture_room_1_508/ravdess_av_test.npy
  │         │     │── Lecture_room_2_403a/ravdess_av_test.npy
  │         │     │── lobby/ ... "Meeting room_2_611"/ Meeting_room_1_503/ Office_1_502/ Office_2_803/
  │         │── mp2/ ...
  │── RAVDESS/videos/Actor_XX/01-01-*.mp4         # config.video_path
  │── checkpoints/HTSAT_AudioSet_Saved_1.ckpt     # config.resume_checkpoint, unless model/HTSAT_AudioSet_Saved_3.ckpt exists
```
Each `.npy` is an `(N, 1)` object array; `dataset[i][0]` is a dict with `wav_name`, `video_name` (relative to `config.video_path`), `waveform` (`(mics, samples)`, 16 kHz) and `target` (0-7).

---

### **3️⃣ Training & Evaluation**
#### **Preprocess Data**
```bash
python3 prepare_data.py --download --checkpoint --rev --video-cache   # RAVDESS + AudioSet checkpoint, reverberant 3-mic splits
python3 prepare_data.py --download-ace                                # optional: ACE "Mobile" RIRs (1.6 GB) -> test_ace sets
```
It adapts `prep_ravdess.ipynb` (first paper) to this work: 8 classes, actor split (19/2/3 actors), and one simulated room per utterance with the parameters of Sec. IV-A, using pyroomacoustics instead of gpuRIR so it also runs without CUDA. `MER_mac_tutorial.ipynb` runs the same steps, then training and testing, from a notebook.
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
#### **Evaluate Performance**
On the test split of `config.dataset_path`:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py test
```
On every ACE room and mic position (`config.ace_rooms`, `config.ace_mic_positions`; `config.ace_models` evaluates several checkpoints). Accuracy and 75% confidence intervals are written to `config.ace_results_dir`. Run it on one GPU: the confidence intervals are only computed in single-device runs.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py test_ace
```

### **🍎 Running on macOS (Apple silicon)**
The same code runs on a Mac, on its GPU (MPS). Tested on an M4 Max (36 GB):
```bash
conda create -n mer python=3.12 -y && conda activate mer
pip install -r requirements-mac.txt
```
Then run the steps above without `CUDA_VISIBLE_DEVICES`, or open `MER_mac_tutorial.ipynb`. On a Mac:
- steps of 8 clips with gradients accumulated over 4 steps keep the effective batch of 32 (32 clips at once don't fit in memory); BatchNorm statistics are per 8 clips;
- R(2+1)D's convolutions run as equivalent 2-D convolutions on MPS (same weights, 2.6x faster);
- every clip's 8 video frames are decoded once into `config.video_cache_dir` (identical tensors), and data loads in the main process (`num_workers = 0`);
- `prepare_data.py` simulates the training rooms with pyroomacoustics (paper parameters in `prepare_data.SIM`); the actor split is a seeded 19/2/3 draw.

On an M4 Max an epoch on the full dataset takes ~2 min for the multimodal model (0.66 s per 8-clip step) and ~1 min audio-only. The data needs ~10 GB (6.6 GB of video); each training run saves up to 8 checkpoints of 746 MB.

---

## 📊 Results

![results](images/results_mer.png)
---

## 🏆 Citation
If you use this work, please cite:
```bibtex
@article{cohen2024multi,
  title={Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environment},
  author={Cohen, Ohad and Hazan, Gershon and Gannot, Sharon},
  journal={arXiv preprint arXiv:2409.09545},
  year={2024}
}
```

---

## 🌟 Acknowledgments
This research was supported by the **European Union’s Horizon 2020 Program** and the **Audition Project, Data Science Program, Israel**.

---

## 👤 Contact
For questions or collaborations, feel free to reach out:  
📧 **ohad.cohen@biu.ac.il**


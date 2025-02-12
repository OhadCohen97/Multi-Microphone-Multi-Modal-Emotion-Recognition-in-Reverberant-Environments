# Multi-Microphone & Multi-Modal Emotion Recognition in Reverberant Environments - Submitted to EUSIPCO 2025

🔥 **Official repository for the paper:**  
**"Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments"**  
This work is an extension of our previous paper "Multi-Microphone Speech Emotion Recognition using the Hierarchical Token-semantic Audio Transformer
Architecture", incorporating multi-modal learning to further improve robustness.  
📄 [Read the Paper](https://arxiv.org/pdf/2409.09545)  


---

## 🔍 Overview
Human emotions are conveyed through **speech and facial expressions**, making **multi-modal emotion recognition (MER)** crucial for robust emotion classification. This work introduces a **multi-microphone, multi-modal system** combining:

- **HTS-AT Transformer** for multi-channel audio processing
- **R(2+1)D ResNet CNN** for video-based emotion recognition
- **Late fusion concatenation** to combine audio & video features

### 🔮 Key Features
✔️ **Multi-Microphone Audio Processing**: Robust against reverberation  
✔️ **Multi-Modal Learning**: Combines speech and facial cues  
✔️ **Tested on RAVDESS with Real-World RIRs** (ACE Database)  
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
#### **Clone the Repository**
```bash
git clone https://github.com/yourusername/Multi-Modal-SER.git
cd Multi-Modal-SER
```
#### **Use the Virtual Environment**
```bash
Python 3.8.13 ('base')
```

---

### **2️⃣ Dataset Setup**
The previous paper evaluated the models on three datasets: **RAVDESS**, **IEMOCAP**, and **CREMA-D**. In this work, we focus exclusively on the **RAVDESS** dataset. The **training and validation splits** are reverberated synthetically using the **'gpuRIR' Python library**, while the **test sets** are reverberated with real-world ACE RIRs recorded in various acoustic environments. You can choose which modality to fine-tune using the ```multimodal``` flag in ```config.py```.
- 🔗 [Dataset & Pretrained Models](LINK_TO_DATA_MODELS)

**Note** that the first work fine-tuned on diffrent amounts of samples and splits.

Place datasets inside `data/` folder:
```bash
MER/
│── <dataset name & type>/
      │── data/
            │── train.npy
            │── val.npy
            │── test.npy
│── ACE/
      │── <dataset name>
            │── MP1/
                  │── Lecture_room_1_508
                        │── train.npy
                        │── val.npy
                        │── test.npy
                  │── Lecture_room_2_403a
                        │── train.npy
                        │── val.npy
                        │── test.npy
                  │── lobby
                     ...
                  │── Meeting room_2 _611
                  │── Office_1_502
                  │── Office_2_803
```

---

### **3️⃣ Training & Evaluation**
#### **Preprocess Data**
For preprocess the data see the notebooks.
#### **Fine-tune the Models**
First, in ```config.py```, make sure you have the HTS-AT AudioSet pre-trained model and the path to it. Set the root path to the dataset you with to use.
To train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 main.py train
```
#### **Evaluate Performance**
To evaluate the model, you must set the paths for the npy files. See in ```main.py test_ace (lines 219,236,253) ```.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 main.py test_ace
```
For single test on one npy file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 main.py test
```

---

## 📊 Results

![results](images/results_mer.png)
---

## 🏆 Citation
If you use this work, please cite:
```bibtex
@inproceedings{cohen2025multimodal,
  author    = {Ohad Cohen, Gershon Hazan, Sharon Gannot},
  title     = {Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environments},
  booktitle = {ICASSP},
  year      = {2025}
}
```

---

## 🌟 Acknowledgments
This research was supported by the **European Union’s Horizon 2020 Program** and the **Audition Project, Data Science Program, Israel**.

---

## 👤 Contact
For questions or collaborations, feel free to reach out:  
📧 **ohad.cohen@biu.ac.il**


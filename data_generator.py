import numpy as np
import logging
import random
from torch.utils.data import Dataset
import cv2
import torch
from torchmultimodal.transforms.video_transform import VideoTransform
# import pytorchvideo
import torchvision
# import pytorchvideo.transforms as pytrans
import torchvision.io
from random import randint

def load_video(video_path):    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def video_to_tensor(frames):
    frames_tensor = torch.from_numpy(np.stack(frames))
    # frames_tensor = frames_tensor.permute(3, 0, 1, 2)
    frames_tensor = frames_tensor.unsqueeze(0)
    return frames_tensor


class Emotion_DataLoader(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config
        if self.config.video == True:
            self.vt = VideoTransform()
            self.vt.time_samples = 8 # 16 for clean data, 8 for REV data
            # self.vt.resize_shape = (180,180)
        self.eval_mode = eval_mode
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]

        if self.config.audio == True:
            if self.config.num_mics == 1: # single-chnnel
                data_waveform = self.dataset[p][0]["waveform"] 
                if data_waveform.shape[0]>1: # for data that build as (num_mics, waveform) and wanted to take the first channel
                    data_waveform = data_waveform[0,:] #(,waveform)
            else: # num_mics > 2 --> multi-channel
                data_waveform = self.dataset[p][0]["waveform"][0:self.config.num_mics,:].T # (waveform, num_mics)
                
            real_len = len(data_waveform)
            if real_len < self.config.clip_samples:
                zero_pad = np.zeros(self.config.clip_samples - real_len)
                if self.config.num_mics > 1:
                    zero_pad = np.resize(zero_pad,(len(zero_pad),self.config.num_mics))
                waveform = np.concatenate([data_waveform, zero_pad])
            else:
                waveform = data_waveform[:self.config.clip_samples]

            if self.config.num_mics == 1:
                waveform = np.expand_dims(waveform, axis=0) #(1, wavform)
            else:
                waveform = waveform.T
            waveform = waveform.astype(np.float32)
        else:
            waveform = []
            real_len = []
        

        if self.config.video ==True:
            video_path = "/home/dsi/ohadico97/HTS-AT_MER/mer/RAVDESS/Video/" + self.dataset[p][0]["video_name"]
            
            # frames = load_video(vid_path+self.dataset[p][0]["video_name"])
            # video_tensor = video_to_tensor(frames)

            video_tensor, _, _ = torchvision.io.read_video(video_path)

            v = self.vt(video_tensor.unsqueeze(0))
            
            # jitter = randint(0,1)

            if self.config.video_augment == True:
                if self.eval_mode == False: #augment in train only
                    # r1 = 0 
                    # r2 = 0.4
                    # jit_params = (r1 - r2) * torch.rand(4) + r2

                    transform = torchvision.transforms.Compose([
                        torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(180),
                            # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05),
                            torchvision.transforms.RandomHorizontalFlip(p=0.30),
                            torchvision.transforms.RandomVerticalFlip(p=0.30),
                            # torchvision.transforms.ColorJitter(brightness=jit_params[0].item(), contrast=jit_params[1].item(), saturation=jit_params[2].item(), hue=jit_params[3].item()),
                            # torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                            torchvision.transforms.RandomRotation(degrees=(-30,30)),
                        ])
                    ])

                    v = transform(v.squeeze(0).float().permute(1,0,2,3))
                    v = v.squeeze(0).permute(1,0,2,3)
                else:
                    v = v.squeeze(0)
            else:
                v = v.squeeze(0)
        else:
            v = []
                
        data_dict = {
            "audio_name": self.dataset[p][0]["wav_name"],
            "video_name": self.dataset[p][0]["video_name"],
            "waveform": waveform,
            "transformed_video":v,
            "real_len": real_len,
            "target": self.dataset[p][0]["target"]
        }
        return data_dict

    def __len__(self):
        return self.total_size

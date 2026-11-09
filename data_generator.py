import os
import types
import numpy as np
import logging
import random
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import normalize, resize
from random import randint


class VideoTransform:
    """Copy of torchmultimodal.transforms.video_transform.VideoTransform (BSD license, Meta Platforms), whose
    logic has not changed since 2022 and is what the original code used: nearest-neighbour sampling of
    `time_samples` frames, bilinear resize to 224x224, Kinetics mean/std normalisation.
    (B, T, H, W, C) uint8 in -> (B, C, time_samples, 224, 224) float out.

    Resize antialiasing is off, as torchvision <= 0.16 did for tensors (the Docker image); torchvision >= 0.17
    would switch it on by default and the frames would differ between machines."""

    def __init__(self, time_samples=32, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989),
                 resize_shape=(224, 224)):
        self.time_samples = time_samples
        self.mean = mean
        self.std = std
        self.resize_shape = resize_shape

    def __call__(self, video):
        if video.shape[-1] != 3:
            raise ValueError("Video must have 3 channels")
        video = self.sample_frames(video)
        video = self.resize_hw(video)
        video = self.normalize(video)
        return video.permute(0, 4, 1, 2, 3).contiguous() # b t h w c -> b c t h w

    # every permute is followed by .contiguous(), like the original's shift_dim: the interpolation kernels of
    # some torch versions (e.g. 1.13) round differently on non-contiguous (channels-last) inputs

    def sample_frames(self, video):
        """Samples frames from video of dims (b, t, h, w, c)"""
        _, t, h, w, _ = video.shape
        if t != self.time_samples:
            video = F.interpolate(video.permute(0, 4, 1, 2, 3).contiguous(), size=[self.time_samples, h, w]) # b c t h w
            video = video.permute(0, 2, 3, 4, 1).contiguous() # b c t h w -> b t h w c
        return video

    def resize_hw(self, video):
        """Resizes height and width of video of dims (b, t, h, w, c)"""
        b, t, h, w, _ = video.shape
        video = video.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2).contiguous() # (b t) c h w
        if (h, w) != tuple(self.resize_shape):
            video = resize(video, list(self.resize_shape), antialias=False)
        return video.unflatten(0, (b, t)).permute(0, 1, 3, 4, 2).contiguous() # b t h w c

    def normalize(self, video):
        """Normalizes video of dims (b, t, h, w, c) to mean 0, std 1"""
        b, t, _, _, _ = video.shape
        video = video.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2).contiguous() # (b t) c h w
        video = normalize(video.float() / 255.0, mean=self.mean, std=self.std)
        return video.unflatten(0, (b, t)).permute(0, 1, 3, 4, 2).contiguous() # b t h w c


def sampled_frame_indices(num_frames, time_samples):
    """The frame indices VideoTransform.sample_frames keeps (nearest-neighbour interpolation over time)."""
    if num_frames == time_samples:
        return list(range(num_frames))
    idx = torch.arange(num_frames, dtype=torch.float32).view(1, 1, num_frames, 1, 1)
    return F.interpolate(idx, size=[time_samples, 1, 1]).flatten().long().tolist()


def _to_rgb(frame):
    """YUV -> RGB with BT.601 / limited range, what PyAV < 14 (and so torchvision.io.read_video in the Docker image)
    used for every video. PyAV >= 14 would use the BT.709 tag of the RAVDESS videos instead: up to 14/255 brighter.
    With this, PyAV >= 14 stays within 3/255 of the old output (FFmpeg's newer swscale rounds differently)."""
    try:
        return frame.reformat(format="rgb24", src_colorspace="ITU601", src_color_range="MPEG").to_ndarray()
    except TypeError: # PyAV without src_color_range: its default is already this conversion
        return frame.to_ndarray(format="rgb24")


def read_video_frames(video_file, time_samples=None):
    """RGB frames of a video as a uint8 tensor (T, H, W, 3): the frames torchvision.io.read_video returns
    (it decodes with PyAV as well), via PyAV directly since that torchvision API is deprecated and breaks with
    PyAV >= 14. With time_samples, only the frames VideoTransform would keep are converted to RGB (faster;
    the result of VideoTransform is unchanged)."""
    import av
    with av.open(video_file) as container:
        frames = {}
        for frame in container.decode(video=0):
            pts = frame.pts if frame.pts is not None else len(frames)
            if pts >= 0:
                frames[pts] = frame # read_video also keeps one frame per pts, in pts order
        frames = [frames[k] for k in sorted(frames)]
        if not frames:
            raise RuntimeError("no video frames decoded from %s" % video_file)
        if time_samples is not None:
            frames = [frames[i] for i in sampled_frame_indices(len(frames), time_samples)]
        return torch.from_numpy(np.stack([_to_rgb(f) for f in frames]))


def video_cache_file(cache_dir, video_name, time_samples):
    stem = os.path.splitext(video_name)[0].replace("\\", "/").replace("/", "__")
    return os.path.join(cache_dir, "%s_t%d.npy" % (stem, time_samples))


def load_video_clip(video_file, time_samples, cache_file=None):
    """VideoTransform(time_samples)(read_video(video_file)) as (1, C, T, 224, 224) float32.
    With cache_file, the sampled + resized uint8 frames are stored there on first use, so later calls skip
    decoding; normalising them again gives exactly the same tensor."""
    vt = VideoTransform(time_samples=time_samples)
    frames = None
    if cache_file is not None and os.path.exists(cache_file):
        try:
            frames = torch.from_numpy(np.load(cache_file))
        except (OSError, ValueError, EOFError): # partial / corrupt file: rebuild it
            frames = None
    if frames is None:
        frames = vt.resize_hw(vt.sample_frames(read_video_frames(video_file, time_samples).unsqueeze(0)))
        if cache_file is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            tmp = "%s.%d.tmp.npy" % (cache_file[:-4], os.getpid())
            np.save(tmp, frames.numpy())
            os.replace(tmp, cache_file)
    return vt.normalize(frames).permute(0, 4, 1, 2, 3).contiguous()


def load_video(video_path):
    import cv2
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


def config_snapshot(config):
    """A plain, picklable copy of the config module. DataLoader workers are started with 'spawn' on macOS,
    which pickles the dataset -- and a module object can't be pickled."""
    if not isinstance(config, types.ModuleType):
        return config
    values = {}
    for key in dir(config):
        value = getattr(config, key)
        if not key.startswith("_") and not isinstance(value, types.ModuleType) and not callable(value):
            values[key] = value
    return types.SimpleNamespace(**values)


class Emotion_DataLoader(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config_snapshot(config)
        if self.config.video == True:
            self.time_samples = getattr(self.config, "video_time_samples", 8) # 16 for clean data, 8 for REV data
            self.video_cache_dir = getattr(self.config, "video_cache_dir", None)
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
        sample = self.dataset[p][0]

        if self.config.audio == True:
            wav = np.asarray(sample["waveform"])
            if wav.ndim == 1: # mono data saved as (samples,)
                wav = wav[np.newaxis, :]
            if wav.shape[0] < self.config.num_mics:
                raise ValueError("%s has %d channel(s) but config.num_mics = %d"
                                 % (sample.get("wav_name", sample.get("name")), wav.shape[0], self.config.num_mics))
            if self.config.num_mics == 1: # single-chnnel: the first channel
                data_waveform = wav[0, :] #(,waveform)
            else: # num_mics > 2 --> multi-channel
                data_waveform = wav[0:self.config.num_mics,:].T # (waveform, num_mics)

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
            cache_file = None
            if self.video_cache_dir:
                cache_file = video_cache_file(self.video_cache_dir, sample["video_name"], self.time_samples)
            v = load_video_clip(self.config.video_path + sample["video_name"], self.time_samples, cache_file)

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
            "audio_name": sample.get("wav_name", sample.get("name", "")),
            "video_name": sample.get("video_name", ""),
            "waveform": waveform,
            "transformed_video":v,
            "real_len": real_len,
            "target": sample["target"]
        }
        return data_dict

    def __len__(self):
        return self.total_size

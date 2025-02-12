import os
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
import wandb
from pytorch_lightning.loggers import WandbLogger
import pandas as pd

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler
        )
        return train_loader
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = eval_sampler
        )
        return eval_loader
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = test_sampler
        )
        return test_loader
    

def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.wa_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.wa_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)

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


def test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    if config.dataset_type == "RAVDESS":
            print("Test on RAVDESS")
            test_set = np.load(os.path.join(config.dataset_path, "ravdess_av_test.npy"), allow_pickle = True)#ravdess_ALL_REV_160.npy ravdess_test_BIUREVgen
            eval_dataset = Emotion_DataLoader(
                dataset = test_set,
                config = config,
                eval_mode = True
            )
    elif config.dataset_type == "IEMOCAP":
            print("Test on IEMOCAP")
            test_set = np.load(os.path.join(config.dataset_path, "iemocap_test.npy"), allow_pickle = True)# iemocap_all IEMOCAP_test_BIUREVgen.npy
            eval_dataset = Emotion_DataLoader(
                dataset = test_set,
                config = config,
                eval_mode = True
            )
    elif config.dataset_type == "CREMA-D":
            print("Test on CREMA-D")
            test_set = np.load(os.path.join(config.dataset_path, "cremad_test.npy"), allow_pickle = True)#cremad_rev_2m_090_all.npy CREMAD_test_BIUREVgen.npy cremad_test
            eval_dataset = Emotion_DataLoader(
                dataset = test_set,
                config = config,
                eval_mode = True
            )

    data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=True,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        resume_from_checkpoint = config.resume_checkpoint_test,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
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
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = eval_dataset
    )
    if config.resume_checkpoint_test is not None:
        ckpt = torch.load(config.resume_checkpoint_test, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight") # put in comment when testing video model 
        ckpt["state_dict"].pop("sed_model.head.bias") # put in comment when testing video model 
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.test(model, datamodule=data)


def test_ace():
    # must change mode_type and datasetype as well in config.py
    models = {"/home/dsi/ohadico97/HTS-AT_MER/HTSAT_RAVDESS_Saved_x.ckpt":1} # {model path: number of mics}

    if config.dataset_type == "RAVDESS":
        rooms = ['/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Lecture_room_1_508',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Lecture_room_2_403a',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/lobby',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Meeting room_2_611',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Meeting_room_1_503',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Office_1_502',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp1/Office_2_803',
                # MP2
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Lecture_room_1_508',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Lecture_room_2_403a',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/lobby',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Meeting room_2_611',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Meeting_room_1_503',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Office_1_502',
                '/home/dsi/ohadico97/HTS-AT_MER/ACE/RAVDESS_actor_split/mp2/Office_2_803']
                
    elif config.dataset_type == "IEMOCAP":
        rooms = ['/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Lecture_room_1_508',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Lecture_room_2_403a',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/lobby',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Meeting room_2 _611',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Meeting_room_1_503',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Office_1_502',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp1_test_set/Office_2_803',
                # MP2
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Lecture_room_1_508',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Lecture_room_2_403a',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/lobby',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Meeting room_2 _611',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Meeting_room_1_503',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Office_1_502',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/IEMOCAP/mp2_test_set/Office_2_803']
        
    elif config.dataset_type == "CREMA-D":
                rooms = ['/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Lecture_room_1_508',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Lecture_room_2_403a',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/lobby',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Meeting room_2 _611',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Meeting_room_1_503',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Office_1_502',
                 '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp1_test_set/Office_2_803',
                # MP2
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Lecture_room_1_508',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Lecture_room_2_403a',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/lobby',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Meeting room_2 _611',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Meeting_room_1_503',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Office_1_502',
                '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/ACE/CREMAD/mp2_test_set/Office_2_803']
    
    i = 0
    for model_path,mics in models.items():
        if i == 2:
            config.mode_type = 'sum'
        config.num_mics = mics
        config.resume_checkpoint_test = model_path
        print('MODEL: ', config.resume_checkpoint_test)
        df = pd.read_csv('/home/dsi/ohadico97/HTS-AT_MER/ConfidenceIntervals-main/multi_channel_sum copy 2.csv', sep='|')
        for idx,dataset_path in enumerate(rooms):
            device_num = torch.cuda.device_count()
            # dataset file pathes
            if config.dataset_type == "RAVDESS":
                    print("Test on RAVDESS")
                    test_set = np.load(os.path.join(dataset_path, "ravdess_av_test.npy"), allow_pickle = True)#ravdess_ALL_REV_160.npy ravdess_test_BIUREVgen
                    eval_dataset = Emotion_DataLoader(
                        dataset = test_set,
                        config = config,
                        eval_mode = True
                    )
            elif config.dataset_type == "IEMOCAP":
                    print("Test on IEMOCAP")
                    test_set = np.load(os.path.join(dataset_path, "iemocap_test.npy"), allow_pickle = True)# iemocap_all IEMOCAP_test_BIUREVgen.npy
                    eval_dataset = Emotion_DataLoader(
                        dataset = test_set,
                        config = config,
                        eval_mode = True
                    )
            elif config.dataset_type == "CREMA-D":
                    print("Test on CREMA-D")
                    test_set = np.load(os.path.join(dataset_path, "cremad_test.npy"), allow_pickle = True)#cremad_rev_2m_090_all.npy CREMAD_test_BIUREVgen.npy
                    eval_dataset = Emotion_DataLoader(
                        dataset = test_set,
                        config = config,
                        eval_mode = True
                    )

            data = data_prep(eval_dataset, eval_dataset, device_num)
            trainer = pl.Trainer(
                deterministic=True,
                gpus = device_num, 
                max_epochs = config.max_epoch,
                auto_lr_find = True,    
                sync_batchnorm = True,
                checkpoint_callback = False,
                accelerator = "ddp" if device_num > 1 else None,
                num_sanity_val_steps = 0,
                resume_from_checkpoint = config.resume_checkpoint_test,
                replace_sampler_ddp = False,
                gradient_clip_val=1.0
            )
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
            
            model = SEDWrapper(
                sed_model = sed_model, 
                config = config,
                dataset = eval_dataset
            )
            if config.resume_checkpoint_test is not None:
                ckpt = torch.load(config.resume_checkpoint_test, map_location="cpu")
                ckpt["state_dict"].pop("sed_model.head.weight") # put in comment when testing video model 
                ckpt["state_dict"].pop("sed_model.head.bias") # put in comment when testing video model 
                model.load_state_dict(ckpt["state_dict"], strict=False)
            res = trainer.test(model, datamodule=data)
            df['RAVDESS'][idx]= (res[0]['acc'],(res[0]['CI_low'],res[0]['CI_high']))
            print('NUM MICS: ', config.num_mics)
            print('ROOM: ',dataset_path)
        i = i + 1
        #write to CSV
        df.to_csv('/home/dsi/ohadico97/HTS-AT_MER/CI_results/res_66/'+model_path.split('/')[-2]+'.csv', sep='|', index=True)

def train():
    if config.debug == False:
        wandb_logger = WandbLogger()
        # start a new wandb run to track this script
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
    else:
        wandb_logger = None
    
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    
    if config.dataset_type == "RAVDESS":   # 8 classes
        train_set = np.load(os.path.join(config.dataset_path, "ravdess_av_train.npy"), allow_pickle = True)#ravdess_train_BIUREVgen
        eval_set = np.load(os.path.join(config.dataset_path, "ravdess_av_eval.npy"), allow_pickle = True)#ravdess_eval_BIUREVgen
    elif config.dataset_type == "IEMOCAP": # 4 classes
        train_set = np.load(os.path.join(config.dataset_path, "IEMOCAP_train_BIUREVgen.npy"), allow_pickle = True) #IEMOCAP_train_BIUREVgen.npy
        eval_set = np.load(os.path.join(config.dataset_path, "IEMOCAP_eval_BIUREVgen.npy"), allow_pickle = True) #IEMOCAP_eval_BIUREVgen.npy
    elif config.dataset_type == "CREMA-D": # 6 classes
        train_set = np.load(os.path.join(config.dataset_path, "CREMAD_train_BIUREVgen.npy"), allow_pickle = True)
        eval_set = np.load(os.path.join(config.dataset_path, "CREMAD_eval_BIUREVgen.npy"), allow_pickle = True)

    # set exp folder
    exp_dir = os.path.join(config.workspace, "results_"+config.dataset_type+"_batch_size"+str(config.batch_size)+"_num_mics"+str(config.num_mics)+"_"+config.mode_type+"_seed"+str(config.random_seed), config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results_"+config.dataset_type+"_batch_size"+str(config.batch_size)+"_num_mics"+str(config.num_mics)+"_"+config.mode_type+"_seed"+str(config.random_seed), config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results_"+config.dataset_type+"_batch_size"+str(config.batch_size)+"_num_mics"+str(config.num_mics)+"_"+config.mode_type+"_seed"+str(config.random_seed)))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)

    if config.dataset_type == "RAVDESS":
        print("Using RAVDESS")
        dataset = Emotion_DataLoader(
            dataset = train_set,
            config = config,
            eval_mode = False
        )
        eval_dataset = Emotion_DataLoader(
            dataset = eval_set,
            config = config,
            eval_mode = True
        )
    if config.dataset_type == "IEMOCAP":
        print("Using IEMOCAP")
        dataset = Emotion_DataLoader(
            dataset = train_set,
            config = config,
            eval_mode = False
        )
        eval_dataset = Emotion_DataLoader(
            dataset = eval_set,
            config = config,
            eval_mode = True
        )
    if config.dataset_type == "CREMA-D":
        print("Using CREMA-D")
        dataset = Emotion_DataLoader(
            dataset = train_set,
            config = config,
            eval_mode = False
        )
        eval_dataset = Emotion_DataLoader(
            dataset = eval_set,
            config = config,
            eval_mode = True
        )

    data = data_prep(dataset, eval_dataset, device_num)

    checkpoint_callback = ModelCheckpoint(
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
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir = checkpoint_dir,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback,early_stopping_callback],
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0,
        logger=wandb_logger
    )
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
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")

        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif config.swin_pretrain_path is not None: # train with pretrained model
        ckpt = torch.load(config.swin_pretrain_path, map_location="cpu")
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
    pl.utilities.seed.seed_everything(seed = config.random_seed)
    
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
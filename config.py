exp_name = "exp_finetune_actor_split" # the saved ckpt prefix name of the model 
workspace = "/home/dsi/ohadico97/HTS-AT_MER" # the folder of your code

multimodal = True
if multimodal == True:
    audio = True
    video = True
    video_path = "/home/dsi/ohadico97/HTS-AT_MER/mer/RAVDESS/Video/"
else: #chose modality
    audio = True
    video = False

num_mics = 3
if num_mics == 1:
    mode_type = 'avg_mel' # vanilla HTS-AT
else:
    mode_type = 'avg_mel' # "sum" | "avg_mel" 
    
dataset_type = "RAVDESS" # "IEMOCAP" | "RAVDESS" | "CREMA-D" 
loss_type ="clip_ce" # or "clip_bce" for binary cross-entropy

# trained from a checkpoint for fine-tuning, set None to train from scratch
resume_checkpoint = "/home/dsi/ohadico97/HTS-AT_single_multi_channel/HTS-Audio-Transformer/HTSAT_AudioSet_Saved_3.ckpt"
# evaluate a single model for testing
resume_checkpoint_test = "/home/dsi/ohadico97/HTS-AT_MER/Experiments/Audio_Video/results_RAVDESS_batch_size32_multimodal_clean_seed1/exp_htsat_pretrain_actor_split_multimodal_80%/HTSAT_RAVDESS_Saved_x.ckpt" 
debug = False

batch_size = 32 # batch size per GPU x GPU number , default is 32 x 4 = 128
learning_rate = 1e-3 # 1e-4 also workable 
max_epoch = 500
num_workers = 8

lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

video_model = None # None for r2dplus, 'x3d_s' for facebookresearch/pytorchvideo model
video_augment = True
confusion_matrix = False # for plotting confusion matrix in sed_model.py

if dataset_type == "RAVDESS":
    random_seed = 1 # 19970318 970131 12412 127777 1009 34047
    dataset_path = "/home/dsi/ohadico97/HTS-AT_MER/data_rev_gen/RAVDESS_AV_actor_split_REV_1440_500_850" #"/home/dsi/ohadico97/HTS-AT_MER/data_clean/RAVDESS_AV" #"/home/dsi/ohadico97/HTS-AT_MER/data_clean/RAVDESS_AV_actor_split"  # the dataset path 
    classes_num = 8 
    sec = 4
    es_patience = 12
    time_drop = 64
    time_stripes = 4
    freq_drop = 8
    freq_stripes = 2
    htsat_depth = [2,2,6,2] # for htsat hyperparamater
elif dataset_type == "IEMOCAP":
    random_seed = 19970318 # 19970318 970131 12412 127777 1009 34047
    dataset_path = "/ohadcolab/ohadcolab/HTS-AT_single_multi_channel/HTS-Audio-Transformer/data_multichannel/generated_IEMOCAP_BIUREVgen_sim" # the dataset path
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
    dataset_path = "/ohadcolab/ohadcolab/HTS-AT_single_multi_channel/HTS-Audio-Transformer/data_multichannel/generated_CREMAD_BIUREVgen_sim"   # the dataset path
    classes_num = 6 # '/ohadcolab/ohadcolab/datasets/ser/real_RIRs_datasets/CREMAD/610ms_test_set/2m_090'
    sec = 2.5
    es_patience = 25
    time_drop = 64
    time_stripes = 3
    freq_drop = 16
    freq_stripes = 2
    htsat_depth = [2,2,6,2] # for htsat hyperparamater

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


# weight average folder
wa_folder  =  workspace+"/results_"+dataset_type+"_batch_size"+str(batch_size)+"_num_mics"+str(num_mics)+"_"+mode_type+"_seed"+str(random_seed)+"/"+exp_name+"/checkpoint/None/version_None/checkpoints"

# weight average output filename
wa_model_path = workspace + "/HTSAT_"+dataset_type+"_Saved_x.ckpt"


# esm_model_pathes = [
#     ".../HTSAT_AudioSet_Saved_1.ckpt",
#     ".../HTSAT_AudioSet_Saved_2.ckpt",
#     ".../HTSAT_AudioSet_Saved_3.ckpt",
#     ".../HTSAT_AudioSet_Saved_4.ckpt",
#     ".../HTSAT_AudioSet_Saved_5.ckpt",
#     ".../HTSAT_AudioSet_Saved_6.ckpt"
# ]


fl_local = False # indicate if we need to use this dataset for the framewise detection

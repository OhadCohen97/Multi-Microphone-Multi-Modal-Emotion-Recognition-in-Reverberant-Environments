import numpy as np
import os
import bisect
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_loss_func
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pytorch_lightning as pl
import config
import wandb
from confidence_intervals import evaluate_with_conf_int
sns.set(rc={'figure.figsize':(12,8)})



def confusion(ans,pred,acc,dic):
    cm = confusion_matrix(y_true=ans, y_pred=np.argmax(pred, 1))
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    sns.heatmap(cmn*100, cmap='Blues', annot=True, fmt='.2f', xticklabels=dic.values(), yticklabels=dic.values())
    ax.xaxis.set_label_position("bottom")
    plt.setp(ax.get_yticklabels())
    plt.setp(ax.get_xticklabels())
    plt.title( "Multi-Channel Avg Mel MER - RAVDESS ACE Lecture Room 2 (T60=1220), Accuracy: " + str(round(acc*100,1))+"%")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("/home/dsi/ohadico97/HTS-AT_MER/confusion_matrix/Multimodal_AVG_MEL_RAVDESS_ACE.png",dpi=300)
    
    
class SEDWrapper(pl.LightningModule):
    def __init__(self, sed_model, config, dataset):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)

    def evaluate_metric(self, pred, ans):
        if self.config.dataset_type == "RAVDESS":
            acc = accuracy_score(ans, np.argmax(pred, 1))
            dic = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fearful",6:"disgust",7:"surprised"} # RAVDESS - 8 emotions 
            
            # dic = {0:"neutral",1:"happy",2:"sad",3:"angry",4:"fear",5:"disgust",6:"surprise"} # RAVDESS - 7 emotions 
            print(classification_report(ans, np.argmax(pred, 1), target_names=list(dic.values())))
            #confidence intervals
            # print(evaluate_with_conf_int(np.argmax(pred, 1), accuracy_score, ans)) #num_bootstraps = 10000
            
            if config.confusion_matrix == True:
                confusion(ans,pred,acc,dic)
            return {"acc": acc,"CI":evaluate_with_conf_int(np.argmax(pred, 1), accuracy_score, ans,alpha=25)}  # 33, 25 
        elif self.config.dataset_type == "IEMOCAP":    
            acc = accuracy_score(ans, np.argmax(pred, 1))
            dic = {0: 'neutral', 1: 'angry', 2: 'happy', 3: 'sad'}
            # dic = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'} # cross-corpus
            print(classification_report(ans, np.argmax(pred, 1), target_names=list(dic.values())))
            
            #confidence intervals
            # print(evaluate_with_conf_int(np.argmax(pred, 1), accuracy_score, ans))
            
            if config.confusion_matrix == True:
                confusion(ans,pred,acc,dic)
            return {"acc": acc}
        elif self.config.dataset_type == "CREMA-D":
            acc = accuracy_score(ans, np.argmax(pred, 1))
            dic = {0: "NEU",1: "HAP", 2:"SAD", 3: "FEA",4: "ANG",5:"DIS"}
            print(classification_report(ans, np.argmax(pred, 1), target_names=list(dic.values())))
            
            #confidence intervals
            # print(evaluate_with_conf_int(np.argmax(pred, 1), accuracy_score, ans))
            
            if config.confusion_matrix == True:
                confusion(ans,pred,acc,dic)
            return {"acc": acc}  

    def forward(self, x,y):
        output_dict = self.sed_model(x,y)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device

        pred, _ = self(batch["waveform"],batch["transformed_video"])
        loss = self.loss_func(pred, batch["target"])
        self.log("train_loss", loss)# on_epoch= True, prog_bar=True)
        # wandb.log({"train_loss":loss})
        return loss
        
    def training_epoch_end(self, outputs):
        self.dataset.generate_queue()

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"],batch["transformed_video"])
        loss = self.loss_func(pred, batch["target"])
        self.log("val_loss", loss, on_epoch= True, prog_bar=True)
        # wandb.log({"val_loss":loss})
        return [pred.detach(), batch["target"].detach()]
    
    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)

        if torch.cuda.device_count() > 1:
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()

        metric_dict = {
                "acc":0.
            }
        if torch.cuda.device_count() > 1:
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
        
            self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()
        else:
            gather_pred = pred.cpu().numpy()
            gather_target = target.cpu().numpy()
            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            print(self.device_type, metric_dict, flush = True)
        
            self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)

            
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        # time shifting optimization
        if self.config.fl_local or self.config.dataset_type != "audioset": 
            shift_num = 1 # framewise localization cannot allow the time shifting
        else:
            shift_num = 10 
        for i in range(shift_num):
            pred, pred_map = self(batch["waveform"],batch["transformed_video"])
            preds.append(pred.unsqueeze(0))
            if self.config.audio == True:
                batch["waveform"] = self.time_shifting(batch["waveform"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        if self.config.fl_local:
            return [
                pred.detach().cpu().numpy(), 
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy()
            ]
        else:
            return [pred.detach(), batch["target"].detach()]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len":real_len[i]
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:
            self.device_type = next(self.parameters()).device
            pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
            target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
            if torch.cuda.device_count() > 1:
                gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
                gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
                dist.barrier()
            metric_dict = {
                    "acc":0.
                }
            if torch.cuda.device_count() > 1:
                dist.all_gather(gather_pred, pred)
                dist.all_gather(gather_target, target)
                if dist.get_rank() == 0:
                    gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                    gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                    metric_dict = self.evaluate_metric(gather_pred, gather_target)
                    print(self.device_type, dist.get_world_size(), metric_dict, flush = True)

                self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
                dist.barrier()
            else:
                gather_pred = pred.cpu().numpy()
                gather_target = target.cpu().numpy()
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, metric_dict, flush = True)

                self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)
                acc,interval = metric_dict['CI']
                self.log("CI_low", interval[0], on_epoch = True, prog_bar=True, sync_dist=False)
                self.log("CI_high", interval[1], on_epoch = True, prog_bar=True, sync_dist=False)
        #TODO: try to return CI. maybe self.log...

    
    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr = self.config.learning_rate, 
        #     betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        # )
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr = self.config.learning_rate)
        # def lr_foo(epoch):       
        #     if epoch < 3:
        #         # warm up lr
        #         lr_scale = self.config.lr_rate[epoch]
        #     else:
        #         # warmup schedule
        #         lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
        #         if lr_pos < -3:
        #             lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
        #         else:
        #             lr_scale = self.config.lr_rate[lr_pos]
        #     return lr_scale
        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo
        # )
        
        def lr_foo(epoch):       
            if epoch < len(self.config.lr_rate):
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        
        return [optimizer], [scheduler]



class Ensemble_SEDWrapper(pl.LightningModule):
    def __init__(self, sed_models, config, dataset):
        super().__init__()

        self.sed_models = nn.ModuleList(sed_models)
        self.config = config
        self.dataset = dataset

    def evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"acc": acc}
        
    def forward(self, x, sed_index):
        self.sed_models[sed_index].eval()
        preds = [] 
        pred_maps = []
        # time shifting optimization
        if self.config.fl_local or self.config.dataset_type != "audioset": 
            shift_num = 1 # framewise localization cannot allow the time shifting
        else:
            shift_num = 10
        for i in range(shift_num):
            pred, pred_map = self.sed_models[sed_index](x)
            pred_maps.append(pred_map.unsqueeze(0))
            preds.append(pred.unsqueeze(0))
            x = self.time_shifting(x, shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred_maps = torch.cat(pred_maps, dim = 0)
        pred = preds.mean(dim = 0)
        pred_map = pred_maps.mean(dim = 0)
        return pred, pred_map

    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        pred = torch.zeros(len(batch["waveform"]), self.config.classes_num).float().to(self.device_type)
        for j in range(len(self.sed_models)):
            temp_pred, _ = self(batch["waveform"], j)
            pred = pred + temp_pred
        pred = pred / len(self.sed_models)
        return [
            pred.detach(), 
            batch["target"].detach(), 
        ]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]

        dist.barrier()

        metric_dict = {
                "acc":0.
            }
        dist.all_gather(gather_pred, pred)
        dist.all_gather(gather_target, target)
        if dist.get_rank() == 0:
            gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
            gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            print(self.device_type, dist.get_world_size(), metric_dict, flush = True)

        self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
        dist.barrier()
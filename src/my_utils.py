import torch
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as sk
from time import ctime

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms



class DatasetWithMetaCelebA(Dataset):
    def __init__(self, image_ds, labels, metas):
        super().__init__()
        self.ds = image_ds
        self.labels = labels
        self.metas = metas
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # input, target, meta-feature
        return (index, self.ds[index][0], self.labels[index], self.metas[index])
    


class DatasetWithMeta(Dataset):
    def __init__(self, inputs, labels, metas):
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.metas = metas
        
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (idx, self.inputs[idx], self.labels[idx].float(), self.metas[idx].float())



def get_data(data, data_dir='../data'):
    
    if data == 'compas':
        train_ds = torch.load(f'{data_dir}/compas_pytorch/compas_train.pt')
        val_ds = torch.load(f'{data_dir}/compas_pytorch/compas_val.pt')
        test_ds = torch.load(f'{data_dir}/compas_pytorch/compas_test.pt')
    elif data == 'adult':
        train_ds = torch.load(f'{data_dir}/adult_pytorch/adult_train.pt')
        val_ds = torch.load(f'{data_dir}/adult_pytorch/adult_val.pt')
        test_ds = torch.load(f'{data_dir}/adult_pytorch/adult_test.pt')
        
    elif data == 'celebA':
        train_ds = torch.load(f'{data_dir}/celebA_pytorch/celebA_train.pt')
        val_ds = torch.load(f'{data_dir}/celebA_pytorch/celebA_val.pt')
        test_ds = torch.load(f'{data_dir}/celebA_pytorch/celebA_test.pt')
    
    elif data == 'synth':
        train_ds =  torch.load(f'{data_dir}/synthData_pytorch/synthData_tr.pt')
        val_ds = torch.load(f'{data_dir}/synthData_pytorch/synthData_val.pt')
        test_ds = torch.load(f'{data_dir}/synthData_pytorch/synthData_test.pt')
        
    return train_ds, val_ds, test_ds
    


def gen_synth_data(ds_size=6000):
    pr_priv_fav=2/3
    pr_unpriv_fav=1/3
    
    n_priv_fav = int((ds_size//2) * pr_priv_fav)
    n_unpriv_fav =int((ds_size//2) * pr_unpriv_fav)
    n_priv_unfav = int((ds_size//2) * (1-pr_priv_fav))
    n_unpriv_unfav = int((ds_size//2) * (1-pr_unpriv_fav))
  
     
    x_priv_fav = torch.cat([torch.normal(mean=1, std=1, size=(n_priv_fav, 1)), torch.ones(n_priv_fav, 1)], dim=1)
    x_unpriv_fav = torch.cat([torch.normal(mean=1, std=1, size=(n_unpriv_fav, 1)), torch.zeros(n_unpriv_fav, 1)], dim=1)
    
    x_priv_unfav = torch.cat([torch.normal(mean=0, std=1, size=(n_priv_unfav, 1)), torch.ones(n_priv_unfav, 1)], dim=1)
    x_unpriv_unfav = torch.cat([torch.normal(mean=0, std=1, size=(n_unpriv_unfav, 1)), torch.zeros(n_unpriv_unfav, 1)], dim=1)
    
    synth_inputs = torch.cat((x_priv_fav, x_unpriv_fav, x_priv_unfav, x_unpriv_unfav), dim=0)
    synth_labels = torch.cat( (torch.ones(n_priv_fav + n_unpriv_fav), torch.zeros(n_priv_unfav + n_unpriv_unfav)), dim=0)
    synth_metas = torch.cat( (torch.ones(n_priv_fav), torch.zeros(n_unpriv_fav), torch.ones(n_priv_unfav), torch.zeros(n_unpriv_unfav)), dim=0)
    
    return DatasetWithMeta(synth_inputs, synth_labels, synth_metas)
    


def find_opt_threshold(logits, lbls):
    probs = logits.sigmoid()

    num_thresh = 200
    bacc_arr = torch.zeros(num_thresh)
    thresholds = torch.linspace(0.01, 0.99, num_thresh)
    for idx, thresh in enumerate(thresholds):
        preds = probs > thresh
        bacc_arr[idx] = sk.balanced_accuracy_score(lbls.cpu(), preds.cpu())

    best_ind = torch.argmax(bacc_arr).item()
    return thresholds[best_ind]

def get_masks(lbls, metas):
    p_idx, up_idx = metas == 1, metas == 0
    fav_idx, unfav_idx = lbls == 1, lbls == 0
    
    p_fav_idx, p_unfav_idx = torch.logical_and(p_idx, fav_idx), torch.logical_and(p_idx, unfav_idx)
    up_fav_idx, up_unfav_idx = torch.logical_and(up_idx, fav_idx), torch.logical_and(up_idx, unfav_idx)
    
    return (p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx)   



def get_weighted_loss(logits, lbls, metas, loss, weights):

    (p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx) = get_masks(lbls, metas)
    if len(weights) == 4:
        loss[p_fav_idx] *= weights[0]
        loss[p_unfav_idx] *= weights[1]
        loss[up_fav_idx] *= weights[2]
        loss[up_unfav_idx] *= weights[3]
    
    else:
        thresholds = torch.linspace(0, 1, (len(weights) // 4) + 1)
        probs = logits.sigmoid()
        for i in range(len(thresholds)-1):
            p_fav_idx_th = torch.logical_and(p_fav_idx, (probs >= thresholds[i]) & (probs < thresholds[i+1]))
            p_unfav_idx_th = torch.logical_and(p_unfav_idx, (probs >= thresholds[i]) & (probs < thresholds[i+1]))
            up_fav_idx_th = torch.logical_and(up_fav_idx, (probs >= thresholds[i]) & (probs < thresholds[i+1]))
            up_unfav_idx_th = torch.logical_and(up_unfav_idx, (probs >= thresholds[i]) & (probs < thresholds[i+1]))
            
            loss[p_fav_idx_th] *= weights[i*4]
            loss[p_unfav_idx_th] *= weights[i*4 + 1]
            loss[up_fav_idx_th] *= weights[i*4 + 2]
            loss[up_unfav_idx_th] *= weights[i*4 + 3]
            
    return loss

def get_logits(model, dataloader, device):
    # compute logits, i.e., pre-sigmoid outputs, of model
    logits, lbls, metas = torch.tensor([], device=device),\
            torch.tensor([], device=device), torch.tensor([], device=device)
    
    for _, (_, inp, lbl, meta) in enumerate(dataloader):
        inp, lbl, meta = inp.to(device, non_blocking=True),\
                lbl.to(device, non_blocking=True).unsqueeze(1),\
                meta.to(device, non_blocking=True).unsqueeze(1)

        out = model(inp)
        
        logits = torch.cat((out, logits))
        lbls = torch.cat((lbl, lbls))
        metas = torch.cat((meta, metas))
    
    return logits, lbls, metas 




def infer(model, dataloader, device, threshold=0.5):
    logits, lbls, metas = get_logits(model, dataloader, device)
    if threshold is None:
        opt_threshold = find_opt_threshold(logits, lbls)
        return opt_threshold, get_metrics(logits, lbls, metas, threshold=opt_threshold)
    else:
       return threshold, get_metrics(logits, lbls, metas, threshold=threshold)


def get_metrics(logits, lbls, metas, threshold=0.5, preds=None):
    loss = F.binary_cross_entropy_with_logits(logits, lbls, reduction='none')
    probs = logits.sigmoid()
    if preds is None:
        preds = probs > threshold
    
    avg_loss = loss.mean().item()
    acc = sk.accuracy_score(lbls.cpu(), preds.cpu())
    bacc = sk.balanced_accuracy_score(lbls.cpu(), preds.cpu())   
    utility_metrics ={
        'loss':avg_loss,
        'acc':acc,
        'bacc':bacc,   
    }
    fairness_loss = get_fairness_loss(probs,  lbls, metas, loss=loss)
    fairness_metrics = get_fairness_metrics(preds, lbls, metas, loss)
    fairness_metrics['fairness_loss'] = fairness_loss.item()
    fairness_metrics['generalized fnr'] = probs.mean().item()
    fairness_metrics['generalized fpr'] = (1-probs.mean()).item()
    utility_metrics['total_loss'] = avg_loss + fairness_loss.item()

    return {**utility_metrics, **fairness_metrics}


def get_fairness_loss(logits, lbls, metas, loss=None):
    (p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx) = get_masks(lbls, metas)
    #probs = logits.sigmoid()
    
    fav_diff = torch.abs(loss[up_fav_idx].mean() - loss[p_fav_idx].mean())
    return fav_diff
    # unfav_diff = torch.abs(loss[up_unfav_idx].mean() - loss[p_unfav_idx].mean()) 
    # return (fav_diff + unfav_diff)/2



def get_fairness_metrics(preds, lbls, metas, loss):
    p_idx, up_idx = metas == 1, metas == 0
    p_loss, up_loss = loss[p_idx].mean().item(), loss[up_idx].mean().item()
    
    p_acc, up_acc = sk.accuracy_score(lbls[p_idx].cpu(), preds[p_idx].cpu()),\
                        sk.accuracy_score(lbls[up_idx].cpu(), preds[up_idx].cpu())
    
    p_bacc, up_bacc = sk.balanced_accuracy_score(lbls[p_idx].cpu(), preds[p_idx].cpu()),\
                    sk.balanced_accuracy_score(lbls[up_idx].cpu(), preds[up_idx].cpu())
    
    loss_diff = np.abs(p_loss - up_loss)
    acc_diff = np.abs(p_acc - up_acc)
    bacc_diff = np.abs(p_bacc - up_bacc)
    
    p_cm = sk.confusion_matrix(lbls[p_idx].cpu(), preds[p_idx].cpu())
    up_cm = sk.confusion_matrix(lbls[up_idx].cpu(), preds[up_idx].cpu())
    p_tn, p_fp, p_fn, p_tp = p_cm.ravel()
    up_tn, up_fp, up_fn, up_tp = up_cm.ravel()
    
    # classic fairness metrics such as aod etc.
    p_tpr, up_tpr = p_tp/(p_tp + p_fn), up_tp/(up_tp + up_fn) 
    p_fpr, up_fpr = p_fp/(p_fp + p_tn), up_fp/(up_fp + up_tn)
    p_fav_pr, up_fav_pr = (p_fp + p_tp)/(p_tn + p_fp + p_fn + p_tp),\
                    (up_fp + up_tp)/(up_tn + up_fp + up_fn + up_tp)
    
    eq_opp_diff = np.abs(up_tpr - p_tpr)
    avg_odds_diff = np.abs( np.abs(up_fpr - p_fpr) + np.abs(up_tpr - p_tpr))/2
    stat_par_diff = np.abs(up_fav_pr - p_fav_pr)
    disp_imp = (up_fav_pr/p_fav_pr)
    
    return {
        'loss_diff':loss_diff,
        'acc_diff':acc_diff,
        'bacc_diff':bacc_diff,
        'AOD':avg_odds_diff,
        'EOD':eq_opp_diff,
        'SPD':stat_par_diff,
        'DI':disp_imp
    }


def log_tensorboard(metrics, writer, ep, typ='Val', print_extra=False):
    #writer.add_scalar(f'Metric/{typ}/Acc', metrics['acc'], ep)
    writer.add_scalar(f'Metric/{typ}/BAcc', metrics['bacc'], ep)
    writer.add_scalar(f'Metric/{typ}/AOD', metrics['AOD'], ep)
    writer.add_scalar(f'Metric/{typ}/EOD', metrics['EOD'], ep)
    writer.add_scalar(f'Metric/{typ}/SPD', metrics['SPD'], ep)
    
    if typ == 'Val' or typ == 'Train':
        writer.add_scalar(f'Loss/{typ}/Loss', metrics['loss'], ep)
        writer.add_scalar(f'Loss/{typ}/Fairness_Loss', metrics['fairness_loss'], ep)
        writer.add_scalar(f'Loss/{typ}/Total_Loss', metrics['total_loss'], ep)
    
    
    if print_extra:   
        writer.add_scalar(f'Metric/{typ}/DI', metrics['DI'], ep)
        writer.add_scalar(f'Metric/{typ}/Acc_Diff', metrics['acc_diff'], ep)
        writer.add_scalar(f'Metric/{typ}/BAcc_Diff', metrics['bacc_diff'], ep)
        writer.add_scalar(f'Utility/{typ}/Total_Loss', metrics['total_loss'], ep)
        writer.add_scalar(f'Fairness/{typ}/Loss_Diff', metrics['loss_diff'], ep) 
    return 




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """ Taken from https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def inf_dataloader(dataloader):
    while True:
        for idx, (didx, inp, lbl, meta) in enumerate(dataloader):
            yield idx, (didx, inp, lbl, meta)
   

def KL_div(target, inp):
    # D_KL(target || inp)
    # making notation consistent with wikipedia's def.
    return F.kl_div(inp.log(), target, reduction='sum')
  
    
    

def get_MI(probs, metas, normalized=False):
    # returns I(yhat;s)/H(s): normalized MI between model and sensitive attribute
    p_idx, up_idx = metas == 1, metas == 0
    
    pr_yhat0 = (1-probs).mean()
    pr_yhat1 = probs.mean()
    pr_s0 = up_idx.sum().float()/len(metas)
    pr_s1 = p_idx.sum().float()/len(metas)

    pr_yhat0_given_s0 = (1-probs[up_idx]).mean()
    pr_yhat0_given_s1 = (1-probs[p_idx]).mean()
    pr_yhat1_given_s0 = probs[up_idx].mean()
    pr_yhat1_given_s1 = probs[p_idx].mean()

    pr_yhat0_and_s0 = pr_yhat0_given_s0*pr_s0
    pr_yhat0_and_s1 = pr_yhat0_given_s1*pr_s1
    pr_yhat1_and_s0 = pr_yhat1_given_s0*pr_s0
    pr_yhat1_and_s1 = pr_yhat1_given_s1*pr_s1


    pr_joint = torch.cat((pr_yhat0_and_s0.reshape(1,), pr_yhat0_and_s1.reshape(1,),\
                        pr_yhat1_and_s0.reshape(1,), pr_yhat1_and_s1.reshape(1,)))
    pr_marginals = torch.cat((pr_yhat0*pr_s0.reshape(1,), pr_yhat0*pr_s1.reshape(1,),\
                            pr_yhat1*pr_s0.reshape(1,),  pr_yhat1*pr_s1.reshape(1,)))
    
    if normalized:
        h_s = -(pr_s0*torch.log(pr_s0) + pr_s1*torch.log(pr_s1))
        return KL_div(pr_joint, pr_marginals)/h_s
    else:
        return KL_div(pr_joint, pr_marginals)



def get_cond_MI(probs, lbls, metas, normalized=False, global_stats=None):
    # mutual information between prediction and sensitive values conditioned on ground truth labels
    # I(yhat;sens | y)
    
    (p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx) = get_masks(lbls, metas)
    fav_idx, unfav_idx = lbls == 1, lbls == 0
    
    if global_stats is not None:
        pr_s0_given_y0 = global_stats[0]
        pr_s1_given_y0 = global_stats[1]
        pr_s0_given_y1 = global_stats[2]
        pr_s1_given_y1 = global_stats[3]
        pr_y0 = global_stats[4]
        pr_y1 = global_stats[5]
    else:
        pr_s0_given_y0 = (up_unfav_idx.sum().float()/unfav_idx.sum()).reshape(1,)
        pr_s1_given_y0 = (p_unfav_idx.sum().float()/unfav_idx.sum()).reshape(1,)
        pr_s0_given_y1 = up_fav_idx.sum().float()/fav_idx.sum().reshape(1,)
        pr_s1_given_y1 = p_fav_idx.sum().float()/fav_idx.sum().reshape(1,)
        pr_y0 = unfav_idx.sum().float()/len(lbls)
        pr_y1 = fav_idx.sum().float()/len(lbls)
        
    
    
    
    pr_yhat0_given_y0 = (1-probs[unfav_idx].mean()).reshape(1, )
    pr_yhat1_given_y0 = probs[unfav_idx].mean().reshape(1, )
    pr_marginals_given_y0 = torch.cat([pr_yhat0_given_y0*pr_s0_given_y0, pr_yhat0_given_y0*pr_s1_given_y0,\
                                pr_yhat1_given_y0*pr_s0_given_y0, pr_yhat1_given_y0*pr_s1_given_y0 ])

    
    pr_yhat0_given_s0_and_y0 = 1-probs[up_unfav_idx].mean().reshape(1,)
    pr_yhat0_given_s1_and_y0 = 1-probs[p_unfav_idx].mean().reshape(1,)
    pr_yhat1_given_s0_and_y0 = probs[up_unfav_idx].mean().reshape(1,)
    pr_yhat1_given_s1_and_y0 = probs[p_unfav_idx].mean().reshape(1,)
    pr_joint_given_y0 = torch.cat([pr_yhat0_given_s0_and_y0*pr_s0_given_y0, pr_yhat0_given_s1_and_y0*pr_s1_given_y0,\
                                    pr_yhat1_given_s0_and_y0*pr_s0_given_y0, pr_yhat1_given_s1_and_y0*pr_s1_given_y0])
    
    KL_div_given_y0 = KL_div(pr_joint_given_y0, pr_marginals_given_y0)
    
   
    pr_yhat0_given_y1 = 1-probs[fav_idx].mean().reshape(1,)
    pr_yhat1_given_y1 = probs[fav_idx].mean().reshape(1,)
    pr_marginals_given_y1 = torch.cat([pr_yhat0_given_y1*pr_s0_given_y1, pr_yhat0_given_y1*pr_s1_given_y1,\
                                pr_yhat1_given_y1*pr_s0_given_y1, pr_yhat1_given_y1*pr_s1_given_y1])
    
    pr_yhat0_given_s0_and_y1 = 1-probs[up_fav_idx].mean().reshape(1,)
    pr_yhat0_given_s1_and_y1 = 1-probs[p_fav_idx].mean().reshape(1,)
    pr_yhat1_given_s0_and_y1 = probs[up_fav_idx].mean().reshape(1,)
    pr_yhat1_given_s1_and_y1 = probs[p_fav_idx].mean().reshape(1,)
    pr_joint_given_y1 = torch.cat([pr_yhat0_given_s0_and_y1*pr_s0_given_y1, pr_yhat0_given_s1_and_y1*pr_s1_given_y1,\
                                    pr_yhat1_given_s0_and_y1*pr_s0_given_y1, pr_yhat1_given_s1_and_y1*pr_s1_given_y1])

    KL_div_given_y1 = KL_div(pr_joint_given_y1, pr_marginals_given_y1)
    

   
    if normalized:
        h_s_given_y0 = -(pr_s0_given_y0*torch.log(pr_s0_given_y0) + pr_s1_given_y0*torch.log(pr_s1_given_y0))
        h_s_given_y1 = -(pr_s0_given_y1*torch.log(pr_s0_given_y1) + pr_s1_given_y1*torch.log(pr_s1_given_y1))
        h_s_given_y = (pr_y0*h_s_given_y0 + pr_y1*h_s_given_y1).reshape([])
        return  (pr_y0*KL_div_given_y0 + pr_y1*KL_div_given_y1) / h_s_given_y
    else:
        return pr_y0*KL_div_given_y0 + pr_y1*KL_div_given_y1
       
    



# def infer(model, dataloader, device, threshold=0.5, bacc_limit=None, AOD_limit=None):
#     logits, lbls, metas = get_logits(model, dataloader, device)
#     if threshold is None:
#         if bacc_limit is None:
#             opt_threshold = find_opt_threshold(logits, lbls, metas)
#         else:
#             opt_threshold = find_opt_threshold(logits, lbls, metas, bacc_limit=bacc_limit, AOD_limit=AOD_limit)
            
#         return opt_threshold, get_metrics(logits, lbls, metas, threshold=opt_threshold)
#     else:
#        return threshold, get_metrics(logits, lbls, metas, threshold=threshold)



    
# def find_opt_threshold(logits, lbls, metas=None, bacc_limit=None, AOD_limit=None):
#     probs = logits.sigmoid()

#     num_thresh = 200
#     bacc_arr = torch.zeros(num_thresh)
#     thresholds = torch.linspace(0.01, 0.99, num_thresh)
#     for idx, thresh in enumerate(thresholds):
#         metric = get_metrics(logits, lbls, metas, threshold=thresh)
#         if (bacc_limit is not None) and (metric['bacc'] >= bacc_limit and metric['AOD'] <= AOD_limit):
#             return thresh
         
#         bacc_arr[idx] = metric['bacc']
             
#     best_ind = torch.argmax(bacc_arr).item()
#     return thresholds[best_ind]

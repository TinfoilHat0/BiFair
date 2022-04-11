#!/usr/bin/env python
# coding: utf-8


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from time import ctime
from torch.utils.tensorboard import SummaryWriter
import my_utils as ut
import my_models as md
from my_utils import DatasetWithMetaCelebA, DatasetWithMeta
import fair_algs as fair
import options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore")




args = options.args_parser()
f_name = (
        f"{ctime()}_data:{args.data}_ARL:{1}_Tout:{args.T_out}"
        f"trBS:{args.tr_bs}_innerwd:{args.inner_wd}_"
        f"labelNoise:{args.label_noise}_demRatio:{args.dem_ratio}"
        )
writer = SummaryWriter(f'logs/{f_name}')



# datasets
tr_ds, v_ds, te_ds = ut.get_data(args.data)
if args.label_noise:
    tr_dem_ds, tr_ds = ut.get_dem_data(tr_ds, dem_size=0, dem_ratio=args.dem_ratio, equal_dist=args.equal_dist)
    tr_ds = ut.add_random_noise_to_labels(tr_ds, noise_ratio=args.label_noise)
    tr_ds.inputs = torch.cat((tr_ds.inputs, tr_dem_ds.inputs))
    tr_ds.metas = torch.cat((tr_ds.metas, tr_dem_ds.metas))
    tr_ds.labels = torch.cat((tr_ds.labels, tr_dem_ds.labels))


# dataloaders
tr_loader = DataLoader(tr_ds, args.tr_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                       persistent_workers=False)
v_loader = DataLoader(v_ds, args.tr_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                      persistent_workers=False)
te_loader = DataLoader(te_ds, args.tr_bs, shuffle=False, pin_memory=True, num_workers=args.num_workers)
tr_generator = ut.inf_dataloader(tr_loader)



# models
model_learner = md.get_model(args.data, input_size=tr_ds.inputs.shape[1]).to(args.device)
opt_learner = torch.optim.Adam(model_learner.parameters())

model_adv = md.get_model(args.data, input_size=(tr_ds.inputs.shape[1]+1)).to(args.device)
opt_adv = torch.optim.Adam(model_adv.parameters())

es = ut.EarlyStopping(patience=args.es_tol, verbose=True, path=f'../models/{f_name}.pt')

# training loop
tr_loss_sum = 0
for t_out in tqdm(range(args.T_out)):
    _, (_, tr_inp, tr_lbl, _) = next(tr_generator)
    tr_inp, tr_lbl = tr_inp.to(args.device, non_blocking=True),\
                tr_lbl.to(args.device, non_blocking=True).unsqueeze(1)
    
    # predict weights for the current batch with adversary
    adv_inp = torch.cat((tr_inp, tr_lbl), dim=1)
    w = model_adv(adv_inp).sigmoid()
    w = 1 + (w.shape[0]*(w/w.sum()))
    
    # training learner on weighted data
    tr_out = model_learner(tr_inp)
    tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction = 'none')
    tr_loss_sum += tr_loss.mean().item()
    
    tr_loss *= w.detach()
    opt_learner.zero_grad()
    tr_loss.mean().backward()
    opt_learner.step()
    
    # training adversary to maximize weighted loss
    adv_loss = -(tr_loss.detach()*w).mean()
    opt_adv.zero_grad()
    adv_loss.backward()
    opt_adv.step()
    
    # inference on validation
    if (t_out+1) % (len(tr_loader)*args.chkpt) == 0:
        with torch.no_grad():
            ep = t_out // len(tr_loader)
            model_learner.eval()
            tr_loss_sum /= (len(tr_loader)*args.chkpt)
            _, v_metrics  = ut.infer(model_learner, v_loader, args.device, threshold=0.5)
            writer.add_scalar(f'Loss/Train/Loss',  tr_loss_sum, ep)
            ut.log_tensorboard(v_metrics, writer, ep, typ='Val')
            model_learner.train()
            if ep > 15:
                es(v_metrics['loss'], model_learner)
                if es.early_stop:
                    print("Early stopping")
                    break
            tr_loss_sum = 0
 
            
with torch.no_grad():
    model_learner.eval()
    v_threshold, v_metrics = ut.infer(model_learner, v_loader, args.device, threshold=None)
    ut.log_tensorboard(v_metrics, writer, 0, typ='Val_Final')
    
    _, te_metrics = ut.infer(model_learner, te_loader, args.device, threshold=v_threshold)
    ut.log_tensorboard(te_metrics, writer, 0, typ='Test', print_extra=False)
    writer.flush()
    writer.close()
    print(v_metrics, te_metrics)
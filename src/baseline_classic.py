#!/usr/bin/env python
# coding: utf-8


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import higher
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
        f"{ctime()}_data:{args.data}_Tout:{args.T_out}_Tpred:{args.T_pred}_"
        f"trBS:{args.tr_bs}_demBS:{args.dem_bs}_innerwd:{args.inner_wd}_"
        f"outerwd:{args.outer_wd}_demLabeled:{args.use_dem_labeled}_equalDist:{args.equal_dist}_"
        f"demRatio:{args.dem_ratio}_prjEta:{args.prj_eta}_kamiran:{args.kamiran}_labelNoise:{args.label_noise}"
        )
writer = SummaryWriter(f'logs/{f_name}')



# datasets
tr_ds, v_ds, te_ds = ut.get_data(args.data)
# partition training data to dem, nondem portions
if args.dem_ratio < 1 and args.dem_ratio > 0:
    tr_dem_ds, tr_ds = ut.get_dem_data(tr_ds, dem_size=0, dem_ratio=args.dem_ratio, equal_dist=args.equal_dist)
    print(len(tr_dem_ds), len(tr_ds), ut.get_ds_stats(tr_dem_ds), ut.get_ds_stats(tr_ds))

# poisoning the training dataset
if args.label_noise:
    tr_ds = ut.add_random_noise_to_labels(tr_ds, noise_ratio=args.label_noise)
    if not args.kamiran or not args.prj_eta:
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

if args.dem_ratio < 1 and args.dem_ratio > 0:
    dem_loader = DataLoader(tr_dem_ds, args.dem_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                        persistent_workers=False)
    dem_generator = ut.inf_dataloader(dem_loader)




# training a model to predict demographic features
if args.T_pred and (not args.use_dem_labeled) and (args.dem_ratio < 1 and args.dem_ratio > 0):
    model = md.get_model(args.data, input_size=tr_ds.inputs.shape[1]).to(args.device)
    opt = torch.optim.Adam(model.parameters(), weight_decay=args.outer_wd)
    
    for t_pred in tqdm(range(args.T_pred)):
        _, (_, dem_inp, _, dem_meta) = next(dem_generator)
        dem_inp, dem_meta = dem_inp.to(args.device, non_blocking=True),\
                    dem_meta.to(args.device, non_blocking=True).unsqueeze(1)

        dem_out = model(dem_inp)
        dem_loss = F.binary_cross_entropy_with_logits(dem_out, dem_meta, reduction='mean')
        
        opt.zero_grad()
        dem_loss.backward()
        opt.step()


    # filling demographics on nondem part of training dataset
    tr_ds, dem_tr_acc = fair.fill_demographics(model, args, tr_loader)
    # add dem portion to the tr_ds
    tr_ds.inputs = torch.cat((tr_ds.inputs, tr_dem_ds.inputs.to(args.device)))
    tr_ds.metas = torch.cat((tr_ds.metas, tr_dem_ds.metas.to(args.device)))
    tr_ds.labels = torch.cat((tr_ds.labels, tr_dem_ds.labels.to(args.device)))
    # new loader& generator
    tr_loader = DataLoader(tr_ds, args.tr_bs, shuffle=True, pin_memory=False, num_workers=args.num_workers,\
                            persistent_workers=False)
    tr_generator = ut.inf_dataloader(tr_loader)

    writer.add_scalar(f'Dem_Acc',  dem_tr_acc, 0)
    print(f'Dem_Train_Acc:{dem_tr_acc}')
 
  
if args.kamiran:
    if args.use_dem_labeled:
        w = fair.get_kamiran_weights(tr_dem_ds)
    else:
        w = fair.get_kamiran_weights(tr_ds)


# model, opt.
model = md.get_model(args.data, input_size=tr_ds.inputs.shape[1]).to(args.device)
opt = torch.optim.Adam(model.parameters(), weight_decay=args.outer_wd)
es = ut.EarlyStopping(patience=args.es_tol, verbose=True, path=f'../models/{f_name}.pt')


# training loop
tr_loss_sum=0
for t_out in tqdm(range(args.T_out)):
    if args.use_dem_labeled:
        _, (_, tr_inp, tr_lbl, tr_meta) = next(dem_generator)
    else:
        _, (_, tr_inp, tr_lbl, tr_meta) = next(tr_generator)
    
    tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                            tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                            tr_meta.to(args.device, non_blocking=True).unsqueeze(1)
    tr_out = model(tr_inp)
    tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction='none')
    tr_loss_sum += tr_loss.mean().item()
    
    if args.prj_eta > 0:
        prj_idx = fair.get_prj_idx(tr_out.sigmoid(), tr_meta) 
        tr_loss = tr_loss.mean() + args.prj_eta*prj_idx
    
    elif args.kamiran:
        masks = ut.get_masks(tr_lbl, tr_meta)
        for i in range(4):
            tr_loss[masks[i]] *= w[i]
        tr_loss = tr_loss.mean()
        
    else:
        tr_loss = tr_loss.mean()
    
    opt.zero_grad()
    tr_loss.backward()
    opt.step()
    
    if (t_out+1) % (len(tr_loader)*args.chkpt) == 0:
        with torch.no_grad():
            ep = t_out // len(tr_loader)
            model.eval()
            tr_loss_sum /= (len(tr_loader)*args.chkpt)
            _, v_metrics  = ut.infer(model, v_loader, args.device, threshold=0.5)
            writer.add_scalar(f'Loss/Train/Loss',  tr_loss_sum, ep)
            ut.log_tensorboard(v_metrics, writer, ep, typ='Val')
            es(v_metrics['loss'], model)
            if ep > 15:
                if es.early_stop:
                    print("Early stopping")
                    break
            model.train()
            tr_loss_sum = 0


# post-training inference
with torch.no_grad():
    model.eval()
    v_threshold, v_metrics = ut.infer(model, v_loader, args.device, threshold=None)
    ut.log_tensorboard(v_metrics, writer, 0, typ='Val')
    
    tr_threshold, tr_metrics = ut.infer(model, tr_loader, args.device, threshold=None)
    ut.log_tensorboard(tr_metrics, writer, 0, typ='Train')
    
    _, te_metrics = ut.infer(model, te_loader, args.device, threshold=v_threshold)
    ut.log_tensorboard(te_metrics, writer, 0, typ='Test', print_extra=False)
    writer.flush()
    writer.close()
    print(v_metrics, te_metrics)
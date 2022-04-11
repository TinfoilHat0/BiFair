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
args.bilevel=1
f_name = (
        f"{ctime()}_bilevel_data:{args.data}_Tout:{args.T_out}_Tin:{args.T_in}_"
        f"trBS:{args.tr_bs}_demBS:{args.dem_bs}_innerwd:{args.inner_wd}_"
        f"outerwd:{args.outer_wd}_equalDist:{args.equal_dist}_demLabeled:{args.use_dem_labeled}_"
        f"demRatio:{args.dem_ratio}_labelNoise:{args.label_noise}"
        )
writer = SummaryWriter(f'logs/{f_name}')


# datasets
tr_ds, v_ds, te_ds = ut.get_data(args.data)
tr_dem_ds, tr_ds = ut.get_dem_data(tr_ds, dem_size=0, dem_ratio=args.dem_ratio, equal_dist=args.equal_dist)
print(len(tr_dem_ds), ut.get_ds_stats(tr_dem_ds))
if args.label_noise:
    tr_ds = ut.add_random_noise_to_labels(tr_ds, noise_ratio=args.label_noise)
if args.dem_noise:
    tr_ds = ut.add_random_noise_to_dems(tr_ds, noise_ratio=args.dem_noise)

    
# join dem_ds, tr_ds to tr_dsAS
tr_ds.inputs = torch.cat((tr_ds.inputs, tr_dem_ds.inputs))
tr_ds.metas = torch.cat((tr_ds.metas, tr_dem_ds.metas))
tr_ds.labels = torch.cat((tr_ds.labels, tr_dem_ds.labels))


# dataloaders
tr_loader = DataLoader(tr_ds, args.tr_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                       persistent_workers=False)
v_loader = DataLoader(v_ds, args.tr_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                      persistent_workers=False)
te_loader = DataLoader(te_ds, args.tr_bs, shuffle=False, pin_memory=True, num_workers=args.num_workers)
dem_loader = DataLoader(tr_dem_ds, args.dem_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
                           persistent_workers=False)

dem_generator = ut.inf_dataloader(dem_loader)
tr_generator = ut.inf_dataloader(tr_loader)

model = md.get_model(args.data, input_size=tr_ds.inputs.shape[1]).to(args.device)
opt = torch.optim.Adam(model.parameters(), weight_decay=args.inner_wd)
es = ut.EarlyStopping(patience=args.es_tol, verbose=True, path=f'../models/{f_name}.pt')


w = torch.rand((len(tr_ds)), 1, requires_grad=False, device=args.device)
#w /= torch.norm(w, p=1)
#w.clip_(min=0) 
w.requires_grad=True
opt_weights = torch.optim.Adam([w], weight_decay=args.outer_wd)


tr_loss_sum = 0
for t_out in tqdm(range(args.T_out)):
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
        # inner optimization
        for t_in in range(args.T_in):
            if args.use_dem_labeled:
                _, (tr_didx, tr_inp, tr_lbl, _) = next(dem_generator)
            else:
                _, (tr_didx, tr_inp, tr_lbl, _) = next(tr_generator)
            tr_inp, tr_lbl = tr_inp.to(args.device, non_blocking=True),\
                        tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\

            tr_out = fmodel(tr_inp)
            tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction = 'none')
            tr_loss_sum += tr_loss.mean().item()
            tr_loss *= w[tr_didx]
            diffopt.step(tr_loss.mean())

        # outer optimization
        _, (_, dem_inp, dem_lbl, dem_meta) = next(dem_generator)                             
        dem_inp, dem_lbl, dem_meta = dem_inp.to(args.device, non_blocking=True),\
                        dem_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                        dem_meta.to(args.device, non_blocking=True).unsqueeze(1)

        dem_out = fmodel(dem_inp)
        dem_util_loss = F.binary_cross_entropy_with_logits(dem_out, dem_lbl, reduction='none')
        dem_fair_loss = ut.get_fairness_loss(dem_out, dem_lbl, dem_meta, loss=dem_util_loss)
        dem_total_loss = dem_fair_loss + args.util_lambda*dem_util_loss.mean()

        opt_weights.zero_grad()
        w.grad = torch.autograd.grad(dem_total_loss, w)[0]
        opt_weights.step()
        with torch.no_grad():
            #w /= torch.norm(w, p=1)
            #w.clip_(min=0)
            model.load_state_dict(fmodel.state_dict())
             
    if (t_out+1) % (len(tr_loader)*args.chkpt) == 0:
        with torch.no_grad():
            ep = t_out // len(tr_loader)
            model.eval()
            tr_loss_sum /= (len(tr_loader)*args.chkpt*args.T_in)
            _, v_metrics  = ut.infer(model, v_loader, args.device, threshold=0.5)
            _, dem_metrics  = ut.infer(model, dem_loader, args.device, threshold=0.5)
            ut.log_tensorboard(v_metrics, writer, ep, typ='Val')
            ut.log_tensorboard(dem_metrics, writer, ep, typ='Dem')
            writer.add_scalar(f'Loss/Train/Loss',  tr_loss_sum, ep)
            
            if ep > 15:
                es(v_metrics['loss'], model)
                if es.early_stop:
                    print("Early stopping")
                    break
            model.train()
            tr_loss_sum = 0

# post-training inference      
with torch.no_grad():
    model.eval()
    v_threshold, v_metrics = ut.infer(model, v_loader, args.device, threshold=None)
    ut.log_tensorboard(v_metrics, writer, 0, typ='Val_Final')
    
    _, te_metrics = ut.infer(model, te_loader, args.device, threshold=v_threshold)
    ut.log_tensorboard(te_metrics, writer, 0, typ='Test', print_extra=False)
    writer.flush()
    writer.close()
    print(v_metrics, te_metrics)
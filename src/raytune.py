#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import higher
from time import ctime
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, HTML
import sklearn.metrics as sk

from torchsummary import summary
import my_utils as ut
import my_models as md
from my_utils import DatasetWithMetaCelebA, DatasetWithMeta
import fair_algs as fair
import options

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import fair_algs as fair

import ray
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

args = options.args_parser()




def train(config, args, data_dir=None):
    
   # datasets and dataloaders
    tr_ds, v_ds, te_ds = ut.get_data(args.data, data_dir)
    tr_loader = DataLoader(tr_ds, config['tr_bs'], shuffle=True, pin_memory=True, num_workers=args.num_workers, persistent_workers=True)
    v_loader = DataLoader(v_ds, config['v_bs'], shuffle=True, pin_memory=True, num_workers=args.num_workers, persistent_workers=True)
    v_generator = ut.inf_dataloader(v_loader)
    tr_generator = ut.inf_dataloader(tr_loader)


    #models, optimizer and LR scheduler
    if args.data =='celebA':
        model = md.get_model(args.data)
    else:
        model = md.get_model(args.data, input_size=tr_ds.inputs.shape[1])
    model.to(args.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), weight_decay=config['inner_wd'])

    # meta-weights
    if args.bilevel:
        #w = torch.rand( (config['weight_len'], 1), requires_grad=False, device=args.device)
        w = torch.rand((len(tr_ds), 1), requires_grad=False, device=args.device)
        w /= torch.norm(w, p=1)
        w.requires_grad=True
        opt_weights = torch.optim.Adam([w], weight_decay=config['outer_wd'])
    elif args.kamiran:
        w = fair.get_kamiran_weights(args.data, data_dir).to(args.device).view(-1, 1)

    # training loop
    for t_out in tqdm(range(args.T_out)):
        if args.bilevel:
            with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
                # inner optimization
                for _ in range(config['T_in']):
                    _, (tr_didx, tr_inp, tr_lbl, tr_meta) = next(tr_generator)
                    tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                                                tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                                tr_meta.to(args.device, non_blocking=True).unsqueeze(1)
                    
                    tr_out = fmodel(tr_inp)
                    tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction = 'none')
                    #tr_loss = ut.get_weighted_loss(tr_out, tr_lbl, tr_meta, tr_loss, w)
                    tr_loss *= w[tr_didx]
                    diffopt.step(tr_loss.mean())
                    
                # outer optimization
                _, (_, v_inp, v_lbl, v_meta) = next(v_generator)                             
                v_inp, v_lbl, v_meta = v_inp.to(args.device, non_blocking=True),\
                                        v_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                        v_meta.to(args.device, non_blocking=True).unsqueeze(1)

                v_out = fmodel(v_inp)
                v_util_loss = F.binary_cross_entropy_with_logits(v_out, v_lbl, reduction='none')
                v_fair_loss = ut.get_fairness_loss(v_out, v_lbl, v_meta, loss=v_util_loss)
                v_total_loss = v_util_loss.mean() +  args.fair_lambda*v_fair_loss
                # update weights
                opt_weights.zero_grad()
                w.grad = torch.autograd.grad(v_total_loss, w)[0]
                opt_weights.step()
                with torch.no_grad():
                    w /= torch.norm(w, p=1)
                    model.load_state_dict(fmodel.state_dict())
                    
        
        else:
            _, (tr_didx, tr_inp, tr_lbl, tr_meta) = next(tr_generator)
            tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                                    tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                    tr_meta.to(args.device, non_blocking=True).unsqueeze(1)

            tr_out = model(tr_inp)
            tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction='none')
            
            if args.kamiran:
                masks = ut.get_masks(tr_lbl, tr_meta)
                for i in range(4):
                    tr_loss[masks[i]] *= w[i]
                tr_loss = tr_loss.mean()
            
            
            elif args.regu:
                tr_fair_loss = ut.get_fairness_loss(tr_out, tr_lbl, tr_meta, loss=tr_loss)
                tr_loss = tr_loss.mean() + config['fair_lambda']*tr_fair_loss
                
                
            elif config['prj_eta'] > 0:
                prj_idx = fair.get_prj_idx(tr_out.sigmoid(), tr_meta) 
                tr_loss = tr_loss.mean() + config['prj_eta']*prj_idx
            
            else:
                tr_loss = tr_loss.mean()
                
            opt.zero_grad()
            tr_loss.backward()
            opt.step()
            
        if t_out % 10 == 0:
            with torch.no_grad():
                model.eval()
                _, v_metrics  = ut.infer(model, v_loader, args.device, threshold=0.5)
                tune.report(loss=v_metrics['loss'],  total_loss=v_metrics['loss'] + v_metrics['fairness_loss'],  BAcc=v_metrics['bacc'], EOD=v_metrics['EOD'], AOD=v_metrics['AOD'], SPD=v_metrics['SPD'])
                model.train()



def main(num_samples, max_iters, gpus_per_trial):
    data_dir = os.path.abspath("../data")
    print(data_dir)
    config = {
        'T_in':tune.choice([1, 2, 4]),
        #'weight_len':tune.choice([4, 8, 16]),
        'fair_lambda':tune.choice([0.5, 1, 2, 4, 8]),
        'outer_wd': tune.choice([0, 1e-4, 1e-3, 1e-2]),
        'v_bs': tune.choice([128, 256, 512]),
        
        #'prj_eta':tune.choice([0]),
        'tr_bs': tune.choice([128, 256, 512]),
        'inner_wd': tune.choice([0, 1e-4, 1e-3, 1e-2])
       
    }
    scheduler = ASHAScheduler(
        metric="total_loss",
        mode="min",
        max_t=max_iters,
        grace_period=1,
        reduction_factor=3,
        brackets=1)
    reporter = CLIReporter(
        metric_columns=['training_iteration', 'loss', 'total_loss', 'BAcc', 'EOD', 'AOD', 'SPD'])
    result = tune.run(
        partial(train, args=args, data_dir=data_dir),
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("total_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(best_trial.last_result)


if __name__ == "__main__":
    main(num_samples=30, max_iters=10, gpus_per_trial=2)





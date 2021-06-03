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


args = options.args_parser()


# datasets and dataloaders
tr_ds, v_ds, te_ds = ut.get_data(args.data)
tr_loader = DataLoader(tr_ds, args.tr_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers, persistent_workers=True)
v_loader = DataLoader(v_ds, args.v_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers, persistent_workers=True)
te_loader = DataLoader(te_ds, args.tr_bs, shuffle=False, pin_memory=True, num_workers=args.num_workers)
v_generator = ut.inf_dataloader(v_loader)
tr_generator = ut.inf_dataloader(tr_loader)



#models, optimizer and LR scheduler
if args.data =='celebA':
    model = md.get_model(args.data)
else:
    model = md.get_model(args.data, input_size=tr_ds.inputs.shape[1])
model.to(args.device)
model.train()
opt = torch.optim.Adam(model.parameters(), weight_decay=args.inner_wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=(args.es_tol//2 - 1), verbose=True)

# training dataset weights
if args.bilevel:
    w = torch.rand( (args.weight_len, 1), requires_grad=False, device=args.device)
    #w = torch.rand((len(tr_ds), 1), requires_grad=False, device=args.device)
    w /= torch.norm(w, p=1)
    #w.clip_(min=0) 
    w.requires_grad=True
    opt_weights = torch.optim.Adam([w], weight_decay=args.outer_wd)
elif args.kamiran:
    w = fair.get_kamiran_weights(args.data).to(args.device).view(-1, 1)



f_name = (f"{ctime()}_data:{args.data}_Tout:{args.T_out}_Tin:{args.T_in}_tr_bs:{args.tr_bs}_"
            f"v_bs:{args.v_bs}_innerwd:{args.inner_wd}_outerwd:{args.outer_wd}_prj:{args.prj_eta}_"
            f"post:{args.post}_kamiran:{args.kamiran}_bilevel:{args.bilevel}_"
            f"fairLambda:{args.fair_lambda}_wlen:{args.weight_len}_regu:{args.regu}"
        )
writer = SummaryWriter(f'logs/{f_name}') 
es = ut.EarlyStopping(patience=args.es_tol, verbose=True, path=f'models/{f_name}.pt')
tr_loss_sum = 0

# training loop
for t_out in tqdm(range(args.T_out)):
    if args.bilevel:
        with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
            # inner optimization
            for _ in range(args.T_in):
                tr_bidx, (tr_didx, tr_inp, tr_lbl, tr_meta) = next(tr_generator)
                tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                                            tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                            tr_meta.to(args.device, non_blocking=True).unsqueeze(1)
                
                tr_out = fmodel(tr_inp)
                tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction = 'none')
                tr_loss_sum += tr_loss.mean().item()
                tr_loss = ut.get_weighted_loss(tr_out, tr_lbl, tr_meta, tr_loss, w)
                #tr_loss *= w[tr_didx]
                diffopt.step(tr_loss.mean())
                
            # outer optimization
            _, (_, v_inp, v_lbl, v_meta) = next(v_generator)                             
            v_inp, v_lbl, v_meta = v_inp.to(args.device, non_blocking=True),\
                                    v_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                    v_meta.to(args.device, non_blocking=True).unsqueeze(1)

            v_out = fmodel(v_inp)
            v_util_loss = F.binary_cross_entropy_with_logits(v_out, v_lbl, reduction='none')
            v_fair_loss = ut.get_fairness_loss(v_out, v_lbl, v_meta, loss=v_util_loss)
            v_total_loss = v_util_loss.mean() + args.fair_lambda*v_fair_loss
            # update weights
            opt_weights.zero_grad()
            w.grad = torch.autograd.grad(v_total_loss, w)[0]
            opt_weights.step()
            with torch.no_grad():
                w /= torch.norm(w, p=1)
                #w.clip_(min=0)
                model.load_state_dict(fmodel.state_dict())
                
    
    else:
        tr_bidx, (tr_didx, tr_inp, tr_lbl, tr_meta) = next(tr_generator)
        tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                                tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                                tr_meta.to(args.device, non_blocking=True).unsqueeze(1)

        tr_out = model(tr_inp)
        tr_loss = F.binary_cross_entropy_with_logits(tr_out, tr_lbl, reduction='none')
        tr_loss_sum += tr_loss.mean().item()
        
        if args.kamiran:
            masks = ut.get_masks(tr_lbl, tr_meta)
            for i in range(4):
                tr_loss[masks[i]] *= w[i]
            tr_loss = tr_loss.mean()
        
        elif args.regu:
            tr_fair_loss = ut.get_fairness_loss(tr_out, tr_lbl, tr_meta, loss=tr_loss)
            tr_loss = tr_loss.mean() + args.fair_lambda*tr_fair_loss
                   
        elif args.prj_eta > 0:
            prj_idx = fair.get_prj_idx(tr_out.sigmoid(), tr_meta) 
            tr_loss = tr_loss.mean() + args.prj_eta*prj_idx
            
        else:
            tr_loss = tr_loss.mean()
            
        opt.zero_grad()
        tr_loss.backward()
        opt.step()
    
    # inference         
    if (t_out+1) % (len(tr_loader)*args.chkpt) == 0:
        with torch.no_grad():
            model.eval()
            ep = t_out // len(tr_loader)
            if args.bilevel:
                tr_loss_sum /= (len(tr_loader)*args.chkpt*args.T_in)
            else:
                tr_loss_sum /= (len(tr_loader)*args.chkpt)
            writer.add_scalar(f'Loss/Train/Loss',  tr_loss_sum, ep)
            _, v_metrics  = ut.infer(model, v_loader, args.device, threshold=0.5)
            ut.log_tensorboard(v_metrics, writer, ep, typ='Val')
            if args.bilevel or args.regu:
                stop_loss = v_metrics['loss'] + args.fair_lambda*v_metrics['fairness_loss']
            else:
                stop_loss = v_metrics['loss']                   
            es(stop_loss, model)
            if es.early_stop:
                print("Early stopping")
                break
            scheduler.step(stop_loss)
            model.train()
            tr_loss_sum = 0


with torch.no_grad():
    model.load_state_dict(torch.load(f'models/{f_name}.pt'))
    model.eval()
    v_threshold, v_metrics = ut.infer(model, v_loader, args.device, threshold=None)
    ut.log_tensorboard(v_metrics, writer, 0, typ='Val_Final')
    
    if args.post:
        post_te_metrics = fair.apply_roc_postproc(model, v_loader, te_loader, args.device, threshold=v_threshold)
        ut.log_tensorboard(post_te_metrics, writer, 0, typ='Test')
        args.post = 0
        f_name = (f"{ctime()}_data:{args.data}_Tout:{args.T_out}_Tin:{args.T_in}_bs:{args.bs}_"
            f"innerwd:{args.inner_wd}_outerwd:{args.outer_wd}_prj:{args.prj_eta}_"
            f"post:{args.post}_kamiran:{args.kamiran}_bilevel:{args.bilevel}_"
            f"fairLambda:{args.fair_lambda}_wlen:{args.weight_len}"
        )
        writer = SummaryWriter(f'logs/{f_name}') 
   
    _, te_metrics = ut.infer(model, te_loader, args.device, threshold=v_threshold)
    ut.log_tensorboard(te_metrics, writer, 0, typ='Test')
    writer.flush()
    writer.close()



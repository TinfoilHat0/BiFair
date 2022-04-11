import torch
from my_utils import get_logits, get_metrics, get_MI, DatasetWithMeta, inf_dataloader
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_models import get_model




def get_kamiran_weights(ds):
    pr_p = ds.metas.sum()/ds.metas.shape[0]
    pr_fav = ds.labels.sum()/ds.labels.shape[0]
    pr_up = 1-pr_p
    pr_unfav=1-pr_fav

    pr_exp_p_fav = pr_p*pr_fav
    pr_exp_p_unfav=pr_p*pr_unfav
    pr_exp_up_fav = pr_up*pr_fav
    pr_exp_up_unfav = pr_up*pr_unfav

    pr_obs_p_fav = ((ds.metas == 1) & (ds.labels == 1)).sum()/ds.metas.shape[0]
    pr_obs_p_unfav = ((ds.metas == 1) & (ds.labels == 0)).sum()/ds.metas.shape[0]
    pr_obs_up_fav = ((ds.metas == 0) & (ds.labels == 1)).sum()/ds.metas.shape[0]
    pr_obs_up_unfav = ((ds.metas == 0) & (ds.labels == 0)).sum()/ds.metas.shape[0]

    w_p_fav = pr_exp_p_fav/pr_obs_p_fav
    w_p_unfav = pr_exp_p_unfav/pr_obs_p_unfav
    w_up_fav = pr_exp_up_fav/pr_obs_up_fav
    w_up_unfav = pr_exp_up_unfav/pr_obs_up_unfav
    
    return (w_p_fav, w_p_unfav, w_up_fav, w_up_unfav)


def get_prj_idx(probs, metas):
    return get_MI(probs, metas, normalized=False)
    

def apply_roc_postproc(model, val_loader, test_loader, device, threshold=0.5, fairness_metric='EOD'):
    
    # find the best value for confidence interval (roc margin) using validation data
    v_logits, v_lbls, v_metas = get_logits(model, val_loader, device)
    v_probs = v_logits.sigmoid()
    
    num_margins=50
    fair_arr = torch.zeros(num_margins)
    roc_margins = torch.linspace(0.0, min(threshold, 1-threshold), num_margins)
    
    for idx, roc_margin in enumerate(roc_margins):

        low_conf_mask = torch.logical_and(v_probs >= (threshold-roc_margin), v_probs <= (threshold+roc_margin))
        p_idx, up_idx = v_metas == 1, v_metas == 0
        low_conf_p = torch.logical_and(low_conf_mask, p_idx)
        low_conf_up = torch.logical_and(low_conf_mask, up_idx)
        
        v_preds = v_probs > threshold
        v_preds[low_conf_p] = False
        v_preds[low_conf_up] = True
    
        cur_metrics = get_metrics(v_logits, v_lbls, v_metas, threshold, preds=v_preds)
        fair_arr[idx] = cur_metrics[fairness_metric]
      
        
    best_ind = torch.argmin(fair_arr)
    best_roc_margin = roc_margins[best_ind]
    te_logits, te_lbls, te_metas = get_logits(model, test_loader, device)
    te_probs = te_logits.sigmoid()
    
    low_conf_mask = torch.logical_and(te_probs >= (threshold-best_roc_margin),\
                                  te_probs <= (threshold+best_roc_margin))
    p_idx, up_idx = te_metas == 1, te_metas == 0
    low_conf_p = torch.logical_and(low_conf_mask, p_idx)
    low_conf_up = torch.logical_and(low_conf_mask, up_idx)

    te_preds = te_probs > threshold
    te_preds[low_conf_p] = False
    te_preds[low_conf_up] = True

    return get_metrics(te_logits, te_lbls, te_metas, threshold, preds=te_preds)


def fill_demographics(model, args, tr_loader):
    # model is pretrained to predict demographics,
    # fill demographics on training dataset
    with torch.no_grad():
        inputs, lbls, metas = torch.tensor([], device=args.device),\
                    torch.tensor([], device=args.device), torch.tensor([], device=args.device)
        n_correct = 0
        
        for _, (_, tr_inp, tr_lbl, tr_meta) in enumerate(tr_loader):
            tr_inp, tr_lbl, tr_meta = tr_inp.to(args.device, non_blocking=True),\
                    tr_lbl.to(args.device, non_blocking=True).unsqueeze(1),\
                    tr_meta.to(args.device, non_blocking=True).unsqueeze(1)
            
            tr_pred_meta = model(tr_inp).sigmoid() > 0.5
            n_correct += (tr_meta == tr_pred_meta).sum().item()
            
            inputs = torch.cat((tr_inp, inputs))
            lbls = torch.cat((tr_lbl, lbls))
            metas = torch.cat((tr_pred_meta, metas))
    
    # accuracy on training dataset for demographics
    dem_tr_acc = n_correct/metas.shape[0]        
    tr_ds = DatasetWithMeta(inputs, lbls.squeeze(), metas.squeeze())
    
    return tr_ds, dem_tr_acc 


# def get_kamiran_weights(data, data_dir='../data'):
#     if data == 'compas':
#         weights = torch.load(f'{data_dir}/compas_pytorch/kamiran_weights_compas.pt')
#     elif data == 'adult':
#         weights = torch.load(f'{data_dir}/adult_pytorch/kamiran_weights_adult.pt')
#     elif data == 'celebA':
#         weights = torch.load(f'{data_dir}/celebA_pytorch/kamiran_weights_celebA.pt')

#     return weights



# def train_for_demographics(v_ds, args, writer):
#     # training a model to predict demographics
#     v_loader = DataLoader(v_ds, args.bs, shuffle=True, pin_memory=True, num_workers=args.num_workers,\
#                       persistent_workers=True)
#     v_generator = inf_dataloader(v_loader)
#     model = get_model(args.data, input_size=(v_ds.inputs.shape[1]+1)).to(args.device)
#     opt = torch.optim.Adam(model.parameters(), weight_decay=args.inner_wd)
    
#     for t_meta in tqdm(range(args.T_meta)):
#     _, (_, v_inp, _, v_meta) = next(v_generator)
#     v_inp, v_meta = v_inp.to(args.device, non_blocking=True),\
#                 v_meta.to(args.device, non_blocking=True).unsqueeze(1)

#     v_out = model(v_inp)
#     v_loss = F.binary_cross_entropy_with_logits(v_out, v_meta, reduction='none')
#     v_loss = v_loss.mean()

#     opt.zero_grad()
#     v_loss.backward()
#     opt.step()
             
#     if (t_meta+1) % (len(v_loader)*args.chkpt) == 0:
#         with torch.no_grad():
#             ep = t_meta // len(v_loader)
#             model.eval()
#             _, v_metrics  = ut.infer(model, v_loader, args.device, threshold=0.5, predict_meta=True)
#             ut.log_tensorboard(v_metrics, writer, ep, typ='Val_Meta')
#             model.train()
    
    
    
    



    
    
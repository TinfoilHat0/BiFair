import torch
from my_utils import get_logits, get_metrics, get_MI



def get_kamiran_weights(data, data_dir='../data'):
    if data == 'compas':
        weights = torch.load(f'{data_dir}/compas_pytorch/kamiran_weights_compas.pt')
    elif data == 'adult':
        weights = torch.load(f'{data_dir}/adult_pytorch/kamiran_weights_adult.pt')
    elif data == 'celebA':
        weights = torch.load(f'{data_dir}/celebA_pytorch/kamiran_weights_celebA.pt')

    return weights


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
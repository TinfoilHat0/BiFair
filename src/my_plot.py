import torch
import my_utils as ut
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches



def plt_scatter(model, dataloader, device, threshold=0.5):
    logits, lbls, metas = ut.get_logits(model, dataloader, device)
    probs = logits.sigmoid()
    (p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx) = ut.get_masks(lbls, metas)                                                                  
    p_fav_probs, up_fav_probs = probs[p_fav_idx], probs[up_fav_idx]
    p_unfav_probs, up_unfav_probs = probs[p_unfav_idx], probs[up_unfav_idx]
    
    
    x_p_fav = p_fav_probs.detach().cpu()
    y_p_fav = ['(P,Fav)']*x_p_fav.shape[0]

    x_up_fav = up_fav_probs.detach().cpu()
    y_up_fav = ['(Up,Fav)']*x_up_fav.shape[0]

    x_p_unfav = p_unfav_probs.detach().cpu()
    y_p_unfav = ['(P,Unfav)']*x_p_unfav.shape[0]

    x_up_unfav = up_unfav_probs.detach().cpu()
    y_up_unfav = ['(Up,Unfav)']*x_up_unfav.shape[0]
        
    
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'figure.figsize': (7,5)})
    
    def pltcolor(lst):
        cols=[]
        for l in lst:
            if l < threshold:
                cols.append('red')
            else:
                cols.append('green')
        return cols
    
    cols_p_fav=pltcolor(x_p_fav)
    cols_up_fav=pltcolor(x_up_fav)
    cols_p_unfav=pltcolor(x_p_unfav)
    cols_up_unfav=pltcolor(x_up_unfav)

    plt.scatter(x_p_fav, y_p_fav, s=10, c=cols_p_fav, alpha=.5)
    plt.scatter(x_up_fav, y_up_fav, s=10, c=cols_up_fav, alpha=.5)
    plt.scatter(x_p_unfav, y_p_unfav, s=10, c=cols_p_unfav, alpha=.5)
    plt.scatter(x_up_unfav, y_up_unfav, s=10, c=cols_up_unfav, alpha=.5)
    plt.scatter(x_p_fav.mean(), ['(P,Fav)'], color="black")
    plt.scatter(x_up_fav.mean(), ['(Up,Fav)'], color="black")
    plt.scatter(x_p_unfav.mean(), ['(P,Unfav)'], color="black")
    plt.scatter(x_up_unfav.mean(), ['(Up,Unfav)'], color="black")
    plt.vlines(threshold, 0, 3, color='blue')

    plt.grid(True)
    green_patch = mpatches.Patch(color='green', label='Predicted as Fav')
    red_patch = mpatches.Patch(color='red', label='Predicted as Unfav')

    plt.legend(handles=[red_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2)
    #plt.ylabel('Groups')
    plt.xlabel('Probability')
    plt.tight_layout()
    plt.plot()
    plt.show()
    
    return
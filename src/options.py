import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default='adult',
        help='dataset')
    
    parser.add_argument('--T_out', type=int, default=10**6,
        help='max. num of iterations for the outer loop')
    
    parser.add_argument('--T_in', type=int, default=0,
        help='inner loop per outer loop')
    
    parser.add_argument('--tr_bs', type=int, default=256,
        help='training dataset batch size')
    
    parser.add_argument('--v_bs', type=int, default=256,
        help='validation dataset batch size')
    
    parser.add_argument('--inner_wd', type=float, default=0,
        help='weight_decay for model')
    
    parser.add_argument('--outer_wd', type=float, default=0,
        help='weight_decay data weights')
        
    parser.add_argument('--chkpt', type=int, default=1,
        help='how often record validation loss')
    
    parser.add_argument('--es_tol', type=int, default=10,
        help='stagnation = es_tol*chkpt')
    
    parser.add_argument('--bilevel', type=int, default=0,
        help='flag for bilevel')
    
    parser.add_argument('--fair_lambda', type=float, default=0,
        help='lambda param. for bilevel reweighing')
    
    parser.add_argument('--weight_len', type=int, default=0,
        help='length of weight vector')

    parser.add_argument('--kamiran', type=int, default=0,
        help='reweighing alg.w of kamiran and calders')
    
    parser.add_argument('--regu', type=int, default=0,
        help='adding L_f as a regularization term')
    
    parser.add_argument('--prj_eta', type=float, default=0,
        help='prejudice_remover')
    
    parser.add_argument('--post', type=int, default=0,
        help='post_processing')
    
    parser.add_argument('--device',  default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        help="To use cuda, set to a specific GPU ID.")
    
    parser.add_argument('--num_workers', type=int, default=2, 
        help="num. of workers for multithreading")
    
    args = parser.parse_args()
    return args
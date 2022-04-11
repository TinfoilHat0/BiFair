import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    
     # training params
    parser.add_argument('--data', type=str, default='adult',
        help='dataset')
    
    parser.add_argument('--T_out', type=int, default=10**6,
        help='max. num of iterations for the outer loop')
    
    parser.add_argument('--T_in', type=int, default=2,
        help='inner iterations for BiFair')

    parser.add_argument('--T_pred', type=int, default=0,
        help='prediction iterations')
    
    parser.add_argument('--tr_bs', type=int, default=512,
        help='training dataset batch size')
    
    parser.add_argument('--dem_bs', type=int, default=512,
        help='dem. dataset batch size')
    
    parser.add_argument('--inner_wd', type=float, default=0,
        help='weight_decay for model')
    
    parser.add_argument('--outer_wd', type=float, default=0,
        help='weight_decay data weights')
    
    # fairness params
    parser.add_argument('--bilevel', type=int, default=0,
        help='flag for bilevel')
    
    parser.add_argument('--util_lambda', type=float, default=0,
        help='lambda param. for bilevel reweighing')
    
    parser.add_argument('--kamiran', type=int, default=0,
        help='reweighing alg.w of kamiran and calders')
    
    parser.add_argument('--prj_eta', type=float, default=0,
        help='prejudice_remover')
    
    parser.add_argument('--use_dem_labeled', type=int, default=0,
        help='only use demographic labeled part to train on')
    
    parser.add_argument('--dem_ratio', type=float, default=0.0,
        help='ratio of training data portion that has demographic label')
    
    parser.add_argument('--equal_dist', type=int, default=0,
        help='whether dem. labeled portion as equal dist. across slices')
   
   
    # adversarial stuff
    parser.add_argument('--label_noise', type=float, default=0,
        help='label noise ratio')
    
    parser.add_argument('--dem_noise', type=float, default=0,
        help='label noise ratio')
    
    
    # others
    parser.add_argument('--chkpt', type=int, default=1,
        help='how often record validation loss')
    
    parser.add_argument('--es_tol', type=int, default=5,
        help='stagnation = es_tol*chkpt')
    
    parser.add_argument('--device',  default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        help="To use cuda, set to a specific GPU ID.")
    
    parser.add_argument('--num_workers', type=int, default=0, 
        help="num. of workers for multithreading")
    
    args = parser.parse_args()
    return args
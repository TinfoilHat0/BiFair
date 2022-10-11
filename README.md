# BiFair

This is the code for our paper titled [Fair Machine Learning under Limited Demographically Labeled Data](https://arxiv.org/abs/2106.04757) which appeared at [ICLR SRML (2022)](https://iclrsrml.github.io/). It is implemented and tested in PyTorch 1.9.0.

Datasets are available in the repo, but you can pre-process them differently if you wish by using ```notebooks/aif360_data_prepare.ipynb```.
See ```src/runner.sh``` for some example usage, and to replicate our results, as well as to run your own experiments.

For example, you can run unconstrained training, and [ARL](https://arxiv.org/abs/2006.13114) fair training algorithm by specifying your dataset as follows:
```
# unconstrained
python baseline_classic.py --data=$data &
# ARL
python baseline_ARL.py --data=$data --device=cuda:1
```
where ```$data``` can either be ```adult``` or ```bank```.

Similarly, you can run [Kamiran Reweighing](https://link.springer.com/article/10.1007/s10115-011-0463-8), and [Prejudice Remover](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3) with strawman adaptation, and our algorithm BiFair in the limited demographics setting as,
```
# Kamiran
python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --kamiran=1  --T_pred=1000 &
# Prj. Rem.
python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --prj_eta=1  --T_pred=1000 &
# BiFair
python bifair_with_iter.py --data=$data --dem_ratio=$dem_ratio --device=cuda:1
 ```
 where ```--dem_ratio``` specifies the ratio of the size of our demographically labeled portion to the whole training data, and ```--T_pred``` specifies the number of training iterations for the model which learns to predict demographic features.
 
 Finally, you can add label noise, test the algorithm under such a setting. For example:
 
 ```python baseline_ARL.py --data=$data --dem_ratio=0.01 --label_noise=0.5```
 
Note that, in this setting, we assume demographically labeled portion has clean labels, and label noise ratio is defined over the remaining training data. So, for the example line this means ``0.01%`` of training data has demographic labels, and clean class labels, whereas the ```50%``` of the remaining data has noisy class labels.

You can also specify hyperparams of training such as batch sizes, weight-decay etc.

```
bifair_with_iter.py [-h] [--data DATA] [--T_out T_OUT] [--T_in T_IN] [--T_pred T_PRED] [--tr_bs TR_BS] [--dem_bs DEM_BS] [--inner_wd INNER_WD] [--outer_wd OUTER_WD] [--bilevel BILEVEL] [--util_lambda UTIL_LAMBDA] [--kamiran KAMIRAN] [--prj_eta PRJ_ETA] [--use_dem_labeled USE_DEM_LABELED] [--dem_ratio DEM_RATIO] [--equal_dist EQUAL_DIST]
                           [--label_noise LABEL_NOISE] [--dem_noise DEM_NOISE] [--chkpt CHKPT] [--es_tol ES_TOL] [--device DEVICE] [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset
  --T_out T_OUT         max. num of iterations for the outer loop
  --T_in T_IN           inner iterations for BiFair
  --T_pred T_PRED       prediction iterations
  --tr_bs TR_BS         training dataset batch size
  --dem_bs DEM_BS       dem. dataset batch size
  --inner_wd INNER_WD   weight_decay for model
  --outer_wd OUTER_WD   weight_decay data weights
  --bilevel BILEVEL     flag for bilevel
  --util_lambda UTIL_LAMBDA
                        lambda param. for bilevel reweighing
  --kamiran KAMIRAN     reweighing alg.w of kamiran and calders
  --prj_eta PRJ_ETA     prejudice_remover
  --use_dem_labeled USE_DEM_LABELED
                        only use demographic labeled part to train on
  --dem_ratio DEM_RATIO
                        ratio of training data portion that has demographic label
  --equal_dist EQUAL_DIST
                        whether dem. labeled portion as equal dist. across slices
  --label_noise LABEL_NOISE
                        label noise ratio
  --dem_noise DEM_NOISE
                        label noise ratio
  --chkpt CHKPT         how often record validation loss
  --es_tol ES_TOL       stagnation = es_tol*chkpt
  --device DEVICE       To use cuda, set to a specific GPU ID.
  --num_workers NUM_WORKERS
                        num. of workers for multithreading
 ```

## Citation

```bibtex
@article{ozdayi2021bifair,
  title={Fair Machine Learning under Limited Demographically Labeled Data},
  author={Ozdayi, Mustafa Safa and Kantarcioglu, Murat and Iyer, Rishabh},
  journal={arXiv preprint arXiv:2106.04757},
  year={2021}
}


rm -rf logs


unconstrained and ARL
for data in bank adult
do
    for i in {1..10}
    do
        # unconstrained
        python baseline_classic.py --data=$data &

        # ARL
        python baseline_ARL.py --data=$data --device=cuda:1
    done
done

#strawman kamiran and prj. remover
for data in adult bank
do
    for dem_ratio in 1 0.1 0.01 0.001
    do
        for i in {1..10}
        do
            # Kamiran
            python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --kamiran=1  --T_pred=1000 &

            # Prj. Rem.
            python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --prj_eta=1  --T_pred=1000 &

            # BiFair
            python bifair_with_iter.py --data=$data --dem_ratio=$dem_ratio --device=cuda:1 &

            # equal dist=1 ensures we have a uniform distribution in demographically labeled subsets.
            # this prevents overfitting in BiFair when dem_ratio < 1, 
            # but it either doesn't seem to have a significant effect on strawman approach, 
            # or worsens it in some cases according to my exper 
            python bifair_with_iter.py --data=$data --dem_ratio=$dem_ratio --equal_dist=1 --device=cuda:1 
            
        done
    done
done


# noisy labels
# every algorithm gets 0.1% percent clean data, rest of the training data has 50% label noise
dem_ratio=0.001
label_noise=0.5
for data in adult bank
do
    for i in {1..10}
    do
        # unconstrained
        python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --label_noise=$label_noise --equal_dist=1 &

        # ARL
        python baseline_ARL.py --data=$data --dem_ratio=$dem_ratio --label_noise=$label_noise --equal_dist=1 &

        # Kamiran
        python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --dem_ratio=$dem_ratio --label_noise=$label_noise --kamiran=1  --T_pred=1000 --equal_dist=1 &

        # Prj. Rem
        python baseline_classic.py --data=$data --dem_ratio=$dem_ratio --dem_ratio=$dem_ratio --label_noise=$label_noise --prj_eta=1  --T_pred=1000 --equal_dist=1 --device=cuda:1 &

        # BiFair
        # util_lambda is a scalar hyperparameter on the utility loss that's computed over the clean data
        python bifair_with_iter.py --data=$data --dem_ratio=$dem_ratio --label_noise=$label_noise --equal_dist=1 --util_lambda=2 --device=cuda:1 &

    done
done


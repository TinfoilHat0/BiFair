rm -rf logs
T_out=1000000



for data in compas adult celebA 
do
    for bs in 128 256 512
    do
    
        for wd in 0 0.0001 0.001
        do

            for run in 1 2 3
            do

                #  default, kamiran
                python main.py --data=$data --bs=$bs --inner_wd=$wd --post=1 --device=cuda:0  
                python main.py --data=$data --bs=$bs --inner_wd=$wd --kamiran=1 --device=cuda:1 

                #  prj remover
                python main.py --data=$data --bs=$bs --inner_wd=$wd --prj_eta=1  --device=cuda:1 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --prj_eta=2  --device=cuda:0  
                python main.py --data=$data --bs=$bs --inner_wd=$wd --prj_eta=4  --device=cuda:1 

                
                # regularization
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=.5  --regu=1 --device=cuda:0 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=1  --regu=1 --device=cuda:1 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=2  --regu=1 --device=cuda:0 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=4  --regu=1 --device=cuda:1 


                # BiFair
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=.5  --bilevel=1 --weight_len=8 --device=cuda:0 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=1  --bilevel=1 --weight_len=8 --device=cuda:1 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=2  --bilevel=1 --weight_len=8 --device=cuda:0 
                python main.py --data=$data --bs=$bs --inner_wd=$wd --fair_lambda=4  --bilevel=1 --weight_len=8 --device=cuda:1 
                 

            done
        done
    done
done



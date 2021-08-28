# # # link_predict experiments
# for i in 16 32 64 128
#     do
#         for j in 100 1000 10000 100000
#             do
#                 echo "python link_predict.py -d FB15k --gpu 0 --eval-protocol raw --n-bases 1 --n-hidden $i --graph-batch-size $j"
# 
#                 python link_predict.py -d FB15k --gpu 0 --eval-protocol raw --n-bases 1 --n-hidden $i --graph-batch-size $j
#             done
#     done

# entity_classify_mp experiments
for i in 16 32 64 128
    do
        for j in 100 1000 10000 100000
            do
                echo "python entity_classify_mp.py -d ogbn-mag --testing --lr 0.01 --num-worker 1 --gpu 0 --dropout 0.7  --n-epochs 20 --batch-size $j --n-hidden $i"

                python entity_classify_mp.py -d ogbn-mag --testing --lr 0.01 --num-worker 1 --gpu 0 --dropout 0.7  --n-epochs 20 --batch-size $j --n-hidden $i

            done
    done

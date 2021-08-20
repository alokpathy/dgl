# link_predict experiments
# for i in 16 64 256 1024
#     do
#         for j in 30000 60000 90000 120000
#             do
#                 echo "python link_predict.py -d FB15k --gpu 0 --eval-protocol raw --n-bases 1 --n-hidden $i --graph-batch-size $j"
# 
#                 python link_predict.py -d FB15k --gpu 0 --eval-protocol raw --n-bases 1 --n-hidden $i --graph-batch-size $j
#             done
#     done

# entity_classify_mp experiments
for i in 16 64 256 1024
    do
        for j in 4 16 64 256
            do
                echo "python entity_classify_mp.py -d ogbn-mag --testing --lr 0.01 --num-worker 1 --gpu 0 --dropout 0.7  --n-epochs 20 --fanout $j,$j --n-hidden $i --low-mem"

                python entity_classify_mp.py -d ogbn-mag --testing --lr 0.01 --num-worker 1 --gpu 0 --dropout 0.7  --n-epochs 20 --fanout $j,$j --n-hidden $i --low-mem

            done
    done

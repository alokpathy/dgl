python ../../../../tools/launch.py \
--workspace $SCRATCH/dgl-intel/dgl/examples/pytorch/graphsage/experimental \
--num_trainers 1 \
--num_samplers 4 \
--num_servers 1 \
--part_config data/reddit.json \
--ip_config ip_config.txt \
"/usr/common/software/pytorch/1.7.1/bin/python train_dist.py --graph_name reddit --ip_config ip_config.txt --num_epochs 30 --batch_size 1000"

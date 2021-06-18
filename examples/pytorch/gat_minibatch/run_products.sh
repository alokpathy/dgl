export PATH="/home/ubuntu/anaconda3/envs/dgl-conda/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/home/ubuntu/anaconda3/condabin:/home/ubuntu/.dl_binaries/bin:/usr/local/cuda/bin:/opt/aws/neuron/bin:/home/ubuntu/anaconda3/condabin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/home/ubuntu/anaconda3/condabin:/home/ubuntu/.dl_binaries/bin:/usr/local/cuda/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin":$PATH
conda activate dgl-conda
python train_sampling.py --dataset ogbn-products --num-epochs 5

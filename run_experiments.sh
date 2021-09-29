#!/bin/bash

MY_PYTHON="srun -p VI_Face_1080TI --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=Hello --kill-on-bad-exit=1 python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda yes  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda yes  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"
CIFAR_10i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar10.pt           --cuda yes --seed 0"
TinyImageNet="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 20 --log_every 100 --samples_per_task 2500 --data_file tiny-imagenet-200.pt           --cuda yes --seed 0"


# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_permutations.py \
	--o mnist_permutations.pt \
	--seed 0 \
	--n_tasks 20

$MY_PYTHON mnist_rotations.py \
	--o mnist_rotations.pt\
	--seed 0 \
	--min_rot 0 \
	--max_rot 180 \
	--n_tasks 20

$MY_PYTHON cifar100.py \
	--o cifar100.pt \
	--seed 0 \
	--n_tasks 20

cd ..

# model "single"
$MY_PYTHON main.py $MNIST_ROTA --model single --lr 0.003 --seed 0&
$MY_PYTHON main.py $MNIST_PERM --model single --lr 0.03 --seed 0&
$MY_PYTHON main.py $CIFAR_100i --model single --lr 1.0 --seed 0&
$MY_PYTHON main.py $TinyImageNet --model single --lr 0.01 --seed 0&

# model "independent"
$MY_PYTHON main.py $MNIST_ROTA --model independent --lr 0.1  --finetune yes --seed 0&
$MY_PYTHON main.py $MNIST_PERM --model independent --lr 0.03 --finetune yes --seed 0&
$MY_PYTHON main.py $CIFAR_100i --model independent --lr 0.3  --finetune yes --seed 0&

# model "multimodal"
$MY_PYTHON main.py $MNIST_ROTA  --model multimodal --lr 0.1 &
$MY_PYTHON main.py $MNIST_PERM  --model multimodal --lr 0.1 &

# model "EWC"
$MY_PYTHON main.py $MNIST_ROTA --model ewc --lr 0.01 --n_memories 1000 --memory_strength 1000 --seed 0&
$MY_PYTHON main.py $MNIST_PERM --model ewc --lr 0.1  --n_memories 10   --memory_strength 3 --seed 0&
$MY_PYTHON main.py $CIFAR_100i --model ewc --lr 1.0  --n_memories 10   --memory_strength 1 --seed 0&

# model "LWF"
$MY_PYTHON main.py $CIFAR_100i --model lwf --lr 1.0 --memory_strength 1 --seed 0&

# model "iCARL"
$MY_PYTHON main.py $CIFAR_100i --model icarl --lr 1.0 --n_memories 1280 --memory_strength 1 --seed 0&

# model "GEM"
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $TinyImageNet --model gem --lr 0.01 --n_memories 256 --memory_strength 0.5 --seed 0&

wait

# plot results
cd results/
$MY_PYTHON plot_results.py
cd ..

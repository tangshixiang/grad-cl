# grad-cl

code for Layerwise Gradient Decomposition for Continual Learning

# Usage
Acknowledgement: We borrow codes from https://github.com/facebookresearch/GradientEpisodicMemory. Please install dependencies from https://github.com/facebookresearch/GradientEpisodicMemory. Logs are saved at results/

Register enviromental variables.
###
source run_config.sh
###

run our codes

single model
```
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0
$MY_PYTHON main.py $MNIST_PERM --model single --lr 0.03 --seed 0 
$MY_PYTHON main.py $CIFAR_100i --model single --lr 1.0 --seed 0
$MY_PYTHON main.py $TinyImageNet --model single --lr 0.01 --seed 0
```

independent
```
$MY_PYTHON main.py $MNIST_ROTA --model independent --lr 0.1  --finetune yes --seed 0
$MY_PYTHON main.py $MNIST_PERM --model independent --lr 0.03 --finetune yes --seed 0
$MY_PYTHON main.py $CIFAR_100i --model independent --lr 0.3  --finetune yes --seed 0
```

model "multimodal"
```
$MY_PYTHON main.py $MNIST_ROTA  --model multimodal --lr 0.1
$MY_PYTHON main.py $MNIST_PERM  --model multimodal --lr 0.1
```

model "ewc"
```
$MY_PYTHON main.py $MNIST_ROTA --model ewc --lr 0.01 --n_memories 1000 --memory_strength 1000 --seed 0
$MY_PYTHON main.py $MNIST_PERM --model ewc --lr 0.1  --n_memories 10   --memory_strength 3 --seed 0
$MY_PYTHON main.py $CIFAR_100i --model ewc --lr 1.0  --n_memories 10   --memory_strength 1 --seed 0
```

model "LWF"
```
$MY_PYTHON main.py $CIFAR_100i --model lwf --lr 1.0 --memory_strength 1 --seed 0
```

model "iCARL"
```
$MY_PYTHON main.py $CIFAR_100i --model icarl --lr 1.0 --n_memories 1280 --memory_strength 1 --seed 0
```

model "GEM"
```
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0&
$MY_PYTHON main.py $TinyImageNet --model gem --lr 0.01 --n_memories 256 --memory_strength 0.5 --seed 0
```

model "SGEM"
```
$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0
```

model "Our model"
```
$MY_PYTHON main.py $CIFAR_100i --model newblockgem_group5_pca3_partmargin.py --lr 0.1 --n_memories 256 --memory_strength 0.5 --seed 0
```

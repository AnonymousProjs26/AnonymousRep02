# Stratified Random Sampling on Neural Network Test Input Selection
This is the homepage of **SSOA** including `tool implementation` and `DNN model training scripts`. 

#### Environment configuration
Before running SSOA, please make sure you have installed various related packages, including keras, tensorflow, numpy and sklearn.

#### Running
Please use the following command to execute SSOA:

```shell
python selection_optimum_train_kmeans.py --exp_id=cn12

python selection_optimum_adaptive_kmeans.py --exp_id=cn12
```

- `exp_id` : the id of the model, including 'cn12', 'cifar10_vgg16', and 'cifar100_vgg16' 
 

#### Datasets and pre-trained models

We published all DNN model training scripts we utilized and you can find them in `model`.

The data of CIFAR-10 and CIFAR-100 can be obtained directly from Keras API.

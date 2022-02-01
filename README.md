# Stratified Random Sampling on Neural Network Test Input Selection
This is the homepage of **SSOA** including `tool implementation`, `evaluation scripts`, `studied DNN models` , `corresponding testing sets` and `experiment results`. 

#### Environment configuration
Before running SSOA, please make sure you have installed various related packages, including keras, tensorflow, and sklearn.

#### Running
Please use the following command to execute SSOA:

```shell
python selection_optimum_train_kmeans.py --exp_id=cn12
```

- `exp_id` : the id of the model, including 'cn12', 'cifar10_vgg16', and 'cifar100_vgg16' 

#### Results
Also, we put the raw data results for all experiments in `AllResult`. 

#### Datasets and pre-trained models

We published all studied DNN models we utilized and you can find them in `model`.

The data of CIFAR-10 and CIFAR-100 can be obtained directly from Keras API.

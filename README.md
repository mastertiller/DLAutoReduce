# Project overview

We proposed a grid search algorithm. **We will remove the neuron if the training loss or training accuracy does not change too much after deleting it; we will remove them recursively.**

# How to use

1. First, users should define their own neural network with Pytorch (We use AlexNet as the example):

2. Users also need to define the 'train_get_loss_acc' function; this function will train their model and return the training loss or accuracy. Users can define the hyper-parameters by themselves, such as epoch, batch size, etc.

3. Then, users can import and call our open-source tool with the parameters as shown below. After calling the functions, the reducing process will begin. 

# Work flow

The workflow of our open-source framework is as follows: 
1. First, it parses the model and extracts the linear layer of the input model. 
2. Then, it calls the 'train_get_loss' function to get the base loss and base accuracy. The next step will reduce each linear layer's neurons, recalculate the loss and accuracy, and store it in a list. 
3. In the list, we find the best loss or accuracy (lowest loss or highest accuracy).
4. If the best loss and accuracy didn't change within a threshold, we will replace the model with the best loss and accuracy as the base model. Then, go back to step 2. 
5. If the best loss and accuracy change out of the threshold, the reduction will stop and output the current base model.

# Evaluation

## Integrated with TF-Meter

TF-Meter is a measurement and visualization tool for deep neural networks, primarily written in TypeScript. 

We updated and added a button on the front end of TF-Meter ([Updated TF-Meter Github Repo](https://github.com/mastertiller/tf-meter)). Then, we implemented our algorithm in TypeScript and integrated it with TF-Meter. The following figure shows the front end of the updated version of TF-Meter. After clicking the search & reduce button we added, the proposed algorithm will start to reduce the neural network.

![](images/tf-meter.png)

## Experiment result

Todo
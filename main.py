from __future__ import print_function
import os
import sys
import logging
import argparse
import time
from time import strftime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import yaml

from vgg_cifar import vgg13

# settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='training batch size (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load-model-path', type=str, default="./cifar10_vgg13.pt",
                    help='Path to pretrained model')
parser.add_argument('--sparsity-type', type=str, default='unstructured',
                    help="define sparsity_type: [unstructured, filter, etc.]")
parser.add_argument('--sparsity-method', type=str, default='omp',
                    help="define sparsity_method: [omp, imp, etc.]")
parser.add_argument('--yaml-path', type=str, default="./vgg13.yaml",
                    help='Path to yaml file')

args = parser.parse_args()

# --- for dubeg use ---------
# args_list = [
#     "--epochs", "160",
#     "--seed", "123",
#     # ... add other arguments and their values ...
# ]
# args = parser.parse_args(args_list)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def get_dataloaders(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, download=True,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=256, shuffle=False)

    return train_loader, test_loader


# ============= the functions that you need to complete start from here =============

def read_prune_ratios_from_yaml(file_name, model):

        """
            This function will read user-defined layer-wise target pruning ratios from yaml file.
            The ratios are stored in "prune_ratio_dict" dictionary, 
            where the key is the layer name and value is the corresponding pruning ratio.

            Your task:
                Write a snippet of code to check if the layer names you provided in yaml file match the real layer name in the model.
                This can make sure your yaml file is correctly written.
        """

        if not isinstance(file_name, str):
            raise Exception("filename must be a str")
        with open(file_name, "r") as stream:
            try:
                raw_dict = yaml.safe_load(stream)
                prune_ratio_dict = raw_dict['prune_ratios']

                # ===== your code starts from here ======
                
                # ===== your code ends here ======

                return prune_ratio_dict

            except yaml.YAMLError as exc:
                print(exc)


def unstructured_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    Implement magnitude-based unstructured pruning for weight tensor (of a layer)
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
  
    :return:
        torch.(cuda.)Tensor, pruning mask (1 for nonzeros, 0 for zeros)
    """

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: Calculate how many weights should be pruned

    # Step 2: Find the threshold of weight magnitude (th) based on sparsity.

    # Step 3: Get the pruning mask tensor based on the th. The mask tensor should have same shape as the weight tensor
    #         |weight| <= th -> mask=0,
    #         |weight| >  th -> mask=1

    # Step 4: Apply mask tensor to the weight tensor
    #         weight_pruned = weight * mask

    ##################### YOUR CODE ENDS HERE #######################

    # return the mask to record the pruning location ()

    # # Step 1: Calculate how many weights should be pruned
    # num_elements = tensor.numel()
    # num_pruned = num_elements * sparsity
    # print(num_pruned)
    # print(num_elements)
    # # Step 2: Find the threshold of weight magnitude (th) based on sparsity
    # # Flatten tensor, get absolute values, sort them, and select the pruning threshold
    # flattened_tensor = tensor.view(-1).abs()
    # threshold = torch.topk(flattened_tensor, num_elements - num_pruned, largest=False).values.max()

    # # Step 3: Get the pruning mask tensor based on the threshold
    # # Values below or equal to the threshold are pruned (set to 0 in the mask)
    # mask = (tensor.abs() > threshold).float()



    num_elements = tensor.numel()
    num_pruned = int(num_elements * sparsity)
    num_pruned = min(num_pruned, num_elements - 1)
    if num_pruned == 0:
        return torch.ones_like(tensor)
    flat_tensor = tensor.view(-1) 
    threshold = torch.kthvalue(torch.abs(flat_tensor), num_pruned)[0]  

    mask = (torch.abs(tensor) > threshold).float()

    return mask


def filter_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    implement L2-norm-based filter pruning for weight tensor (of a layer)
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
  
    :return:
        torch.(cuda.)Tensor, pruning mask (1 for nonzeros, 0 for zeros)
    """

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: Calculate how many filters should be pruned

    # Step 2: Find the threshold of filter's L2-norm (th) based on sparsity.

    # Step 3: Get the pruning mask tensor based on the th. The mask tensor should have same shape as the weight tensor
    #         ||filter||2 <= th -> mask=0,
    #         ||filter||2 >  th -> mask=1

    # Step 4: Apply mask tensor to the weight tensor
    #         weight_pruned = weight * mask

    ##################### YOUR CODE ENDS HERE #######################

    # return the mask to record the pruning location ()
    return mask



def apply_pruning(model, sparity_type, prune_ratio_dict):
    # calculate layer_wise prune ratio for current round (if IMP)
    
    # call unstructured_prune()  
    # or 
    # call filter_prune (...)
    # print(dict(model.named_parameters()))
    print()
    print("=========================layer names====================================")
    for name, param in model.named_parameters():
        print(f"{name}")
        # print(f"Shape: {param.shape}")
    print("=========================layer names====================================")
    print()    
    # print(flattened_weights)
    prune_masks_store = {}  
    if sparity_type =='unstructured':
        for layer_name, tensor in prune_ratio_dict.items():
                # layer = dict(model.named_parameters())[layer_name]
                # print(layer_name)
            pruning_mask = unstructured_prune(dict(model.named_parameters())[layer_name],tensor)
            prune_masks_store[layer_name] = pruning_mask
            # print(pruning_mask)
            with torch.no_grad(): 
                layer = dict(model.named_parameters())[layer_name] 
                layer.data *= pruning_mask  
        return model,prune_masks_store
    elif sparity_type =='filter':
        pass        
    else:
        raise ValueError("Invalid sparsity type. Only 'unstructured' and 'filter' are supported.")    

    


def test_sparity(model, sparisty_type):
    pass
    # This function is used to check the model sparsity.
    # It should be able to print the sparisty ratio of each layer.

    # This example is obtained by testing a dense vgg13 model, 
    # this is why the sparity and number of zeros are all 0.
    # When you successfully pruned the model, then it should show the target sparisty ratio.
    # In other words, if the sparity of your pruned model is 0%, this indicates there must be something wrong.

    # features.x.weight is the layer name. 
    # You can the layer name and its weights by using the following for loop.
    # for name, weight in model.named_parameters():

    # For sparisty_type="unstructured":

    # Sparsity type is: xxxx (e.g., unstructured pruned or filter pruned)
    # (zero/total) weights of features.0.weight is: (0/1728). Sparsity is: 0.00%
    # (zero/total) weights of features.3.weight is: (0/36864). Sparsity is: 0.00%
    #       ...
    #       ...
    # ---------------------------------------------------------------------------
    # total number of zeros: 0, non-zeros: 9402048, overall sparsity is: 0.0000

    # For sparisty_type="filter":

    # (empty/total) filter of features.0.weight is: (0/64). filter sparsity is: 0.00%
    # (empty/total) filter of features.3.weight is: (0/64). filter sparsity is: 0.00%
    #       ...
    #       ...
    # ---------------------------------------------------------------------------
    # total number of filters: 2944, empty-filters: 0, overall filter sparsity is: 0.0000


# def masked_retrain(model, prune_masks, optimizer, loss_fn, data_loader, num_epochs=1):
#     pass
#     # when you fine-tune your pruned model, you only want to update the remaining weights (i.e., the weights that are not pruned),
#     # while keeping the pruned weights to be 0.
#     # A simple way to achieve this is:
#     #   1. before update the weights, you find the pruning mask first.
#     #   2. update all weights (including both remained and pruned weights).
#     #   3. based on the pruning mask, prune the weights again.
#     #      In this way, you can "keep" the pruned weights to be 0 after a training iteration.
#     # then manually pruned the weights again ()

#     # Example:
#     # For each training iteration
#     #       ...
#     #       optimizer.zero_grad()
#     #       loss.backward()
#     #       optimizer.step()
#     #       # Here you may need a loop to loop over entire model layer by layer, then
#     #       weight = weight * mask 


def masked_retrain(model, prune_masks, optimizer, loss_fn, data_loader, num_epochs=1):
    """
    Fine-tune the pruned model, updating only the unpruned weights, and print the accuracy for each iteration.
    
    :param model: nn.Module, the pruned model to fine-tune
    :param prune_masks: dict, dictionary containing the pruning mask for each layer
    :param optimizer: optimizer, optimizer for training the model
    :param loss_fn: loss function, criterion to calculate the loss
    :param data_loader: data loader, provides batches of input data and targets
    :param num_epochs: int, number of epochs to retrain the model
    """
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Compute accuracy for the current batch
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0) * 100  # percentage
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Reapply the pruning mask to keep pruned weights at zero
            with torch.no_grad():
                for layer_name, mask in prune_masks.items():
                    # Retrieve the weight tensor for each layer using the layer name
                    layer = dict(model.named_parameters())[layer_name]
                    # Reapply mask to maintain pruned weights at zero
                    layer.data *= mask
            
            # Print loss and accuracy for each iteration
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")



def oneshot_magnitude_prune(model, sparity_type, prune_ratio_dict,train_loader,optimizer,loss_fn):

    model,prune_masks=apply_pruning(model, sparity_type, prune_ratio_dict)
    # masked_retrain(model, prune_masks, optimizer, loss_fn, train_loader, num_epochs=5)
    
    # masked_retrain()
    
    # Implement the function that conducting oneshot magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # This function should also include the masked_retrain() function to conduct fine-tuning to restore the accuracy

def iterative_magnitude_prune():
    pass
    # Implement the function that conducting iterative magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # You can choose the way to gradually increase the pruning ratio.
    # For example, if the overall target sparsity is 80%, 
    # you can achieve it by 20%->40%->60%->80% or 50%->60%->70%->80% or something else e.g., in LTH paper.
    # At each sparsity level, you need to retrain your model. 
    # Therefore, this IMP method requires more overall training epochs than OMP.
    # ** IMP method needs to use at least 3 iterations.

def prune_channels_after_filter_prune():
    pass
    # 
    # You need to implement this function to complete the following task:
    # 1. This function takes a filter pruned and fine-tuned model as input
    # 2. Find out the indices of all pruned filters in each CONV layer
    # 3. Directly prune the corresponding channels (that has the same indices) in next CONV layer (on top of the filter-pruned model).
    #    There is no need to fine-tune this model again.
    # 4. Return the newly pruned model

    # E.g., if you prune the filter_1, filter_4, filter_7 from the i_th CONV layer,
    # Then, this function will let you prune the Channel_1, Channel_4, Channel_7, from the next CONV layer, i.e., (i+1)_th CONV layer.

    # How to use this function:
    # 1. You will apply this function on a filter-pruned model (after fine-tune/mask retraine)
    # 2. There is no need to fine-tune the model again after apply this function
    # 3. Compare the test accuracy before and after apply this function
    #   
    # E.g., 
    #       pruned_model = your pruned and fine/tuned model
    #       test_accuracy(pruned_model)
    #       new_model = prune_channels_after_filter_prune(pruned_model)
    #       test_accuracy(new_model)

    # Answer the following questions in your report:
    # 1. After apply this function (further prune the corresponding channels), what is the change in sparsity?
    # 2. Will accuray decrease, increase, or not change?
    # 3. Based on question 2, explain why?
    # 4. Can we apply this function to ResNet and get the same conclusion? Why?


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set up model archetecture and load pretrained dense model

    model = vgg13()
    model.load_state_dict(torch.load(args.load_model_path))
    if use_cuda:
        model.cuda()

    train_loader, test_loader = get_dataloaders(args)

    # Select loss function. You may change to whatever loss function you want.
    criterion = nn.CrossEntropyLoss()

    # Select optimizer. You may change to whatever optimizer you want.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # you may use this lr scheduler to fine-tune/mask-retrain your pruned model.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)

    # ========= your code starts here ========
    # ...
    # pruning_process()
    # masked_retrain()
    # ...

    # ---- you can test your model accuracy and sparity using the following fuction ---------
    # test_sparity()
    # test(model, device, test_loader)

    # ========================================
    prune_ratio_dict=read_prune_ratios_from_yaml(args.yaml_path,args.load_model_path)
    print()
    print("=========================================loaded dictonary===========================================================")
    print(args.sparsity_type,prune_ratio_dict)
    print("=========================================loaded dictonary===========================================================")
    print()
    oneshot_magnitude_prune(model, args.sparsity_type, prune_ratio_dict,train_loader,optimizer,criterion)



if __name__ == '__main__':
    main()

#python main.py --sparsity-method omp --sparsity-type unstructured --epochs 10
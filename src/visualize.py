import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
#import cv2 as cv
import argparse

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ##########################################################################################
    #   Il faut transposée les images car PyTorch lis les image en [Chanels, Width, Height]  #
    #   et pour les voir il faut qu'elles soient [Width, Height, Chanels]                    #
    ##########################################################################################
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

def accuracy_loss_plots(train_epoch_loss, train_epoch_acc):
    plt.subplot(1,2,1)
    plt.plot(range(len(train_epoch_loss)), train_epoch_loss)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss curve")

    plt.subplot(1,2,2)
    plt.plot(range(len(train_epoch_acc)), train_epoch_acc)

    plt.xlabel("Accuracy")
    plt.ylabel("Loss")
    plt.title("accuracy curve")

    plt.savefig("outputs/training_results/loss_accuracy_curve")
    plt.show()

def visualize_conv_layers(model):
    print(model)
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0 

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # visualize the first conv layer filters
    plt.figure(figsize=(10, 8))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(6, 6, i+1) # (6, 6) because in conv0 we have 5x5 filters and total of 36 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.savefig('outputs/feature_maps/filter.png')
    plt.show()
    return conv_layers

def visualize_feature_maps(img, conv_layers):
    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results
    # visualize 36 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer].cpu()
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 36: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(6, 6, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"outputs/feature_maps/layer_{num_layer}.png")
        # plt.show()
        plt.close()
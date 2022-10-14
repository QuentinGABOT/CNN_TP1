from train_utils import train, validate, test, test_class_by_class
from model import CustomNet
from datasets import get_data_loaders
from visualize import imshow, accuracy_loss_plots, visualize_conv_layers, visualize_feature_maps
from search_and_train import run_search
from config import (
    MAX_NUM_EPOCHS, GRACE_PERIOD, EPOCHS, CPU, GPU,
    NUM_SAMPLES, DATA_ROOT_DIR, NUM_WORKERS, IMAGE_SIZE, VALID_SPLIT
)

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np

if __name__ == '__main__':
    # hyperparameters tuning
    while True:
        word = input("Voulez-vous lancer l'algorithme de Random Search ? (y/n)").lower()
        if word == "y":
            optimal_config = run_search()
        if word == "n": 
            break

    # set those values !
    batch_size = 16
    first_conv_out = 10
    first_fc_out = 120
    lr = 0.01

    model = CustomNet(first_conv_out = first_conv_out, first_fc_out = first_fc_out)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9)

    train_loader, valid_loader, test_loader, dataset_classes = get_data_loaders(
        IMAGE_SIZE, DATA_ROOT_DIR, VALID_SPLIT,
        batch_size, NUM_WORKERS, training=True
    )

    
    summary(model, (batch_size, 3, 32, 32)) # dimensions d'une image !

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # show images
    imshow(torchvision.utils.make_grid(images.cpu()))

    list_loss = []
    list_acc = []
    # Start the training.
    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
  
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print('-'*50)
        list_loss.append(train_epoch_loss)
        list_acc.append(train_epoch_acc)

    torch.save({
                'nb_epoch': EPOCHS,
                'model' : model.state_dict(),
                'listLoss': train_epoch_loss,
                'listAcc' : train_epoch_acc
            }, "outputs/training_results/modelNN.pth")

    print('Finished Training and save the model as `modelNN.pth`')

    accuracy_loss_plots(train_epoch_loss = list_loss, train_epoch_acc = list_acc)
    
    conv_layers = visualize_conv_layers(model = model)
    visualize_feature_maps(images[0], conv_layers) # choisir une image

    
    # Charger un batch de l'ensemble de test
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # Passer le batch dans le reseau
    outputs = model(images)
    predicted1 = torch.softmax(outputs.data, 1) #decision probabiliste (floue)20

    _, predicted2 = torch.max(predicted1, 1) #decision dure (classification)

    # print images
    imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth: ', ' '.join('%5s' % dataset_classes[labels[j]] for j in range(16)))
    print('Predicted: ', ' '.join('%5s' % dataset_classes[predicted2[j]] for j in range(16)))

    print("The accuray on the test set is : " + str(test(model = model, data_loader=test_loader, device=device)*100) + "%")
    test_class_by_class(model = model, data_loader=test_loader, classes=dataset_classes, device=device)
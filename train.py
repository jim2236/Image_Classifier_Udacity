# Image Classifier Project
# Train a network and display accuracy results
# Jim Michael 7/4/20

# Import software modules 
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

from collections import OrderedDict

import matplotlib.pyplot as plt


import PIL
from PIL import Image

import numpy as np

import argparse

import os

# Define parser arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', action='store', type=str, default='flowers', help="Path to image")
parser.add_argument('--arch', action='store', type=str, default='vgg16', help='Choose model architecture')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help="Sets the learning rate")
parser.add_argument('--hidden_units', action='store', type=int, nargs=2, default=[512, 256], help="Sets number of hidden units")
parser.add_argument('--epochs', action='store', type=int, default=2, help="Sets number of epochs")
parser.add_argument('--gpu', action='store_true', default='cpu', help="Turn on GPU")

args = parser.parse_args()

# Define data directories

if args.data_dir != 'flowers':
    os.system('mkdir '+ args.data_dir)
    print(args.data_dir)
    os.chdir(str(args.data_dir))    # <<<<<<<<< problem here 
    #zz = os.getcwd()
    #print(zz)
    os.chdir('/home/workspace/ImageClassifier')
    zzz = os.getcwd()
    print(zzz)
    os.system('mkdir train')
    os.system('mkdir valid')
    os.system('mkdir test')
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    print('$$$$$$.    ', train_dir, '  ', valid_dir, '   ', test_dir)
else:
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


# Create data transforms 
print("Starting Transforms")
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Load the datasets with ImageFolder
print('-------------   ', train_dir)
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

image_datasets = train_data

validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

# Move to GPU is available

if args.gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = getattr(models, args.arch)(pretrained=True)

#model = models.vgg16(pretrained=True)


# Freeze parametes to avoid backprop
for param in model.parameters():
    param.requires_grad=False


# Define the model   
model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hu1)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(hu1, hu2)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.2)),
                            ('fc3', nn.Linear(hu2, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

# Define the loss function
criterion = nn.NLLLoss()

# Define the optimizer
learning_rate = args.learning_rate    #  0.001
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# Send the model to the GPU or CPU

model.to(device)


# Train the network
print("Train the network")  # remove

epochs = args.epochs 

steps = 0
steps_2 = 0   # remove
steps_3 = 0.  # remove
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1 
        print("Step ", steps)  # remove       
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)  # what about flower_to_name?
        loss.backward()
        optimizer.step()
        
        # Validation
        running_loss += loss.item()
        
        if steps % print_every == 0:
            steps_2 += 1  # remove
            print(steps_2)  # remove 
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    steps_3 += 1   # remove 
                    print(steps_3)
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                
                    validation_loss += batch_loss.item()
                
                    # Calculate accuracy 
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                     
            print(f"Epoch {epoch+1} / {epochs}.."
                  f"Train Loss:  {running_loss / print_every: .3f}.."
                  f"Validation Loss: {validation_loss/len(validation_loader): .3f}.. "
                  f"Validation Accuracy: {accuracy / len(validation_loader): .3f}")
            running_loss = 0
            model.train()
          
            
# Run validation on the test set
print("Run validation on test set")
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
    
        test_loss += batch_loss.item()
    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Test Accuracy: {accuracy/len(test_loader): .3f}") 
    
model.train()    


# Save the checkpoint 

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input size': 25088,
              'hidden layers': [512, 256],
              'output size': 102,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}  

torch.save(checkpoint,'checkpoint.pth')

print("Done")  # remove 
# End
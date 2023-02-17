import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
# for command line interface
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Network Training')

    parser.add_argument('data_dir', type = str, default='flowers', help='Data Directory' )

    parser.add_argument('--arch', dest='arch', type = str, default='vgg13', help='Model Architecture' )
    parser.add_argument('--learn_rate', dest='learn_rate', type = float, default=0.001, help='Learning rate utilized by the model' )
    parser.add_argument('--epochs', dest='epochs', type = int, default=5, help='Number of epochs')
    parser.add_argument('--hidden_units', dest='hidden_units', type = int, default=4096, help='Number of hidden units')
    parser.add_argument('--gpu', type = str, default='False', help='Write True to use gpu for training' )

    

    #11ata_dir = arg.data_dir
    #11arch = arg.arch
    #11learn_rate = arg.learn_rate
    #11epoch_arg = arg.epochs
    #11hidden_units = arg.hidden_units
    #11gpu = arg.gpu
    
    return parser.parse_args()

def main():
    
    arg = get_args()
    
    if arg.gpu == 'True':
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    data_dir = arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])



    test_transform =  transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    
    layer_num = 0
    
    if arg.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        layer_num = 1024
    elif arg.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        layer_num = 25088
    else:
        print('we only support densenet121 and vgg13')
        
    for param in model.parameters():
        param.requires_grad = False
    
    h_unit = arg.hidden_units
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(layer_num, h_unit)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(h_unit, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    lr_arg = arg.learn_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr_arg)
    model.to(device);
    epochs = arg.epochs
    steps = 0
    print_every = 10
    print('Starting Training')
    for epoch in range(epochs):
        running_loss = 0
        for inputs,labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                    
                        # calculate losses
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
        
            ps = torch.exp(logps)       
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    
    
    model.class_to_idx = train_data.class_to_idx

    
    checkpoint = {'epochs': epochs,
                  'arch': arg.arch,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint_new.pth')

    
if __name__ == "__main__":
    main()
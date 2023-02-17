import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Prediction')

    parser.add_argument('image_path', type = str, default='flowers/test/37/image_03734.jpg', help='Image path')
    parser.add_argument('--checkpoint', type = str, default='checkpoint_new.pth', help='Model Checkpoint' )
    
    parser.add_argument('--json', type = str, default='cat_to_name.json', help='load a JSON file that maps the class values to other category names' )
    parser.add_argument('--gpu', type = str, default='False', help='Write True to use gpu for predicting' )
    parser.add_argument('--topk', type = int, default='5', help='Input TopK' )

    
    return parser.parse_args()

def process_image(image):
   
    
    pil_image = Image.open(image)
    
    
    
    transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    image_tr = transform(pil_image)
    
    
    
    return image_tr


def load_check(filepath, device): 
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    
    model.load_state_dict = checkpoint['state_dict']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    
    return model


def predict(img_path, filepath, model, device, topk=5):
    
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.to(device)
    model.eval()
    class_dict = model.class_to_idx
    class_idx_inverted = dict(zip(class_dict.values(), class_dict.keys()))
    
    
    img_trch = process_image(img_path)
    
    #img_trch = torch.from_numpy(img_trch).type(torch.FloatTensor)
    
    
    with torch.no_grad():
        img_trch = img_trch.to(device)
        img_trch = img_trch.unsqueeze_(0)
        img_trch.float()
        logps = model.forward(img_trch)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(topk, dim=1)
        
    classes_predictions = [class_idx_inverted[key] for key in top_class.cpu().numpy()[0]]
    prob = top_ps.cpu().numpy()[0]
    return prob, classes_predictions

def main():
    arg = get_args()
    
    if arg.gpu == 'True':
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')
        
    with open(arg.json , 'r') as f:
        cat_to_name = json.load(f)
    
    model_load = load_check(arg.checkpoint, device)
    
    image_path = arg.image_path
    prob, classes_prediction = predict(image_path, arg.checkpoint, model_load, device, arg.topk)
    
    classes_names = [cat_to_name[j] for j in classes_prediction]
    print(prob, classes_names)
    print(f'{prob[0]*100:.2f}% of Probability of Being a', classes_names[0])

    
if __name__ == "__main__":
    main() 
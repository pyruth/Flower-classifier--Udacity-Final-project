import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
import argparse
from PIL import Image
import numpy as np
import json

#user inputs for the prediction
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', action='store_true', help='gpu is disabled by default')
parser.add_argument('image_path', type=str, help='to pass the image path')
parser.add_argument('checkpoint', type=str, help='pass the model save file .pth')
parser.add_argument('--topk', type = int, default=5)
parser.add_argument('--category_names', type=str)
args = parser.parse_args()

#reading flowers.json and storing in a list
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#checking the device is in cpu or gpu
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

# TODO: Write a function that loads a checkpoint and rebuilds the model
def checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = getattr(models, checkpoint['model'])(pretrained=True)
    epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.features = checkpoint['features']   
    model.load_state_dict(checkpoint['state_dict'])    
    
    optimizer = checkpoint['optimizer']
    learning_rate = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    return model 


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(244)])
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    pil_image = Image.open(image)

    pil_image = transform(pil_image)
  
    np_image = np.array(pil_image) / 255
    
    pil_image = (np_image - mean) / std
    
    pil_image = pil_image.transpose((2, 0, 1))
    
    
    return  torch.tensor(pil_image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = process_image(image_path).unsqueeze_(0).float()
    model = checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    with torch.no_grad():
        
        image = image.to(device)
        log_ps = model(image)
    
    ps = torch.exp(log_ps)
    top_p, top_indices = ps.topk(topk, dim=1)
        
    idx_to_class = {value:key for key,value in model.class_to_idx.items()}
    
    top_classes = []
   
    for i in np.array(top_indices)[0]: 
        top_classes.append(idx_to_class[i])
        

    return top_p.cpu().numpy()[0], top_classes



    
image_path = args.image_path
filepath = args.checkpoint


top_k, top_class = predict(image_path, filepath, args.topk)

flower = [cat_to_name[i] for i in top_class]

#print(top_k, flower)
print("flower name=======percentage")
for (name, percentage) in zip(flower, top_k):
          print(f" {name}======={percentage * 100:.2f}%")


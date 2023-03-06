import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, help='where the flower images are stored')
parser.add_argument('--gpu', action='store_true', help='gpu is disabled by default')
parser.add_argument('--save_dir', type = str, default='checkpoint.pth', help='default save dir path is checkpoint.pth')
parser.add_argument('--arch', type = str, default='vgg', help="Models supports: VGG, densenet", choices=('vgg' 'densenet'))
parser.add_argument('--learning_rate', type = float, default=0.0025, help='default learning rate is 0.0025')
parser.add_argument('--hidden_units1', type = int, default=4096, help = 'default hidden layer1 is 4096')
parser.add_argument('--hidden_units2', type = int, default=256,help = 'default hidden layer1 is 4096')
parser.add_argument('--epochs', type = int, default=5, help= 'default epochs is 5')

args = parser.parse_args()
# Assign the data directories in variables
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size =64, shuffle = True)

#creating a model
if args.arch == "vgg":    
    model = getattr(models, args.arch)(pretrained =True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units1),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(args.hidden_units1, args.hidden_units2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(args.hidden_units2, 102),
                        nn.LogSoftmax(dim=1))
elif args.arch == "densenet":
    model = getattr(models, args.arch)(pretrained =True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(1024, args.hidden_units1),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(args.hidden_units1, args.hidden_units2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(args.hidden_units2, 102),
                        nn.LogSoftmax(dim=1))

    
#checking the device is in cpu or gpu
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

#moving the model to device (cuda/cpu)
model.to(device)

#loss function
criterion = nn.NLLLoss()

#optimizer

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

#training the dataset
epochs = args.epochs 
step = 0
running_loss = 0
print_acc = 20

#looping the epoch
for epoch in range(epochs):
    print("training started")
    for images, labels in trainloaders:
        #print("training")
        step += 1
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % print_acc == 0:
            model.eval()
            acc = 0
            test_loss = 0
            with torch.no_grad():
                for images, labels in validloaders:
                    #print("validation")
            
                    images, labels = images.to(device), labels.to(device)
                
                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()
            
                    #calculate acc
                    ps = torch.exp(logps)
                    top_k, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc += torch.mean(equals.type(torch.FloatTensor)).item()
            print("Epoch {}/{}..".format(epoch+1, epochs),
                  "Average Train loss: {:.3f}..".format(running_loss/print_acc),
                  "Average Validation loss: {:.3f}..".format(test_loss/len(validloaders)),
                  "Average Validation accuracy: {:.3f}".format(acc/len(validloaders))) 
        
            running_loss = 0
            model.train()

            
            # TODO: Do validation on the test set
model.eval()
acc = 0
test_loss = 0
with torch.no_grad():        
    for images, labels in testloaders:
            
        images, labels = images.to(device), labels.to(device)
            
        logps = model(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()
            
        #calculate acc
        ps = torch.exp(logps)
        top_k, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        acc += torch.mean(equals.type(torch.FloatTensor)).item()
print( 
      f"Test accuracy: {acc/len(testloaders):.3f}"
      ) 
# TODO: Save the checkpoint 
checkpoint = {'input_size': 25088,
                'epochs' : args.epochs,
             'model':args.arch,
              'hidden_units1' : args.hidden_units1,
              'hidden_units2' : args.hidden_units2,
             'output': 102,
             'features': model.features,
             'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': train_datasets.class_to_idx,
             'learning_rate':0.0025}
torch.save(checkpoint, args.save_dir)
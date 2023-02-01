# slides https://docs.google.com/presentation/d/1A38mJUAfDo-4yYzy6UCBZrEo3aE50ceO/edit?usp=sharing&ouid=107585355306558125830&rtpof=true&sd=true
# ref : https://github.com/Yangyangii/DANN-pytorch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import pandas as pd

class MyDataset(Dataset):
    def __init__ (self,path):
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in data:
            i[0] = os.path.join('/'.join(path.split('/')[:-1]),'data',i[0])

        self.filenames = data
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename[0]).convert('L')
        img = self.transform(img)
        return img,filename[1]

    def __len__(self):
        return len(self.filenames)             

def loadData(path,batch_size,workers=0,drop=False):
    dataset = MyDataset(path=path)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers,drop_last=drop)
    return data_loader

class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=1, hidden_dims=512):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 50, 5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        #x = x.expand(x.shape[0],3,28,28)
        h = self.conv(x).view(-1,50*4*4) # (N, hidden_dims)
        return h

class Classifier(nn.Module):
    def __init__(self, input_size=50*4*4, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100,10),
            nn.LogSoftmax(dim=1),
        )
        
    def forward(self, h):
        c = self.layer(h)
        return c
class Discriminator(nn.Module):
    def __init__(self, input_size=50*4*4, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100,1),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y


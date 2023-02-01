import torch
import torch.nn as nn
from torchvision import transforms, utils, models
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from os import listdir
from PIL import Image

class MyDataset(Dataset):
    def __init__ (self,path,trans,data_aug):
        filenames = []
        for f in listdir(path):
            label = f.split('_')[0]
            filenames.append((path+'/'+f,int(label)))
            if(data_aug):
                filenames.append((path+'/'+f+'a',int(label)))
        
        self.filenames = filenames
        if(trans):
            self.transform=trans
        else:
            self.transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        filename, label = self.filenames[index]
        if filename[-1]=='a':
            filename = filename[:-1]
            img = Image.open(filename).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = Image.open(filename).convert('RGB')
        
        img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.filenames)

class TestDataset(Dataset):
    def __init__ (self,path,trans=False):
        filenames = []
        for f in listdir(path):
            filenames.append(path+'/'+f)
        
        self.filenames = filenames
        if(trans):
            self.transform=trans
        else:
            self.transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        filename = filename.split('/')[-1]
        return img,filename

    def __len__(self):
        return len(self.filenames)


def loadData(path,trans=False,data_aug=False,test=False):
    batch_size = 256
    if(test):
        dataset = TestDataset(path=path,trans=trans)
    else:
        dataset = MyDataset(path=path,trans=trans,data_aug=data_aug)

    
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    return data_loader


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, 50))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def pca(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out


def myResnet(finetune=False):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(2048,50)
    return model,weights.transforms()

def loadModel(pth):
    model = models.resnet50()
    model.fc = nn.Linear(2048,50)
    model.load_state_dict(torch.load(pth))
    model.eval().cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model

def myPredict(pth,input,output):
    trans = models.ResNet50_Weights.DEFAULT.transforms()
    testdata = loadData(input,trans=trans,test=True)
    model = loadModel(pth)
    with open(output, 'w') as f:
        f.writelines('filename, label\n')

    for i, (images,filename) in enumerate(testdata, 0):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)
        for i in range(len(filename)):
            with open(output, 'a') as f:
                f.writelines(filename[i]+','+str(int(predicted[i]))+'\n')


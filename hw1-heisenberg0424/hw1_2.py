import torch
import torch.nn as nn
from torchvision import transforms, utils, models
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from os import listdir
from PIL import Image
import numpy as np
#https://docs.google.com/presentation/d/1lXkZrUrV209kMSGn6Lg37rno0Kp_zbdyxOl0K8F9U_E/edit?usp=sharing

class MyDataset(Dataset):
    def __init__ (self,path,trans,data_aug):
        filenames = []
        for f in listdir(path):
            if(f.endswith('jpg')):
                filenames.append(path+'/'+f)
                if(data_aug):
                    filenames.append(path+'/'+f+'a')
                    filenames.append(path+'/'+f+'b')
                    filenames.append(path+'/'+f+'c')
        
        self.filenames = filenames
        if(trans):
            self.transform=trans
        else:
            self.transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
            ])
        

    def __getitem__(self, index):
        filename = self.filenames[index]
        if filename[-1]=='a':
            filename = filename[:-1]
            img = Image.open(filename).convert('RGB').rotate(90, expand=True)
            mask = Image.open(filename[:-7]+'mask.png').convert('RGB').rotate(90, expand=True)
        elif filename[-1]=='b':
            filename = filename[:-1]
            img = Image.open(filename).convert('RGB').rotate(180, expand=True)
            mask = Image.open(filename[:-7]+'mask.png').convert('RGB').rotate(180, expand=True)
        elif filename[-1]=='c':
            filename = filename[:-1]
            img = Image.open(filename).convert('RGB').rotate(270, expand=True)
            mask = Image.open(filename[:-7]+'mask.png').convert('RGB').rotate(270, expand=True)
        else:
            img = Image.open(filename).convert('RGB')
            mask = Image.open(filename[:-7]+'mask.png').convert('RGB')

        mask_img = np.array(mask).astype(np.uint8)
        mask_img = (mask_img >= 128).astype(int) 
        mask_img = 4 * mask_img[:, :, 0] + 2 * mask_img[:, :, 1] + mask_img[:, :, 2]
        mask = np.empty((512,512))
        mask[ mask_img == 3] = 0  # (Cyan: 011) Urban land 
        mask[ mask_img == 6] = 1  # (Yellow: 110) Agriculture land 
        mask[ mask_img == 5] = 2  # (Purple: 101) Rangeland 
        mask[ mask_img == 2] = 3  # (Green: 010) Forest land 
        mask[ mask_img == 1] = 4  # (Blue: 001) Water 
        mask[ mask_img == 7] = 5  # (White: 111) Barren land 
        mask[ mask_img == 0] = 6  # (Black: 000) Unknown 
        mask[ mask_img == 4] = 6  # (Red: 100) Unknown

        img = self.transform(img)
        return img,mask

    def __len__(self):
        return len(self.filenames)

class TestDataset(Dataset):
    def __init__ (self,path,trans=False):
        filenames = []
        for f in listdir(path):
            if f.endswith('jpg'):
                filenames.append(path+'/'+f)
        
        self.filenames = filenames
        if(trans):
            self.transform=trans
        else:
            self.transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        filename = filename.split('/')[-1].replace('jpg','png')
        return img,filename

    def __len__(self):
        return len(self.filenames)


def loadData(path,trans=False,data_aug=False,test=False):
    batch_size = 12
    if(test):
        dataset = TestDataset(path=path,trans=trans)
        batch_size = 1
    else:
        dataset = MyDataset(path=path,trans=trans,data_aug=data_aug)

    
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=6)
    
    return data_loader

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()
        weights = models.VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=weights)
        # for param in vgg16.features.parameters():
        #     param.requires_grad = False
        self.features = vgg16.features
        self.classifier = nn.Sequential(
        nn.Conv2d(512, 4096, 7),
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Conv2d(4096, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Conv2d(4096, 7, 1),
        nn.ConvTranspose2d(7, 7, 224, stride=32)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class FCN8(nn.Module):
    def __init__(self):
        super(FCN8,self).__init__()
        weights = models.VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=weights)
        # for param in vgg16.features.parameters():
        #     param.requires_grad = False
        self.features = vgg16.features
        self.features[0].padding = [100,100]
        self.fcn = nn.Sequential(
            nn.Conv2d(512,4096,7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.pool3 = nn.Conv2d(256,7,1)
        self.pool4 = nn.Conv2d(512,7,1)
        self.pool5 = nn.Conv2d(4096,7,1)
        self.upsample32 = nn.ConvTranspose2d(7,7,4,stride = 2)
        self.upsample16 = nn.ConvTranspose2d(7,7,4,stride = 2)
        self.upsample8 = nn.ConvTranspose2d(7,7,16,stride = 8)
        torch.nn.init.xavier_uniform_(self.pool3.weight)
        torch.nn.init.xavier_uniform_(self.pool4.weight)
        torch.nn.init.xavier_uniform_(self.pool5.weight)
        torch.nn.init.xavier_uniform_(self.upsample32.weight)
        torch.nn.init.xavier_uniform_(self.upsample16.weight)
        torch.nn.init.xavier_uniform_(self.upsample8.weight)
        self.features.apply(init_weights)

    def forward(self,x):
        x_shape = x.shape
        x = self.features[:17](x)
        pool3 = x

        x = self.features[17:24](x)
        pool4 = x

        x = self.features[24:](x)
        x = self.fcn(x)

        x = self.pool5(x)
        x = self.upsample32(x)
        fcn32 = x

        x = self.pool4(pool4)
        x = x[:, :, 5:5+fcn32.size()[2], 5:5+fcn32.size()[3]]
        x = fcn32 + x
        x = self.upsample16(x)
        fcn16 = x

        x = self.pool3(pool3)
        x = x[:, :, 9:9+fcn16.size()[2], 9:9+fcn16.size()[3]]
        x = fcn16 + x
        x = self.upsample8(x)
        x = x[:, :, 31: 31 + x_shape[2], 31: 31 + x_shape[3]]
        return x

class FCN16(nn.Module):
  def __init__(self):
    super(FCN16, self).__init__()
    weights = models.VGG16_Weights.DEFAULT
    vgg16 = models.vgg16(weights=weights)
    self.features = vgg16.features
    self.classifier = nn.Sequential(
      nn.Conv2d(512, 4096, 7),
      nn.ReLU(inplace=True),
      nn.Conv2d(4096, 4096, 1),
      nn.ReLU(inplace=True),
      nn.Conv2d(4096, 21, 1)
    )
    self.score_pool4 = nn.Conv2d(512, 21, 1)
    self.upscore2 = nn.ConvTranspose2d(21, 21, 14, stride=2, bias=False)
    self.upscore16 = nn.ConvTranspose2d(21, 21, 16, stride=16, bias=False)

  def forward(self, x):
    pool4 = self.features[:-7](x)
    pool5 = self.features[-7:](pool4)
    pool5_upscored = self.upscore2(self.classifier(pool5))
    pool4_scored = self.score_pool4(pool4)
    combined = pool4_scored + pool5_upscored
    res = self.upscore16(combined)
    return res

def myVggFcn():
    fcn = FCN32().cuda()
    return fcn

def myFcn8():
    fcn = FCN8().cuda()
    return fcn

def myFcn16():
    fcn = FCN16().cuda()
    return fcn

def outputMask(pred):
    pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
    label_colors = np.array([(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(255,255,255),(0,0,0)])
    r = np.zeros_like(pred).astype(np.uint8)
    g = np.zeros_like(pred).astype(np.uint8)
    b = np.zeros_like(pred).astype(np.uint8)
    
    for l in range(0, 6):
        idx = pred == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    mask = transforms.ToPILImage()(rgb)
    return mask

def loadModel(pth):
    model = FCN16()
    model.load_state_dict(torch.load(pth))
    model.cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def myPredict(pth,input,output):
    testdata = loadData(input,test=True)
    model = loadModel(pth)
    for images,filename in testdata:
        outputs = model(images.cuda())
        maskimg = outputMask(outputs.cpu())
        maskimg.save(output+filename[0])
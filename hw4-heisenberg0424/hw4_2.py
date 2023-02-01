#ref https://github.com/lucidrains/byol-pytorch

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import pandas as pd

str2label = {'Fork': 0, 'Radio': 1, 'Glasses': 2, 'Webcam': 3, 'Speaker': 4, 'Keyboard': 5, 'Sneakers': 6, 'Bucket': 7, 'Alarm_Clock': 8, 'Exit_Sign': 9, 'Calculator': 10, 'Folder': 11, 'Lamp_Shade': 12, 'Refrigerator': 13, 'Pen': 14, 'Soda': 15, 'TV': 16, 'Candles': 17, 'Chair': 18, 'Computer': 19, 'Kettle': 20, 'Monitor': 21, 'Marker': 22, 'Scissors': 23, 'Couch': 24, 'Trash_Can': 25, 'Ruler': 26, 'Telephone': 27, 'Hammer': 28, 'Helmet': 29, 'ToothBrush': 30, 'Fan': 31, 'Spoon': 32, 'Calendar': 33, 'Oven': 34, 'Eraser': 35, 'Postit_Notes': 36, 'Mop': 37, 'Table': 38, 'Laptop': 39, 'Pan': 40, 'Bike': 41, 'Clipboards': 42, 'Shelf': 43, 'Paper_Clip': 44, 'File_Cabinet': 45, 'Push_Pin': 46, 'Mug': 47, 'Bottle': 48, 'Knives': 49, 'Curtains': 50, 'Printer': 51, 'Drill': 52, 'Toys': 53, 'Mouse': 54, 'Flowers': 55, 'Desk_Lamp': 56, 'Pencil': 57, 'Sink': 58, 'Batteries': 59, 'Bed': 60, 'Screwdriver': 61, 'Backpack': 62, 'Flipflops': 63, 'Notebook': 64}
label2str = {0: 'Fork', 1: 'Radio', 2: 'Glasses', 3: 'Webcam', 4: 'Speaker', 5: 'Keyboard', 6: 'Sneakers', 7: 'Bucket', 8: 'Alarm_Clock', 9: 'Exit_Sign', 10: 'Calculator', 11: 'Folder', 12: 'Lamp_Shade', 13: 'Refrigerator', 14: 'Pen', 15: 'Soda', 16: 'TV', 17: 'Candles', 18: 'Chair', 19: 'Computer', 20: 'Kettle', 21: 'Monitor', 22: 'Marker', 23: 'Scissors', 24: 'Couch', 25: 'Trash_Can', 26: 'Ruler', 27: 'Telephone', 28: 'Hammer', 29: 'Helmet', 30: 'ToothBrush', 31: 'Fan', 32: 'Spoon', 33: 'Calendar', 34: 'Oven', 35: 'Eraser', 36: 'Postit_Notes', 37: 'Mop', 38: 'Table', 39: 'Laptop', 40: 'Pan', 41: 'Bike', 42: 'Clipboards', 43: 'Shelf', 44: 'Paper_Clip', 45: 'File_Cabinet', 46: 'Push_Pin', 47: 'Mug', 48: 'Bottle', 49: 'Knives', 50: 'Curtains', 51: 'Printer', 52: 'Drill', 53: 'Toys', 54: 'Mouse', 55: 'Flowers', 56: 'Desk_Lamp', 57: 'Pencil', 58: 'Sink', 59: 'Batteries', 60: 'Bed', 61: 'Screwdriver', 62: 'Backpack', 63: 'Flipflops', 64: 'Notebook'}

class MyDataset(Dataset):
    def __init__ (self,path,csvpath,finetune):
        data = pd.read_csv(csvpath)
        
        self.finetune = finetune
        self.label = data['label']
        self.path = path
        self.filenames = data['filename']
        self.transform=transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        filename = os.path.join(self.path,filename)
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        if not self.finetune:
            return img,self.filenames[index]
        return img, str2label[self.label[index]]

    def __len__(self):
        return len(self.filenames)

def loadData(path,batch_size,csvpath,num_workers=2,finetune=False):
    dataset = MyDataset(path=path,csvpath=csvpath,finetune=finetune)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=finetune,num_workers=num_workers)
    return data_loader

def loadModel(pth,fixedbackbone=False):
    model = models.resnet50()
    if pth != ' ':
        model.load_state_dict(torch.load(pth))
    if fixedbackbone:
       for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048,65)
    return model.cuda()
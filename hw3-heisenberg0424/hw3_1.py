# ref : https://colab.research.google.com/github/andsteing/CLIP/blob/zeroshot/notebooks/zeroshot_evaluation.ipynb#scrollTo=NbVBHtzviG8h

from torch.utils.data import Dataset
import clip
import torch
import os
from tqdm import tqdm
from PIL import Image
import json
import sys

class MyDataset(Dataset):
    def __init__ (self,path,trans,val=False):
        filenames = []
        labels = []
        for f in os.listdir(path):
            if f.endswith('png'):
                filenames.append(os.path.join(path,f))
            if val:
                labels.append(f.split('_')[0])

        self.transform = trans
        self.filenames = filenames
        self.labels = labels
        self.val = val

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        if self.val:
            return img,int(self.labels[index])

        return img,filename.split('/')[-1]

    def __len__(self):
        return len(self.filenames)

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def zero_shot(path,labelpath,output=None,val=False):
    model, preprocess = clip.load('ViT-L/14','cuda')
    model.cuda().eval()
    dataset = MyDataset(path,preprocess,val)
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)

    with open(labelpath) as json_file:
        data = json.load(json_file)
    classes = list(data.values())
    templates = ['A photo of a {}.']
    zeroshot_weights = zeroshot_classifier(classes,templates,model)
    if val:
        with torch.no_grad():
            top1 = 0.
            n = 0
            for images,target in tqdm(dataloader):

                images = images.cuda()
                target = target.cuda()

                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1,keepdim=True)

                logits = 100. * image_features @ zeroshot_weights
                acc = accuracy(logits,target)
                top1 += acc[0]

                n += images.size(0)
            top1 = (top1 / n) * 100 
            print('Acc: ',top1)
    else:
        with open(output, 'w') as f:
            f.writelines('filename,label\n')
        with torch.no_grad():
            for images,filename in tqdm(dataloader):

                images = images.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1,keepdim=True)

                logits = 100. * image_features @ zeroshot_weights
                pred = logits.topk(1, 1, True, True)[1].t()
                pred = pred.squeeze()

                for i in range(len(filename)):
                    with open(output, 'a') as f:
                        f.writelines(filename[i]+','+str(int(pred[i]))+'\n')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Enter 3 arg')
        sys.exit()

    zero_shot(sys.argv[1],sys.argv[2],sys.argv[3])

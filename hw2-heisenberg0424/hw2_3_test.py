from hw2_3 import *
import torch
import sys

class TestDataset(Dataset):
    def __init__ (self,path):
        filenames = []
        for f in os.listdir(path):
            if f.endswith('png'):
                filenames.append(os.path.join(path,f))

        self.filenames = filenames
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('L')
        img = self.transform(img)
        return img,filename.split('/')[-1]

    def __len__(self):
        return len(self.filenames)

def loadTestData(path,batch_size):
    dataset = TestDataset(path=path)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return data_loader

def loadModel(Fpth,Cpth):
    F = FeatureExtractor()
    C = Classifier()
    F.load_state_dict(torch.load(Fpth))
    C.load_state_dict(torch.load(Cpth))
    F.cuda().eval()
    C.cuda().eval()
    for param in F.parameters():
        param.requires_grad = False
    for param in C.parameters():
        param.requires_grad = False
    return F,C

def myPredict(input,output):
    testdata = loadTestData(input,128)
    if 'svhn' in input.split('/'):
        print('testing svhn...')
        F,C = loadModel('hw2_3_svhn_F1.pth','hw2_3_svhn_C1.pth')
    else:
        print('testing usps...')
        F,C = loadModel('hw2_3_usps_F.pth','hw2_3_usps_C.pth')
    with open(output, 'w') as f:
        f.writelines('image_name, label\n')

    for i, (images,filename) in enumerate(testdata, 0):
        outputs = C(F(images.cuda()))
        _, predicted = torch.max(outputs, 1)
        for i in range(len(filename)):
            with open(output, 'a') as f:
                f.writelines(filename[i]+','+str(int(predicted[i]))+'\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Enter 2 arg')
        sys.exit()

    myPredict(sys.argv[1],sys.argv[2])

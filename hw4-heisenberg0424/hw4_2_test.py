from hw4_2 import *
import sys
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Enter 3 arg')
        sys.exit()

    input_csv,input_dir,output_csv = sys.argv[1],sys.argv[2],sys.argv[3]

    model = loadModel(' ')
    model.load_state_dict(torch.load('hw4_2.pth'))
    model.cuda().eval()
    for param in model.parameters():
        param.requires_grad = False

    dataloader = loadData(input_dir,64,input_csv)

    with open(output_csv, 'w') as f:
        f.writelines('id,filename,label\n')

    id = 0
    for images,filename in dataloader:
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)
        for i in range(len(filename)):
            with open(output_csv, 'a') as f:
                f.writelines(str(id)+','+filename[i]+','+label2str[int(predicted[i])]+'\n')
                id+=1
from hw2_2 import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Enter 1 arg')
        sys.exit()
    mymodel = loadModel('hw2_2.pth')
    mymodel.mysample(500, (3, 28, 28), device='cuda', guide_w=2.0,path = sys.argv[1])
    mymodel.mysample(500, (3, 28, 28), device='cuda', guide_w=2.0,path = sys.argv[1],second=True)

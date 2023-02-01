from hw1_1 import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Enter 2 args')
        sys.exit()
    myPredict('hw1_1.pth',sys.argv[1],sys.argv[2])

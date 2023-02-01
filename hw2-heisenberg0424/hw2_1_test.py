from hw2_1 import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Enter 1 arg')
        sys.exit()
    myPredict('hw2_1.pth',sys.argv[1])

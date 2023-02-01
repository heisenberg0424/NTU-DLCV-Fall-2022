from hw1_2 import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Enter 2 args')
        sys.exit()
    myPredict('hw1_2.pth',sys.argv[1], sys.argv[2])
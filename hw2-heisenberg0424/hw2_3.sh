#!/bin/bash
gdown 1IiAE7k1JmyiNXk08OgZ1CLepftnrl4JZ -O hw2_3_svhn_C1.pth
gdown 1IXa_Rng_JxaX2uLpw2caklQT1o8ooFkf -O hw2_3_svhn_F1.pth
gdown 1816VTbPofz9jW549_wEG2auikqrjpctf -O hw2_3_usps_C.pth
gdown 1ZrpAV44Y1Zh8h8qrJBwILFvxcdrc7YpZ -O hw2_3_usps_F.pth
python3 hw2_3_test.py $1 $2

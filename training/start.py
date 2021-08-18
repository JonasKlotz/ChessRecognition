import sys

# set base path
sys.path.append("/home/users/j/jonasklotz/ChessRecognition")

from training.mobile_nasnet import start_training as nasnet
from training.mobilenet_v2 import start_training as mobilenet
from training.resnet_v2 import start_training as resnet
from training.xception import start_training as xceptio

if __name__ == '__main__':
    mobilenet()
    nasnet()
    xceptio()
    resnet()

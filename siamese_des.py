# coding:UTF-8
"""
Created on July 16,2018
@author: dapinglee & yangxiaodong
@university: CUIT
"""
import sys
import caffe
import numpy as np
sys.path.insert(1, '/usr/local/caffe-master/python')


def siamese_des(img1, img2):

    num1 = img1.size
    num2 = img2.size

    # get patch
    height1, width1 = img1.shape[0], img1.shape[1]
    height2, width2 = img2.shape[0], img2.shape[1]

    padding1 = np.zeros((height1 + 64, width1 + 64))
    padding2 = np.zeros((height2 + 64, width2 + 64))

    padding1[32:height1 + 32, 32:width1 + 32] = img1
    padding2[32:height2 + 32, 32:width2 + 32] = img2

    patch1 = np.zeros((num1, 64, 64))
    patch2 = np.zeros((num2, 64, 64))

    k = 0
    for i in range(32, height1+32):
        for j in range(32, width1+32):
            patch_L = int(i - 32)
            patch_R = int(i + 32)
            patch_T = int(j - 32)
            patch_B = int(j + 32)
            patch1[k, :, :] = padding1[patch_L:patch_R, patch_T:patch_B]
            k += 1

    k = 0
    for i in range(32, height2+32):
        for j in range(32, width2+32):
            patch_L = int(i - 32)
            patch_R = int(i + 32)
            patch_T = int(j - 32)
            patch_B = int(j + 32)
            patch2[k, :, :] = padding2[patch_L:patch_R, patch_T:patch_B]
            k += 1

    # compute siamese feature
    MODEL_FILE = '/home/yangxiaodong/caffe/matchnet/models/matchnet.prototxt'
    # decrease if you want to preview during training
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/yxdLOG/matchnet_iter_10000.caffemodel'
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/matchnet_iter_30000.caffemodel'
    PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/weights/matchnet_iter_30000.caffemodel'
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

    raw_data1 = patch1.reshape(num1, 1, 64, 64)
    raw_data2 = patch2.reshape(num2, 1, 64, 64)

    caffe_in1 = (raw_data1 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out1 = net.forward_all(data=caffe_in1)

    caffe_in2 = (raw_data2 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out2 = net.forward_all(data=caffe_in2)

    raw_feat1 = out1['bottleneck']
    raw_feat2 = out2['bottleneck']

    # Adjust the formatting of features
    feat1 = np.zeros((height1, width1, 512))
    feat2 = np.zeros((height2, width2, 512))

    k = 0
    for i in range(height1):
        for j in range(width1):
            feat1[i][j] = raw_feat1[k]
            k += 1

    k = 0
    for i in range(height2):
        for j in range(width2):
            feat2[i][j] = raw_feat2[k]
            k += 1

    return feat1, feat2

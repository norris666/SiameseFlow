import numpy as np
import sys
sys.path.insert(0, '/usr/local/caffe-master/python')
import caffe


# add every point to array
def get_point(img1, img2):
    point1 = []
    point2 = []

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            point1.append((i, j))
    point1 = np.array(point1)

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            point2.append((i, j))
    point2 = np.array(point2)

    return point1, point2


def get_patch(img1, img2, kp1, kp2):
    height1, width1 = img1.shape[0], img1.shape[1]
    height2, width2 = img2.shape[0], img2.shape[1]

    padding1 = np.zeros((height1 + 64, width1 + 64))
    padding2 = np.zeros((height2 + 64, width2 + 64))

    padding1[32:height1 + 32, 32:width1 + 32] = img1
    padding2[32:height2 + 32, 32:width2 + 32] = img2

    patch1 = np.zeros((len(kp1), 64, 64))
    patch2 = np.zeros((len(kp2), 64, 64))

    kp1 = np.array(kp1)
    kp1 = kp1 + 32

    kp2 = np.array(kp2)
    kp2 = kp2 + 32

    for i in range(len(kp1)):
        x = kp1[i][0]
        y = kp1[i][1]
        patch_L = int(x - 32)
        patch_R = int(x + 32)
        patch_T = int(y - 32)
        patch_B = int(y + 32)
        patch1[i, :, :] = padding1[patch_L:patch_R, patch_T:patch_B]

    for i in range(len(kp2)):
        x = kp2[i][0]
        y = kp2[i][1]
        patch_L = int(x - 32)
        patch_R = int(x + 32)
        patch_T = int(y - 32)
        patch_B = int(y + 32)
        patch1[i, :, :] = padding2[patch_L:patch_R, patch_T:patch_B]
    return patch1, patch2

def siamese_des(patch1, patch2, kp1, kp2):

    MODEL_FILE = '/home/yangxiaodong/caffe/matchnet/models/matchnet.prototxt'
    # decrease if you want to preview during training
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/yxdLOG/matchnet_iter_10000.caffemodel'
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/matchnet_iter_30000.caffemodel'
    PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/weights/matchnet_iter_30000.caffemodel'
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

    raw_data1 = patch1.reshape(len(kp1), 1, 64, 64)
    raw_data2 = patch2.reshape(len(kp2), 1, 64, 64)

    caffe_in1 = (raw_data1 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out1 = net.forward_all(data=caffe_in1)

    caffe_in2 = (raw_data2 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out2 = net.forward_all(data=caffe_in2)

    feat1 = out1['bottleneck']
    feat2 = out2['bottleneck']

    return feat1, feat2


def getFinaldes(img1, img2, feat1, feat2):
    feature1 = np.zeros((img1.shape[0], img1.shape[1], 512))
    feature2 = np.zeros((img2.shape[0], img2.shape[1], 512))

    count = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            feature1[i][j] = feat1[count]
            count += 1

    count = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            feature2[i][j] = feat2[count]
            count += 1

    return feature1, feature2
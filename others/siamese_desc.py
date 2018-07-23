from others.generatePatches import *
import sys
sys.path.insert(0, '/usr/local/caffe-master/python')
import caffe
#caffe('set_device',1)


def siamese_des(from generatePatches import *
import sys
sys.path.insert(0, '/usr/local/caffe-master/python')
    pass

):
    patchesFix,patchesMov = generate_patches(fixedImg,movingImg,fixedPoint,movingPoint)

    MODEL_FILE = '/home/yangxiaodong/caffe/matchnet/models/matchnet.prototxt'
    # decrease if you want to preview during training
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/yxdLOG/matchnet_iter_10000.caffemodel'
    # PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/matchnet_iter_30000.caffemodel'
    PRETRAINED_FILE = '/home/yangxiaodong/matchnet/models/weights/matchnet_iter_30000.caffemodel'
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

    raw_data1 = patchesFix.reshape(len(fixedPoint), 1, 64, 64)
    raw_data2 = patchesMov.reshape(len(movingPoint), 1, 64, 64)
    # print patch.shape
    # raw_data = patch.reshape(1, 1, 64, 64)


    caffe_in1 = (raw_data1 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out1 = net.forward_all(data=caffe_in1)

    caffe_in2 = (raw_data2 - 128) * 0.00625  # manually scale data instead of using `caffe.io.Transformer`
    out2 = net.forward_all(data=caffe_in2)

    feat1 = out1['bottleneck']
    feat2 = out2['bottleneck']
    return feat1, feat2
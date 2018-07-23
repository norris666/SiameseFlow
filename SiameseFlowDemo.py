# coding:UTF-8
"""
Created on 2018年7月20日
@author: dapinglee
"""
import cv2
import time
import matlab.engine
import scipy.io as sio
import SimpleITK as sitk

from siamese_des import siamese_des
from siamese_feat import feat2rgb

# start time
start = time.time()
# read image
img1 = cv2.imread("MR.bmp")  # fixedImg p.s. -1 to read 8-bit image
img2 = cv2.imread("CT_rotate.bmp")  # movingImg

# rgb2gray  p.s. do it while the image is not 8-bit
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# compute dense siamese feature
feature1, feature2 = siamese_des(img1, img2)

# visualize the feature
feature1_rgb = feat2rgb(feature1)
feature2_rgb = feat2rgb(feature2)

# save raw.mat
sio.savemat('raw.mat', {'img1': img1, 'img2': img2, 'feature1': feature1, 'feature2': feature2})

# start matlab engine
eng = matlab.engine.start_matlab()

# running the demo
eng.SiameseFlowDemo(nargout=0)

# load source.mat
source = sio.loadmat('source.mat')

# vx = source['vx']  # if you need it
# vy = source['vy']  # if you need it
warpI2 = source['warpI2']
grayerror = source['grayerror']
siamese_flow_rgb = source['siamese_flow_rgb']

# rgb error
fixed = sitk.GetImageFromArray(img1)
out = sitk.GetImageFromArray(warpI2)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
rgberror = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
rgberror = sitk.GetArrayFromImage(rgberror)

# stop time
elapsed = time.time() - start
print('Time used: {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))

# show the results
cv2.imshow('Fixed Image', img1)
cv2.imshow('Moving Image', img2)

cv2.imshow('Siamese Feature Image1', feature1_rgb)
cv2.imshow('Siamese Feature Image2', feature2_rgb)

cv2.imshow('Warped Image', warpI2)
cv2.imshow('Gray Error', grayerror)
cv2.imshow('RGB Error', rgberror)

cv2.imshow('Siamese Flow Image', siamese_flow_rgb)

k = cv2.waitKey(0)  # wait for ESC key to exit
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait fos 's' key to save and exit
    cv2.imwrite('./results/Fixed Image.png', img1)
    cv2.imwrite('./results/Moving Image.png', img2)

    cv2.imwrite('./results/Siamese Feature Image1.png', feature1_rgb)
    cv2.imwrite('./results/Siamese Feature Image2.png', feature2_rgb)

    cv2.imwrite('./results/Warped Image.png', warpI2)
    cv2.imwrite('./results/Gray Error.png', grayerror)
    cv2.imwrite('./results/RGB Error.png', rgberror)

    cv2.imwrite('./results/Siamese Flow Image.png', siamese_flow_rgb)

    cv2.destroyAllWindows()

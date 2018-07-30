# coding:UTF-8
"""
Created on July 20,2018
@author: DapingLee
@university: CUIT
"""
import cv2
import time
import matlab.engine
import scipy.io as sio
import SimpleITK as sitk

from siamese_des import siamese_des
from siamese_feat import feat2rgb
from joint_images import joint

# start time
start = time.time()
# read image
img1 = cv2.imread('source/fixed1.jpg')  # fixedImg p.s. -1 to read 8-bit image
img2 = cv2.imread('source/moving.jpg')  # movingImg

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

# joint results
images = {'fixed': img1, 'moving': img2, 'flow': siamese_flow_rgb,
          'warped': warpI2, 'gray': grayerror, 'rgb': rgberror,
          'feat1': feature1_rgb, 'feat2': feature2_rgb}
image = joint(images)

# show the results
cv2.imshow('results', image)
k = cv2.waitKey(0)  # wait for ESC key to exit
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait fos 's' key to save and exit
    cv2.imwrite('results/results.bmp', image)

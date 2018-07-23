import SimpleITK as sitk
import cv2
import numpy as np
fixed = sitk.ReadImage('../results/Fixed Image.png', sitk.sitkFloat32)
out = sitk.ReadImage('../results/Warped Image.png', sitk.sitkFloat32)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
cimg = sitk.GetArrayFromImage(cimg)
cv2.imshow('test',cimg)
cv2.waitKey(0)
# sitk.WriteImage(cimg, 'cimg.bmp')
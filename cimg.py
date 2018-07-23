import SimpleITK as sitk
fixed = sitk.ReadImage('fixedImg_0_37.bmp',sitk.sitkFloat32)
out = sitk.ReadImage('sift_registration_0_37_1_0_0.95_2_0.0213.bmp',sitk.sitkFloat32)

simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)

sitk.WriteImage(cimg,'C:\Users\Yangxiaodong\Desktop\\result_sliceV2\sift\sift_show1\cimg.bmp')
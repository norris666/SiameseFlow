import cv2

movingImg_GT = cv2.imread('CT.bmp')
rows = movingImg_GT.shape[0]
cols = movingImg_GT.shape[1]
TM_MOV = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
movingImg = cv2.warpAffine(movingImg_GT, TM_MOV, (cols, rows))
movingImg = cv2.cvtColor(movingImg, cv2.COLOR_BGR2GRAY)
cv2.imwrite('CT_rotate.bmp',movingImg)
cv2.imshow('image1',movingImg_GT)
cv2.imshow('image2',movingImg)
cv2.waitKey(0)

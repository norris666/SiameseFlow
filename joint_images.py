# coding:UTF-8
"""
Created on July 29,2018
@author: DapingLee
@university: CUIT
"""
import cv2
import numpy as np


def joint(images):
    fixed = images['fixed']
    moving = images['moving']
    flow = images['flow']
    warped = images['warped']
    gray = images['gray']
    rgb = images['rgb']
    feat1 = images['feat1']
    feat2 = images['feat2']

    fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
    moving = cv2.cvtColor(moving, cv2.COLOR_GRAY2BGR)
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    height = fixed.shape[0]
    width = fixed.shape[1]

    image = np.zeros((height*2+40, width*4+6, 3), dtype='uint8')
    image[:] = (255, 255, 255)
    image[20:height+20, 0:width, :] = fixed
    image[height+40:height*2+40, 0:width, :] = moving
    image[20:height+20, width+2:width*2+2, :] = flow
    image[height+40:height*2+40, width+2:width*2+2, :] = warped
    image[20:height+20, width*2+4:width*3+4, :] = gray
    image[height+40:height*2+40, width*2+4:width*3+4, :] = rgb
    image[20:height+20, width*3+6:width*4+6, :] = feat1
    image[height+40:height*2+40, width*3+6:width*4+6, :] = feat2

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    image = cv2.putText(image, 'Fixed Image', (0, 14), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Moving Image', (0, 34+height), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Flow Image', (width+2, 14), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Warped Image', (width+2, 34+height), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Gray Error', (width*2+4, 14), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'RGB Error', (width*2+4, 34+height), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Feature1 Image', (width*3+6, 14), font, 1, (0, 0, 255), 1)
    image = cv2.putText(image, 'Feature2 Image', (width*3+6, 34+height), font, 1, (0, 0, 255), 1)

    return image

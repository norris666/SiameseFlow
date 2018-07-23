import numpy as np


# this function is generate 64 * 64 size patches from two image though point which detceted by sift
def generate_patches(img1,img2,point1,point2):
    m1, n1 = img1.shape
    m2, n2 = img2.shape

    img1_pad = np.zeros((m1+64,n1+64)).astype(np.uint8)
    img2_pad = np.zeros((m2+64,n2+64)).astype(np.uint8)

    img1_pad[32:m1+32,32:n1+32] = img1
    img2_pad[32:m2+32,32:n2+32] = img2

    patch1 = np.zeros((len(point1),64,64))
    patch2 = np.zeros((len(point2),64,64))

    point1 = np.array(point1) + 32
    point2 = np.array(point2) + 32

    for i in range(len(point1)):
        y = point1[i][0]
        x = point1[i][1]
        patch_L = int(x - 32)
        patch_R = int(x + 32)
        patch_T = int(y - 32)
        patch_B = int(y + 32)
        patch1[i,:,:] = img1_pad[patch_L:patch_R, patch_T:patch_B]

    for i in range(len(point2)):
        y = point2[i][0]
        x = point2[i][1]
        patch_L = int(x - 32)
        patch_R = int(x + 32)
        patch_T = int(y - 32)
        patch_B = int(y + 32)
        patch2[i,:,:] = img2_pad[patch_L:patch_R, patch_T:patch_B]

    return patch1, patch2

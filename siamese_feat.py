# coding:UTF-8
"""
Created on 2018年7月22日
@author: dapinglee
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def feat2rgb(feature):
    # add feature to list
    feat = []
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feat.append(feature[i][j])

    feat = np.array(feat)

    # max_index = np.argsort(-feat, axis=1)  # sort the matrix by row

    # PCA process
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat)
    # normalization
    scaler = MinMaxScaler(feature_range=(0, 255))
    feat_pca = scaler.fit_transform(feat_pca)

    feat_img = np.zeros((feature.shape[0], feature.shape[1], 3), dtype='uint8')  # create a ndarray

    # add the feat to feat_img
    k = 0
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feat_img[i][j][0] = feat_pca[k][0]
            feat_img[i][j][1] = feat_pca[k][1]
            feat_img[i][j][2] = feat_pca[k][2]
            k += 1

    return feat_img

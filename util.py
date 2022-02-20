"""This code is completed by Cimy Wang.
 If you have any question, please contact me fell free.
 e-mail: jinping_wang@foxmail.com"""

from keras.layers import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random
from sklearn import preprocessing


#    create the image patches
def createPatches(X, y, windowSize, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[1] - margin):
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c -
                                                           margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def DiscriminantAnalysis(X, Y, numComponents):
    Y = Y.reshape(-1)
    newX = np.reshape(X, (-1, X.shape[2]))
    model_lda = LinearDiscriminantAnalysis(n_components=numComponents)
    newX = model_lda.fit(newX, Y).transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def report(true_label, y_pred):
    classification = classification_report(true_label, y_pred)
    confusion = confusion_matrix(true_label, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    List = []
    List.append(np.array(oa)), List.append(np.array(aa)), List.append(np.array(K))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    # ==== Print table accuracy====
    ind = [list(range(1, len(ca) + 1, 1)) + ['OA', 'AA', 'KA']][0]
    target_names = [u'%s' % l for l in ind]
    last_line_heading = 'avg / total'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), 2)
    headers = ["accuracy"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'
    rows = zip(target_names, accuracy_matrix)
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' u'\n'
    for row1 in rows:
        report += row_fmt.format(*row1, width=width, digits=5)
    report += u'\n'
    # print(report)
    return classification, confusion, accuracy_matrix


def random_sample(train_sample, validate_sample, patchesLabels):
    num_classes = np.max(patchesLabels)
    dataList = patchesLabels
    TrainIndex = []
    TestIndex = []
    ValidateIndex = []

    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(patchesLabels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp+validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)

    return TrainIndex, ValidateIndex, TestIndex


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def CalAccuracy(predict, label):
    n = label.shape[0]
    OA = np.sum(predict == label) * 1.0 / n
    correct_sum = np.zeros((max(label) + 1))
    reali = np.zeros((max(label) + 1))
    predicti = np.zeros((max(label) + 1))
    producerA = np.zeros((max(label) + 1))

    for i in range(0, max(label) + 1):
        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)
        reali[i] = np.sum(label == i)
        predicti[i] = np.sum(predict == i)
        producerA[i] = correct_sum[i] / reali[i]

    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
    return OA, Kappa, producerA

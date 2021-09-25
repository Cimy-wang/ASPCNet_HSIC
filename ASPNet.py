# -*- coding: utf-8 -*-
"""
__author__ = 'Cimy Wang'
__mtime__  = '2021/9/26'
If necessary, please contact us. e-mail: jinping_wang@foxmail.com
"""

import os
import scipy.io as scio
from keras import utils, callbacks
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *
from ASPCaps import ASPCaps, CapsuleLayer, CapsToScalars, ConvertToCaps, FlattenCaps
from ASP import ASP
import argparse
from util import createPatches, report, random_sample, applyPCA


def creat_model_aspcaps(x_train, num_classes):
    img_rows, img_cols, num_dim = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_layer = Input((img_rows, img_cols, num_dim))
    layer_01 = ASP(filters=128, kernel_size = args.KS, dilation = args.DR_1, stride = 1)(input_layer)
    layer_02 = Conv2D(filters=128, kernel_size=(1, 1), strides = (2, 2), activation='relu', padding='same')(layer_01)
    layer_03 = ASP(filters=256, kernel_size = args.KS, dilation = args.DR_1, stride = 1)(layer_02)
    layer_04 = Conv2D(filters=256, kernel_size=(1, 1), strides = (2, 2), activation='relu', padding='same')(layer_03)
    layer_05 = BatchNormalization(momentum=args.momentum)(layer_04)
    layer_06 = ConvertToCaps()(layer_05)
    layer_07 = ASPCaps(32, 4, kernel_size=(args.KS, args.KS), strides=(1, 1), dilation_rate=(args.DR_1, args.DR_1))(layer_06)
    layer_08 = ASPCaps(32, 4, kernel_size=(args.KS, args.KS), strides=(1, 1), dilation_rate=(args.DR_1, args.DR_1))(layer_07)
    layer_09 = FlattenCaps()(layer_08)
    layer_10 = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3, channels=0)(layer_09)
    output_layer = CapsToScalars()(layer_10)

    if args.numGPU > 1:
        model = Model(inputs=input_layer, outputs=output_layer)
        parallel_model = utils.multi_gpu_model(model, gpus=args.numGPU)
    else:
        parallel_model = Model(inputs=input_layer, outputs=output_layer)
    return parallel_model

parser = argparse.ArgumentParser(description="ASPCNet")
parser.add_argument("--numGPU", type=int, default="1", action="store", help="The total numbers of GPU to use")
parser.add_argument("--GPUid", type=int, default="0", action="store", help="The specific numbers of GPU to use")
parser.add_argument("--dir", type=str, default="./data", action="store", help="input(default: ./data)")
parser.add_argument("--WindowsSize", type=int, default="29", action="store", help="The Patch size for each pixel")
parser.add_argument("--N_C", type=int, default="15", action="store", help="The reducing demision")
parser.add_argument("--batch_size", type=int, default="96", action="store", help="the minimal training number in each iteration")
parser.add_argument("--epochs", type=int, default="500", action="store", help="the training epoch")
parser.add_argument("--KS", type=int, default="3", action="store", help="The convolution kernel size")
parser.add_argument("--DR_1", type=int, default="3", action="store", help="The Dilation Rate")
parser.add_argument("--epsilon", type=float, default="1e-08", action="store", help="epsilon of adam optimizer",)
parser.add_argument("--beta_2", type=float, default="0.999", action="store", help="the second parameter of adam optimizer")
parser.add_argument("--beta_1", type=float, default="0.9", action="store", help="the first parameter of adam optimizer")
parser.add_argument("--learning_rate", type=float, default="0.001", action="store", help="learning rate")
parser.add_argument("--patience", type=int, default="40", action="store", help="early stopping step")
parser.add_argument("--momentum", type=float, default="0.9", action="store", help="BN layer")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPUid)
for root, dirs, files in os.walk(args.dir):
    print('||======The Training set is localed in [%s/%s]======||' %(root, files[0]))
labels = scio.loadmat(root + '/' + files[0])['ground']
data1 = scio.loadmat(root + '/' + files[0])['img']

num_classes = labels.max()

data, pca = applyPCA(data1, numComponents=args.N_C)

patchesData, patchesLabels = createPatches(data, labels, windowSize=args.WindowsSize)
patchesLabels = patchesLabels.astype(np.int32)
train_sample = (40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40)
validate_sample = [8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4]

trainIndex, valIndex, testIndex = random_sample(train_sample, validate_sample, patchesLabels)
x_train, x_test, x_val = patchesData[trainIndex, :, :, :], \
                         patchesData[testIndex, :, :, :],\
                         patchesData[valIndex, :, :, :]
y_train, y_test, y_val = utils.to_categorical(patchesLabels[trainIndex] - 1, num_classes), \
                         utils.to_categorical(patchesLabels[testIndex] - 1, num_classes), \
                         utils.to_categorical(patchesLabels[valIndex] - 1, num_classes)
true_label = patchesLabels[testIndex]

parallel_model = creat_model_aspcaps(x_train, num_classes)

parallel_model.compile(loss=lambda y_true, y_pred_t: y_true * K.relu(0.9 - y_pred_t) ** 2 +
                                                     0.25 * (1 - y_true) * K.relu(y_pred_t - 0.1) ** 2,
                       optimizer=Adam(lr=args.learning_rate,
                                      beta_1=args.beta_1,
                                      beta_2=args.beta_2,
                                    epsilon=args.epsilon),
                       metrics=['accuracy'])

callback = callbacks.EarlyStopping(monitor='val_acc',
                                   min_delta=0,
                                   patience=args.patience,
                                   verbose=1,
                                   mode='auto',
                                   restore_best_weights=True)

history = parallel_model.fit(x_train,
                            y_train,
                            verbose=2,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[callback])

print('======================Train End============================')
y_pred = np.argmax(parallel_model.predict(x_test), axis=1) + 1
classification, confusion, accuracy_matrix = report(true_label, y_pred)
print(accuracy_matrix)
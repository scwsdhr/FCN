#!usr/bin/python
#-*- coding:utf-8 -*-

'''
Fully Convolutional Neural Network Based on VGG16
'''

'''
Standard libraries
'''
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape, Dropout, Conv2D, Lambda
from keras.layers.merge import add
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16

'''
Custom functions
'''
from BilinearUpSampling import BilinearUpSampling2D

def fc_vgg16(filters=4096, weight_decay=0., block_name='block5'):
    '''
    Add two fully connected layer to the model
    '''
    def f(x):
        x = Conv2D(filters=filters,
                kernel_size=(7, 7),
                activation='relu',
                padding='same',
                dilation_rate=(2, 2),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay),
                name='{}_fc6'.format(block_name))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=filters,
                kernel_size=(1, 1),
                activation='relu',
                padding='same',
                dilation_rate=(2, 2),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay),
                name='{}_fc7'.format(block_name))(x)
        x = Dropout(0.5)(x)
        return x
    return f

def upsampling_vgg16(classes, target_shape=None, scale=1, weight_decay=0., block_name='featx'):
    '''
    Upsampling layers
    '''
    def f(x, y):
        score = Conv2D(filters=classes,
                kernel_size=(1, 1),
                activation='linear',
                padding='valid',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay),
                name='score_{}'.format(block_name))(x)
        if y is not None:
            def scaling(xx, ss=1):
                return xx*ss
            scaled = Lambda(scaling, arguments={'ss': scale},
                    name='scale_{}'.format(block_name))(score)
            score = add([y, scaled])
        upscore = BilinearUpSampling2D(target_shape=target_shape,
                name='upscore_{}'.format(block_name))(score)
        return upscore
    return f

def UpSampler_vgg16(pyramid, scales, classes, weight_decay=0.):
    '''
    Upsampling block
    '''
    blocks = []

    for i in range(len(pyramid) - 1):
        block_name = 'feat{}'.format(i + 1)
        block = upsampling_vgg16(classes=classes,
                target_shape=K.int_shape(pyramid[i+1]),
                scale=scales[i],
                weight_decay=weight_decay,
                block_name=block_name)
        blocks.append(block)

    decoded = None

    for feat, blk in zip(pyramid[:-1], blocks):
        decoded = blk(feat, decoded)

    return decoded

def fcn_vgg16(input_shape, classes, weight_decay=0.,
        trainable_encoder=True, weights='imagenet'):
    '''
    Construct model
    '''
    inputs = Input(shape=input_shape)

    pyrmid_layers = 3

    # Use VGG16 as the encoder
    encoder = VGG16(include_top=False, input_shape=input_shape, weights=weights, classes=classes)

    # Set parameters in VGG16 to be untrainable
    for layer in encoder.layers:
        layer.trainable = False

    first = True
    pyramid = []
    for layer in encoder.layers:
        if first:
            x = layer(inputs)
            first = False
            pyramid.append(x)
        else:
            x = layer(x)
            if x.shape[1:-1] != pyramid[-1].shape[1:-1]:
                pyramid.append(x)

    x = fc_vgg16(filters=4096)(x)
    pyramid[-1] = x

    feat_pyramid = pyramid[::-1][:pyrmid_layers]
    feat_pyramid.append(inputs)

    outputs = UpSampler_vgg16(feat_pyramid,
            scales=[1, 1e-2, 1e-4],
            classes=classes,
            weight_decay=weight_decay)

    scores = Activation('softmax')(outputs)

    return Model(inputs=inputs, outputs=scores)

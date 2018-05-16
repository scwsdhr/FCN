#!/usr/bin/python
#-*- coding:utf-8 -*-

from keras.utils import plot_model
import model

def plot_fcn_vgg16():
    input_shape = (224, 224, 3)
    fcn_vgg16 = model.fcn_vgg16(input_shape=input_shape, classes=21)
    plot_model(fcn_vgg16, to_file='fcn_vgg16.png', show_shapes=True)

if __name__ == '__main__':
    plot_fcn_vgg16()

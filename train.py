#!/usr/bin/python
#-*- coding:utf-8 -*-

'''
Standard libraries
'''
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint,
        ReduceLROnPlateau,
        EarlyStopping,
        TerminateOnNaN,
        CSVLogger)
from keras import Input
from voc_generator import PascalVocGenerator, ImageSetLoader
import numpy as np

'''
Custom libraries
'''
import model

# Clear session
K.clear_session()

def arg_gen(dataset_name):
    '''
    Generate arguments of dataset generators
    '''
    image_set = 'data/VOC2011/ImageSets/Segmentation/{}.txt'.format(dataset_name)
    image_dir = 'data/VOC2011/JPEGImages/'
    label_dir = 'data/VOC2011/SegmentationClass/'
    target_size = (224, 224)
    return image_set, image_dir, label_dir, target_size

def main():
    '''
    Main function
    '''

    # Define common arguments
    checkpointer = ModelCheckpoint(
            filepath='output/fcn_vgg16_weights_tmp.h5',
            verbose=1,
            save_best_only=True)
    lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=10,
            min_lr=1e-12)
    early_stopper = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30)
    nan_terminator = TerminateOnNaN()
    csv_logger = CSVLogger('output/tmp_fcn_vgg16.csv')

    # Set data generator
    datagen = PascalVocGenerator(image_shape=[224, 224, 3],
            image_resample=True,
            pixelwise_center=True,
            pixel_mean=[115.85100, 110.50989, 102.16182],
            pixelwise_std_normalization=True,
            pixel_std=[70.30930, 69.41244, 72.60676])

    # Define training set and validation set
    train_loader = ImageSetLoader(*arg_gen('train'))
    val_loader = ImageSetLoader(*arg_gen('val'))

    # Construct model
    fcn_vgg16 = model.fcn_vgg16(input_shape=(224, 224, 3),
            classes=21,
            weight_decay=3e-3,
            weights='imagenet',
            trainable_encoder=False)

    # Set optimizer
    optimizer = Adam(1e-4)

    # Compile model
    fcn_vgg16.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Fit model with the above generators
    fcn_vgg16.fit_generator(
        datagen.flow_from_imageset(
            class_mode='categorical',
            classes=21,
            batch_size=1,
            shuffle=True,
            image_set_loader=train_loader),
        steps_per_epoch=1112,
        epochs=40,
        validation_data=datagen.flow_from_imageset(
            class_mode='categorical',
            classes=21,
            batch_size=1,
            shuffle=True,
            image_set_loader=val_loader),
        validation_steps=1111,
        verbose=1,
        callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer, nan_terminator])

    # Save weights
    fcn_vgg16.save('output/fcn_vgg16.h5')

if __name__ == '__main__':
    main()

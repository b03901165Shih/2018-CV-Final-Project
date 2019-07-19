import numpy as np
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import h5py
import glob
import matplotlib.pyplot as plt
#np.random.seed(2019)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
from keras.layers.normalization import BatchNormalization
from preprocess import getTrain
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

SEED=2019

def augment_data( generator, X1, X2, y, batch_size = 32 ):
    generator_seed = np.random.randint( SEED )
    gen_X1 = generator.flow( X1, y, 
                             batch_size = batch_size, seed = generator_seed )
    gen_X2 = generator.flow( X2, y, 
                             batch_size = batch_size, seed = generator_seed )
    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()
        yield [ X1i[0], X2i[0] ], X1i[1]

def unison_shuffled_copies(a, b, c):
    p = np.random.permutation(a.shape[0])
    return a[p], b[p], c[p]

# Define the parameters for training
batch_size = 256
nb_classes = 2
nb_epoch = 120

# input image dimensions
img_rows, img_cols = 11, 11

# Volume of the training set
#sample_number = 10000#430608

nb_filters = 112

# CNN kernel size
kernel_size = (3,3)

# Here some additional preprocess methods like rotation etc. could be added.
input_shape = (img_rows, img_cols, 1)

for i in range(1):
    #finetune = True
    if(i==0):
        finetune = True

    tic = time.time()
    (X1_train, X2_train, y_train)  = getTrain()
    toc = time.time()
    print ("Time for loading the training set: ", toc-tic)

    # Briefly check some patches. Positive-matching patches are expected to be of similar features. We store two left patches in X1_train. One for matching the positve right patch in X2_train. The other for matching negative right patch in X2_train.
    X1_train = X1_train.astype('float32').reshape((X1_train.shape[0],img_rows, img_cols, 1))
    X2_train = X2_train.astype('float32').reshape((X1_train.shape[0],img_rows, img_cols, 1))

    X1_train, X2_train, y_train =unison_shuffled_copies(X1_train, X2_train, y_train)
	
    valid_split = 0.9
	
    train_size  = (int)(valid_split*X1_train.shape[0])
    
    X1_train_split = X1_train[:train_size]
    X2_train_split = X2_train[:train_size]
    y_train_split  = y_train[:train_size]
	
    X1_valid_split = X1_train[train_size:]
    X2_valid_split = X2_train[train_size:]
    y_valid_split  = y_train[train_size:]
	
    print('X1_valid_split.shape:',X1_valid_split.shape)
    print('X2_valid_split.shape:',X2_valid_split.shape)
    print('y_valid_split.shape :',y_valid_split.shape)

    datagen = ImageDataGenerator(
        rotation_range     = 20,
        #width_shift_range  = 0.1,
        #height_shift_range = 0.1,
        shear_range		   = 0.1,
        #zoom_range		   = [0.8,1],
        #channel_shift_range= 0.1,
        horizontal_flip    = True,
        vertical_flip      = True
    )

    train_generator = augment_data( datagen, X1_train_split,  X2_train_split, y_train_split, batch_size = batch_size )
    '''
    [X1b, X2b], y = next(train_generator)
    print(X1b)
    print(np.array(X1b).shape)
    print(np.array(X2b).shape)
    for k in range(len(X1b)):
        plt.imshow(np.concatenate([X1_train_split[k,:,:,0],X2_train_split[k,:,:,0],X1b[k,:,:,0],X2b[k,:,:,0]],1)); plt.show()'''
	
    # This neural network is working finely and ends up with a training accuracy of more than 90%. 
    #for i in range(3):
    #y_train = np.expand_dims(y_train,axis=2)
    print ('X1_train.shape',X1_train.shape)
    print ('y_train.shape',y_train.shape)

    '''
    for i in range(sample_number>>1):
        print(y_train[2*i],y_train[2*i+1])
        plt.imshow(np.concatenate([X1_train[2*i], X2_train[2*i], X2_train[2*i+1]],1)); plt.show()
    ''' 

    left_inputs = Input(input_shape)
    right_inputs = Input(input_shape)

    Conv1   = Conv2D(nb_filters, kernel_size, padding='valid', activation='relu')
    Conv2   = Conv2D(nb_filters, kernel_size, padding='valid', activation='relu')
    Conv3   = Conv2D(nb_filters, kernel_size, padding='valid', activation='relu')
    Conv4   = Conv2D(nb_filters, kernel_size, padding='valid', activation='relu')
    sub_net = Conv2D(nb_filters, kernel_size, padding='valid', activation='relu')
    left_branch  = sub_net(Conv4(Conv3(Conv2(Conv1(left_inputs )))))
    right_branch = sub_net(Conv4(Conv3(Conv2(Conv1(right_inputs)))))
    
    merged = Concatenate(axis=-1)([left_branch, right_branch])
    
    ft = Flatten()(merged)
    dn1 = Dense(384, activation='relu')(ft)
    dn2 = Dense(384, activation='relu')(dn1)
    dn3 = Dense(384, activation='relu')(dn2)
    output = Dense(1, activation='sigmoid')(dn3)
    
    fc = Model([left_inputs,right_inputs],output)
    fc.summary()
    
    if(finetune):
        fc.load_weights(filepath='my_mccnn_new4.h5')
    
    #callbacks = [ModelCheckpoint(filepath='my_mccnn_new3.h5', verbose=1, save_best_only=True)]
	
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=0, verbose=1, min_lr=1e-8),
        ModelCheckpoint(monitor='val_loss', filepath='my_mccnn_new4.h5', verbose=1, save_best_only=True, mode='auto')
    ]

    optimizer = optimizers.adam(lr=2e-4, decay=1e-7)#optimizers.SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)#optimizers.adam(lr=8e-5, decay=1e-8)
    
    fc.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	
    fc.fit_generator( train_generator, 
                     steps_per_epoch = len(X1_train_split)//batch_size, 
                     epochs = nb_epoch,
                     callbacks = callbacks, verbose = 1, 
                     validation_data = [[X1_valid_split, X2_valid_split], y_valid_split] )
	
    #fc.fit([X1_train,X2_train], y_train, validation_split=0.1, batch_size=batch_size, epochs = nb_epoch, shuffle=True, callbacks = callbacks)
    # Evaluate the result based on the training set
    #score = fc.evaluate([X1_train,X2_train], y_train, verbose=0)
    
    # print score.shape
    #fc.save('my_mccnn_new4.h5') 
    #print('Test score: ', score[0])
    #print('Test accuracy: ', score[1])
    



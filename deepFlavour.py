from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate,Softmax,Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import tensorflow as tf
from keras import backend as K

import math

def block_deepFlavourConvolutions(cpf_input,npf_input,vtx_input,dropoutRate,active=True,l2=0.00001,batchnorm=False,batchmomentum=0.6,add_summary=False):
    with tf.name_scope('cpf_conv'):
        if active:
            cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(cpf_input)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                
            cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                  
            cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                  
            cpf = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(cpf)
            #if batchnorm:
            #    cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            #cpf = Dropout(dropoutRate)(cpf)   
        else:
            cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf_input)
        
    with tf.name_scope('npf_conv'):
        if active:
            npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(npf_input)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf)
            npf = Dropout(dropoutRate)(npf)
            npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(npf)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf)
            npf = Dropout(dropoutRate)(npf)
            npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(npf)
            #if batchnorm:
            #    npf = BatchNormalization(momentum=batchmomentum)(npf)
            #npf = Dropout(dropoutRate)(npf)
        else:
            npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf_input)

    with tf.name_scope('vtx_conv'):
        if active:
            vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(vtx_input)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(vtx)
            #vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            #vtx = Dropout(dropoutRate)(vtx)
        else:
            vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx_input)
    if add_summary:
        tf.summary.histogram('conv_cpf', tf.reduce_mean(cpf,axis=1))
        tf.summary.histogram('conv_npf', tf.reduce_mean(npf,axis=1))
        tf.summary.histogram('conv_vtx', tf.reduce_mean(vtx,axis=1))
        
    return cpf,npf,vtx
    
    
def sortedConv(x,dropoutRate,active=True,l2=0.00001,batchnorm=False,batchmomentum=0.6,add_summary=False):
    entries = []
    computed_weights = []
    N = x.shape.as_list()[1]
    print "indices: ",N
    for _ in range(N):
        #calculate weights for each candidate (e.g. [?,25,18] -> [?,25,1] where summing over all 25 candidates yields 1)
        weights = Convolution1D(x.shape.as_list()[2], 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(x)
        weights = Dropout(dropoutRate)(weights)
        weights = Convolution1D(1, 1, kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(weights)
        
        #ensures that weights are not producing 25 times the same tensor
        computed_weights.append(weights)
        for w in computed_weights[:-1]:
            weights = K.subtract(weights,w)
            
        #Note: keras backend has no div op or multiply op :-(
        '''
        #kind of softmax over candidates dimension
        weights = K.exp(weights)
        weights_sum = K.tile(K.sum(weights,axis=1,keepdims=True),[1,x.shape.as_list()[1],1])
        norm_weights = K.div(weights,weights_sum)
        '''
        norm_weights = Activation('relu')(weights)
        #expand weights over features for multiplication (e.g. [?,25,1] -> [?,25,18])
        norm_weights = K.tile(norm_weights,[1,1,x.shape.as_list()[2]])
        #weighted sum over candidates (e.g. [?,25,18] -> [?,18]
        entry = K.sum(K.prod(x,norm_weights),axis=1)
        entries.append(entry)
    #repeat this per candidate and stack resulting tensors (e.g. 25x [?,18] -> [?,25,18])
    sorted_x = K.stack(entries,axis=1)
    #print sorted_x
    
    #print x.shape.as_list()[2]/2
    #now go nuts and do convolutions with filter size 2 (or more) to catch correlations
    sorted_x = Convolution1D(x.shape.as_list()[2]*2, 2,padding='same', kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(sorted_x)
    sorted_x = Dropout(dropoutRate)(sorted_x)          
    sorted_x = Convolution1D(x.shape.as_list()[2], 2,padding='same', kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(sorted_x)
    sorted_x = Dropout(dropoutRate)(sorted_x)
    sorted_x = Convolution1D(x.shape.as_list()[2], 2,padding='same', kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(sorted_x)
    sorted_x = Dropout(dropoutRate)(sorted_x)                 
    sorted_x = Convolution1D(int(math.sqrt(2.*x.shape.as_list()[2])), 2,padding='same', kernel_initializer='lecun_uniform',  activation='relu',kernel_regularizer=regularizers.l2(l2))(sorted_x)
    sorted_x = Dropout(dropoutRate)(sorted_x)        
    
    print sorted_x
    return sorted_x

def block_deepFlavourDense(x,dropoutRate,active=True,l2=0.00001,batchnorm=False,batchmomentum=0.6,add_summary=False):
    with tf.name_scope('dense'):
        if active:
            x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(l2))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
        else:
            x= Dense(1,kernel_initializer='zeros',trainable=False)(x)
        
        return x

def model_deepFlavourReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6,l2=0.00001,batchnorm=False,lstm=False,add_summary=False):
    #deep flavor w/o pt regression
    
    
    if (batchnorm):
        globalvars = BatchNormalization(momentum=momentum)(Inputs[0])
        cpf    =     BatchNormalization(momentum=momentum)(Inputs[1])
        npf    =     BatchNormalization(momentum=momentum)(Inputs[2])
        vtx    =     BatchNormalization(momentum=momentum)(Inputs[3])
    else:
        globalvars = Inputs[0]
        cpf    =     Inputs[1]
        npf    =     Inputs[2]
        vtx    =     Inputs[3]
        
    #sortedConv(vtx,dropoutRate,active=True,l2=l2,batchnorm=batchnorm,batchmomentum=0.6,add_summary=add_summary)
        
    if add_summary:
        tf.summary.histogram('input_globals', tf.reduce_mean(globalvars,axis=1))
        tf.summary.histogram('input_cpf', tf.reduce_mean(cpf,axis=1))
        tf.summary.histogram('input_npf', tf.reduce_mean(npf,axis=1))
        tf.summary.histogram('input_vtx', tf.reduce_mean(vtx,axis=1))
        
    #gen    =     Inputs[4]
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])

    '''
    cpf,npf,vtx = block_deepFlavourConvolutions(cpf,
                                                npf,
                                                vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                l2=l2,
                                                batchnorm=batchnorm, batchmomentum=momentum,add_summary=add_summary)
    '''
    cpf = sortedConv(cpf,dropoutRate,active=True,l2=l2,batchnorm=batchnorm,batchmomentum=0.6,add_summary=add_summary)
    npf = sortedConv(npf,dropoutRate,active=True,l2=l2,batchnorm=batchnorm,batchmomentum=0.6,add_summary=add_summary)
    vtx = sortedConv(vtx,dropoutRate,active=True,l2=l2,batchnorm=batchnorm,batchmomentum=0.6,add_summary=add_summary)
    
    with tf.name_scope('lstm'):
        if lstm:
            cpf  = LSTM(150,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate,
                activation='relu',
                recurrent_activation='relu'
            )(cpf)
            if batchnorm:
                cpf=BatchNormalization(momentum=momentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)

            npf = LSTM(50,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate,
                activation='relu',
                recurrent_activation='relu'
            )(npf)
            if batchnorm:
                npf=BatchNormalization(momentum=momentum)(npf)
            npf = Dropout(dropoutRate)(npf)

            vtx = LSTM(50,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate,
                activation='relu',
                recurrent_activation='relu'
            )(vtx)
            if batchnorm:
                vtx=BatchNormalization(momentum=momentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
        else:
            cpf = Flatten()(cpf)
            npf = Flatten()(npf)
            vtx = Flatten()(vtx)
    if add_summary:
        tf.summary.histogram('lstm_cpf', cpf)
        tf.summary.histogram('lstm_npf', npf)
        tf.summary.histogram('lstm_vtx', vtx)
        
        

    x = Concatenate()( [globalvars,cpf,npf,vtx])

    flavour_pred = block_deepFlavourDense(x,dropoutRate,active=True,l2=l2,batchnorm=batchnorm,batchmomentum=momentum,add_summary=add_summary)
    flavour_pred = Dense(nclasses, activation=None,kernel_initializer='lecun_uniform')(flavour_pred)
    #reg = Concatenate()( [flavour_pred, globalvars[:,1:1] ] ) 
    #reg_pred=Dense(2, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)

    #ptAndUnc = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    #ptAndUnc = Dense(2, activation='relu',kernel_initializer='ones',name='ID_pred')(ptAndUnc)

    return flavour_pred
    

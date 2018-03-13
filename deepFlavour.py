from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K

def block_deepFlavourConvolutions(cpf_input,npf_input,vtx_input,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6,add_summary=False):
    with tf.name_scope('cpf_conv'):
        if active:
            cpf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf_input)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf,training=K.learning_phase())
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf,training=K.learning_phase())
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf,training=K.learning_phase())
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            #if batchnorm:
            #    cpf = BatchNormalization(momentum=batchmomentum)(cpf,training=K.learning_phase())
            #cpf = Dropout(dropoutRate)(cpf)         
        else:
            cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf_input)
        
    with tf.name_scope('npf_conv'):
        if active:
            npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf_input)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf,training=K.learning_phase())
            npf = Dropout(dropoutRate)(npf) 
            npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf,training=K.learning_phase())
            npf = Dropout(dropoutRate)(npf)
            npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
            #if batchnorm:
            #    npf = BatchNormalization(momentum=batchmomentum)(npf,training=K.learning_phase())
            #npf = Dropout(dropoutRate)(npf)
        else:
            npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf_input)

    with tf.name_scope('vtx_conv'):
        if active:
            vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx_input)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx,training=K.learning_phase())
            vtx = Dropout(dropoutRate)(vtx) 
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx,training=K.learning_phase())
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx,training=K.learning_phase())
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            #vtx = BatchNormalization(momentum=batchmomentum)(vtx,training=K.learning_phase())
            #vtx = Dropout(dropoutRate)(vtx)
        else:
            vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx_input)
    if add_summary:
        tf.summary.histogram('conv_cpf', tf.reduce_mean(cpf,axis=1))
        tf.summary.histogram('conv_npf', tf.reduce_mean(npf,axis=1))
        tf.summary.histogram('conv_vtx', tf.reduce_mean(vtx,axis=1))
        
    return cpf,npf,vtx

def block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6,add_summary=False):
    with tf.name_scope('dense'):
        if active:
            x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x,training=K.learning_phase())
            x = Dropout(dropoutRate)(x)
        else:
            x= Dense(1,kernel_initializer='zeros',trainable=False)(x,training=K.learning_phase())
        
        return x

def model_deepFlavourReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6,batchnorm=False,lstm=False,add_summary=False):
    #deep flavor w/o pt regression
    
    
    if (batchnorm):
        globalvars = BatchNormalization(momentum=momentum)(Inputs[0],training=K.learning_phase())
        cpf    =     BatchNormalization(momentum=momentum)(Inputs[1],training=K.learning_phase())
        npf    =     BatchNormalization(momentum=momentum)(Inputs[2],training=K.learning_phase())
        vtx    =     BatchNormalization(momentum=momentum)(Inputs[3],training=K.learning_phase())
    else:
        globalvars = Inputs[0]
        cpf    =     Inputs[1]
        npf    =     Inputs[2]
        vtx    =     Inputs[3]
        
    if add_summary:
        tf.summary.histogram('input_globals', tf.reduce_mean(globalvars,axis=1))
        tf.summary.histogram('input_cpf', tf.reduce_mean(cpf,axis=1))
        tf.summary.histogram('input_npf', tf.reduce_mean(npf,axis=1))
        tf.summary.histogram('input_vtx', tf.reduce_mean(vtx,axis=1))
        
    #gen    =     Inputs[4]
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])

    cpf,npf,vtx = block_deepFlavourConvolutions(cpf,
                                                npf,
                                                vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=batchnorm, batchmomentum=momentum,add_summary=add_summary)



    with tf.name_scope('lstm'):
        if lstm:
            cpf  = LSTM(150,
                go_backwards=True,
                implementation=2,
                #dropout=dropoutRate,
                #recurrent_dropout=dropoutRate**1.5,
                #activation='relu',
                #recurrent_activation='relu'
            )(cpf)
            cpf=BatchNormalization(momentum=momentum)(cpf,training=K.learning_phase())
            cpf = Dropout(dropoutRate)(cpf)

            npf = LSTM(50,
                go_backwards=True,
                implementation=2,
                #dropout=dropoutRate,
                #recurrent_dropout=dropoutRate**1.5,
                #activation='relu',
                #recurrent_activation='relu'
            )(npf)
            npf=BatchNormalization(momentum=momentum)(npf,training=K.learning_phase())
            npf = Dropout(dropoutRate)(npf)

            vtx = LSTM(50,
                go_backwards=True,
                implementation=2,
                #dropout=dropoutRate,
                #recurrent_dropout=dropoutRate**1.5,
                #activation='relu',
                #recurrent_activation='relu'
            )(vtx)
            vtx=BatchNormalization(momentum=momentum)(vtx,training=K.learning_phase())
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

    flavour_pred = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=batchnorm,batchmomentum=momentum,add_summary=add_summary)
    flavour_pred = Dense(nclasses, activation=None,kernel_initializer='lecun_uniform',name='ID_pred')(flavour_pred)
    #reg = Concatenate()( [flavour_pred, globalvars[:,1:1] ] ) 
    #reg_pred=Dense(2, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)

    #ptAndUnc = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    #ptAndUnc = Dense(2, activation='relu',kernel_initializer='ones',name='ID_pred')(ptAndUnc)

    return flavour_pred
    

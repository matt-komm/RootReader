from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

def block_deepFlavourConvolutions(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    with tf.name_scope('cpf_conv'):
        cpf=charged
        if active:
            cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
            if batchnorm:
                cpf = BatchNormalization(momentum=batchmomentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)                                                   
            cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        else:
            cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    with tf.name_scope('npf_conv'):
        npf=neutrals
        if active:
            npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf)
            npf = Dropout(dropoutRate)(npf) 
            npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
            if batchnorm:
                npf = BatchNormalization(momentum=batchmomentum)(npf)
            npf = Dropout(dropoutRate)(npf)
            npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        else:
            npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    with tf.name_scope('vtx_conv'):
        vtx = vertices
        if active:
            vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx) 
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
            if batchnorm:
                vtx = BatchNormalization(momentum=batchmomentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
            vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        else:
            vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    with tf.name_scope('dense'):
        if active:
            x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', )(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
            x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum)(x)
            x = Dropout(dropoutRate)(x)
        else:
            x= Dense(1,kernel_initializer='zeros',trainable=False)(x)
        
        return x

def model_deepFlavourReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6,batchnorm=False,lstm=False):
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
        
    #gen    =     Inputs[4]
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])

    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=batchnorm, batchmomentum=momentum)



    with tf.name_scope('lstm'):
        if lstm:
            cpf  = LSTM(150,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate**1.5,
                activation='relu',
                recurrent_activation='relu'
            )(cpf)
            cpf=BatchNormalization(momentum=momentum)(cpf)
            cpf = Dropout(dropoutRate)(cpf)

            npf = LSTM(50,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate**1.5,
                activation='relu',
                recurrent_activation='relu'
            )(npf)
            npf=BatchNormalization(momentum=momentum)(npf)
            npf = Dropout(dropoutRate)(npf)

            vtx = LSTM(50,
                go_backwards=True,
                implementation=2,
                dropout=dropoutRate,
                recurrent_dropout=dropoutRate**1.5,
                activation='relu',
                recurrent_activation='relu'
            )(vtx)
            vtx=BatchNormalization(momentum=momentum)(vtx)
            vtx = Dropout(dropoutRate)(vtx)
        else:
            cpf = Flatten()(cpf)
            npf = Flatten()(npf)
            vtx = Flatten()(vtx)

    x = Concatenate()( [globalvars,cpf,npf,vtx])

    flavour_pred = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    flavour_pred = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(flavour_pred)
    
    #reg = Concatenate()( [flavour_pred, globalvars[:,1:1] ] ) 
    #reg_pred=Dense(2, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)

    #ptAndUnc = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    #ptAndUnc = Dense(2, activation='relu',kernel_initializer='ones',name='ID_pred')(ptAndUnc)

    return flavour_pred
    

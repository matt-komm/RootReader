import keras
import tensorflow as tf
from keras import backend as K
import os
import sys

def conv(x,filters,size,stride,options={}):
    x = keras.layers.Conv1D(
        filters, 
        size, 
        strides=stride, 
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(10**(-8)),
    )(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    return x
    
def lstm_stack(x,units,reverse=True,options={}):
    
    #run test/evaluation through non GPU version
    if os.environ.has_key('CUDA_VISIBLE_DEVICES') and options.has_key("GPULSTM") and options["GPULSTM"]:
        x  = keras.layers.CuDNNLSTM(units,
            go_backwards=reverse,
            #implementation=2,
            return_sequences=True,
            #recurrent_dropout=0.05,
            #activation='tanh',
            #recurrent_activation='tanh'
        )(x)
    else:
        x  = keras.layers.LSTM(units,
            go_backwards=reverse,
            implementation=2,
            return_sequences=True,
            recurrent_dropout=0.05,  #not possible with CuDNNLSTM
            activation='tanh',  #same as CuDNNLSTM
            recurrent_activation='sigmoid'  #same as CuDNNLSTM
        )(x)
    x = keras.layers.Dropout(0.1)(x)
    return x
    
def lstm(x,units,reverse=True,options={}):
    
    #run test/evaluation through non GPU version
    if os.environ.has_key('CUDA_VISIBLE_DEVICES') and options.has_key("GPULSTM") and options["GPULSTM"]:
        x  = keras.layers.CuDNNLSTM(units,
            go_backwards=reverse,
            #implementation=2,
            #recurrent_dropout=0.05,
            #activation='tanh',
            #recurrent_activation='tanh'
        )(x)
    else:
    
        x  = keras.layers.LSTM(units,
            go_backwards=reverse,
            implementation=2,
            recurrent_dropout=0.05, #not possible with CuDNNLSTM
            activation='tanh', #same as CuDNNLSTM
            recurrent_activation='sigmoid'  #same as CuDNNLSTM
        )(x)
    x = keras.layers.Dropout(0.1)(x)
    return x
    
def dense(x,nodes,dropout=0.1,options={}):
    x= keras.layers.Dense(
        nodes,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(10**(-8)),
    )(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    return x
    
def predict(x,nodes,name=None,options={}):
    x= keras.layers.Dense(
        nodes,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=keras.regularizers.l2(10**(-8)),
        activation='softmax',
        name=name
    )(x)
    return x
    
    
def model(globalvars,cpf,npf,sv,nclasses,options={}):
    def stacking(v):
        x = v[0]
        g = v[1]
        gx = tf.tile(g,[1,x.shape.as_list()[1]])
        gx = tf.reshape(gx,[-1,x.shape.as_list()[1],g.shape.as_list()[1]])
        return tf.concat([x,gx],axis=2)
        
        
    g = dense(globalvars,32,options=options)
    g = dense(g,16,options=options)
    g = dense(g,16,options=options)
    g = dense(g,6,options=options)
        
    with tf.name_scope('stacking'):
        cpf_comb = keras.layers.Lambda(stacking)([cpf,g])
        npf_comb = keras.layers.Lambda(stacking)([npf,g])
        sv_comb = keras.layers.Lambda(stacking)([sv,g])
    
    with tf.name_scope('cpf_conv'):
        cpf_conv = conv(cpf_comb,64,1,1,options=options)
        cpf_conv = conv(cpf_conv,32,1,1,options=options)
        cpf_conv = conv(cpf_conv,32,1,1,options=options)
        cpf_conv = conv(cpf_conv,8,1,1,options=options)
        cpf_conv_pred = keras.layers.Flatten()(cpf_conv)
    
    with tf.name_scope('npf_conv'):
        npf_conv = conv(npf_comb,32,1,1,options=options)
        npf_conv = conv(npf_conv,16,1,1,options=options)
        npf_conv = conv(npf_conv,16,1,1,options=options)
        npf_conv = conv(npf_conv,4,1,1,options=options)
        npf_conv_pred = keras.layers.Flatten()(npf_conv)
        
    with tf.name_scope('sv_conv'):
        sv_conv = conv(sv_comb,32,1,1,options=options)
        sv_conv = conv(sv_conv,16,1,1,options=options)
        sv_conv = conv(sv_conv,16,1,1,options=options)
        sv_conv = conv(sv_conv,8,1,1,options=options)
        sv_conv_pred = keras.layers.Flatten()(sv_conv)
        
    conv_prediction = keras.layers.Concatenate()([globalvars,cpf_conv_pred,npf_conv_pred,sv_conv_pred])
    conv_prediction = dense(conv_prediction,20,options=options)
    conv_prediction = dense(conv_prediction,20,options=options)
    conv_prediction = predict(conv_prediction,nclasses,name="conv_prediction",options=options)

    with tf.name_scope('lstm'):
        cpf_lstm1 = lstm_stack(cpf_conv,100,True,options=options) #8*25=200 inputs
        cpf_lstm2 = lstm(cpf_lstm1,70,False,options=options)
        npf_lstm1 = lstm_stack(npf_conv,50,True,options=options) #4*25=100 inputs
        npf_lstm2 = lstm(npf_lstm1,30,False,options=options)
        sv_lstm1 = lstm_stack(sv_conv,16,True,options=options) #8*4=32 inputs
        sv_lstm2 = lstm(sv_lstm1,10,False,options=options)
        
        cpf_lstm1_pred = keras.layers.Flatten()(cpf_lstm1)
        npf_lstm1_pred = keras.layers.Flatten()(npf_lstm1)
        sv_lstm1_pred = keras.layers.Flatten()(sv_lstm1)
        
    lstm1_prediction = keras.layers.Concatenate()([globalvars,cpf_lstm1_pred,npf_lstm1_pred,sv_lstm1_pred])
    lstm1_prediction = dense(lstm1_prediction,20,options=options)
    lstm1_prediction = dense(lstm1_prediction,20,options=options)
    lstm1_prediction = predict(lstm1_prediction,nclasses,name="lstm_prediction",options=options)
    
    with tf.name_scope('leak'):
        cpf_leak = keras.layers.Lambda(lambda v: v[:,0:5,:])(cpf)
        cpf_leak = keras.layers.Flatten()(cpf_leak) #5*16=80
        cpf_leak = dense(cpf_leak,100,options=options)
        cpf_leak = dense(cpf_leak,100,options=options)
        cpf_leak = dense(cpf_leak,60,options=options)
        
        npf_leak = keras.layers.Lambda(lambda v: v[:,0:5,:])(npf)
        npf_leak = keras.layers.Flatten()(npf_leak) #5*6=30
        npf_leak = dense(npf_leak,50,options=options)
        npf_leak = dense(npf_leak,50,options=options)
        npf_leak = dense(npf_leak,20,options=options)
        
        sv_leak = keras.layers.Lambda(lambda v: v[:,0:1,:])(sv)
        sv_leak = keras.layers.Flatten()(sv_leak) #1*12=12
        sv_leak = dense(sv_leak,16,options=options)
        sv_leak = dense(sv_leak,16,options=options)
        sv_leak = dense(sv_leak,8,options=options)
    
    full_prediction = keras.layers.Concatenate()([
        globalvars,
        cpf_lstm2,npf_lstm2,sv_lstm2,
        cpf_leak,npf_leak,sv_leak
    ])

    full_prediction = dense(full_prediction,200,options=options)
    full_prediction = dense(full_prediction,100,options=options)
    full_prediction = dense(full_prediction,100,options=options)
    full_prediction = dense(full_prediction,100,options=options)
    full_prediction = dense(full_prediction,100,options=options)
    full_prediction = predict(full_prediction,nclasses,name="full_prediction",options=options)
    
    return conv_prediction,lstm1_prediction,full_prediction
    

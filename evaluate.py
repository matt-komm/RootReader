import tensorflow as tf
import keras
from keras import backend as K
import os
import time
from root_reader import root_reader
from root_writer import root_writer

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass

from deepFlavour import model_deepFlavourReference


fileList = []

filePath = "/vols/cms/mkomm/LLP/samples/rootFiles.txt"

f = open(filePath)
for l in f:
    absPath = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileList.append(absPath)
f.close()
print len(fileList)

fileList = fileList[:10]

print fileList

featureDict = {

     "sv" : {
        "branches":[
            'sv_pt',
            'sv_deltaR',
            'sv_mass',
            'sv_ntracks',
            'sv_chi2',
            'sv_normchi2',
            'sv_dxy',
            'sv_dxysig',
            'sv_d3d',
            'sv_d3dsig',
            'sv_costhetasvpv',
            'sv_enratio',
            
        ],
        "multiplicity":"n_sv",
        "max":4
    },

    "truth": {
        "branches":[
            'isB/UInt_t',
            'isBB/UInt_t',
            'isGBB/UInt_t',
            'isLeptonicB/UInt_t',
            'isLeptonicB_C/UInt_t',
            'isC/UInt_t',
            'isCC/UInt_t',
            'isGCC/UInt_t',
            'isUD/UInt_t',
            'isS/UInt_t',
            'isG/UInt_t',
            'isUndefined/UInt_t',
            'isFromLLgno_isB/UInt_t',
            'isFromLLgno_isBB/UInt_t',
            'isFromLLgno_isGBB/UInt_t',
            'isFromLLgno_isLeptonicB/UInt_t',
            'isFromLLgno_isLeptonicB_C/UInt_t',
            'isFromLLgno_isC/UInt_t',
            'isFromLLgno_isCC/UInt_t',
            'isFromLLgno_isGCC/UInt_t',
            'isFromLLgno_isUD/UInt_t',
            'isFromLLgno_isS/UInt_t',
            'isFromLLgno_isG/UInt_t',
            'isFromLLgno_isUndefined/UInt_t'
        ],
        "multiplicity":None
    },
    "global": {
        "branches": [
            'jet_pt',
            'jet_eta',
            'nCpfcand',
            'nNpfcand',
            'nsv',
            'npv',
            'TagVarCSV_trackSumJetEtRatio', 
            'TagVarCSV_trackSumJetDeltaR', 
            'TagVarCSV_vertexCategory', 
            'TagVarCSV_trackSip2dValAboveCharm', 
            'TagVarCSV_trackSip2dSigAboveCharm', 
            'TagVarCSV_trackSip3dValAboveCharm', 
            'TagVarCSV_trackSip3dSigAboveCharm', 
            'TagVarCSV_jetNSelectedTracks', 
            'TagVarCSV_jetNTracksEtaRel'
        ],
        "multiplicity":None

    },


    "Cpfcan": {
        "branches": [
            'Cpfcan_BtagPf_trackEtaRel',
            'Cpfcan_BtagPf_trackPtRel',
            'Cpfcan_BtagPf_trackPPar',
            'Cpfcan_BtagPf_trackDeltaR',
            'Cpfcan_BtagPf_trackPParRatio',
            'Cpfcan_BtagPf_trackSip2dVal',
            'Cpfcan_BtagPf_trackSip2dSig',
            'Cpfcan_BtagPf_trackSip3dVal',
            'Cpfcan_BtagPf_trackSip3dSig',
            'Cpfcan_BtagPf_trackJetDistVal',

            'Cpfcan_ptrel', 
            'Cpfcan_drminsv',
            'Cpfcan_VTX_ass',
            'Cpfcan_puppiw',
            'Cpfcan_chi2',
            'Cpfcan_quality'
        ],
        "multiplicity":"n_Cpfcand",
        "max":25
    },
    "Npfcan": {
        "branches": [
            'Npfcan_ptrel',
            'Npfcan_deltaR',
            'Npfcan_isGamma',
            'Npfcan_HadFrac',
            'Npfcan_drminsv',
            'Npfcan_puppiw'
        ],
        "multiplicity":"n_Npfcand",
        "max":25
    }
}

loss_mean = []

for f in fileList:
    fileListQueue = tf.train.string_input_producer([f], num_epochs=1, shuffle=False)

    rootreader_op = root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=1).batch()
    
    globalvars = keras.layers.Input(tensor=rootreader_op['global'])
    cpf = keras.layers.Input(tensor=rootreader_op['Cpfcan'])
    npf = keras.layers.Input(tensor=rootreader_op['Npfcan'])
    vtx = keras.layers.Input(tensor=rootreader_op['sv'])
    #gen = keras.layers.Input(tensor=tf.constant(0.,shape=[1,1]),shape=(1,))
    truth = rootreader_op["truth"]
    


    nclasses = truth.shape.as_list()[1]
    print nclasses
    inputs = [globalvars,cpf,npf,vtx]
    prediction = model_deepFlavourReference(inputs,nclasses,1,dropoutRate=0.1,momentum=0.6)
    loss = tf.reduce_sum(tf.square(keras.losses.categorical_crossentropy(truth, prediction)))
    accuracy,accuracy_op = tf.metrics.accuracy(tf.argmax(truth,1),tf.argmax(prediction,1))
    model = keras.Model(inputs=inputs, outputs=prediction)

    rootwriter_op = root_writer(prediction,featureDict["truth"]["branches"],"deepntuplizer/tree","out.root").write()
    #init_op = tf.global_variables_initializer() #bug https://github.com/tensorflow/tensorflow/issues/1045
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_loss = 0
    
    print "loading weights ..."
    model.load_weights("model_epoch9.hdf5") #use after init_op which initializes random weights!!!
    
    try:
        step = 0
        while not coord.should_stop():
            start_time = time.time()

            loss_value,accuracy_value,_= sess.run([loss,accuracy_op,rootwriter_op], feed_dict={K.learning_phase(): 0}) #pass 1 for training, 0 for testing
            #print result
            #data = sess.run(trainingBatch)
            duration = time.time() - start_time
            if step % 100 == 0:
                print 'Step %d: loss = %.2f, accuracy=%.1f%% (%.3f sec)' % (step, loss_value,accuracy_value*100.,duration)
            step += 1
            '''
            if step>10:
                break
            '''
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))

    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    break
    
    
    
    

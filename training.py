import tensorflow as tf
import keras
from keras import backend as K
import os
import time
from root_reader import root_reader

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from deepFlavour import model_deepFlavourReference

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass



fileList = []

filePath = "/vols/cms/mkomm/LLP/samples/rootFiles.txt"

f = open(filePath)
for l in f:
    absPath = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileList.append(absPath)
f.close()
print "files ",len(fileList)

fileList = fileList[:20]

#print fileList

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
    "gen": {
        "branches":[
            'genLL_decayLength'
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

for epoch in range(60):
    epoch_duration = time.time()
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=True)

    rootreader_op = [
        root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=100).batch() for _ in range(4)
    ]
    
    batchSize = 10000
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3*batchSize
    
    #check: tf.contrib.training.stratified_sample
    #for online resampling for equal pt/eta weights
    #trainingBatch = tf.train.batch_join(
    trainingBatch = tf.train.shuffle_batch_join(
        rootreader_op, 
        batch_size=batchSize, 
        capacity=capacity,
        min_after_dequeue=minAfterDequeue,
        enqueue_many=True #requires to read examples in batches!
    )

    globalvars = keras.layers.Input(tensor=trainingBatch['global'])
    cpf = keras.layers.Input(tensor=trainingBatch['Cpfcan'])
    npf = keras.layers.Input(tensor=trainingBatch['Npfcan'])
    vtx = keras.layers.Input(tensor=trainingBatch['sv'])
    #gen = keras.layers.Input(tensor=trainingBatch['gen'])
    truth = trainingBatch["truth"]
    #dequeueBatch = trainingBatch['Npfcan'].dequeue()

    nclasses = truth.shape.as_list()[1]
    inputs = [globalvars,cpf,npf,vtx]
    prediction = model_deepFlavourReference(inputs,nclasses,1,dropoutRate=0.1,momentum=0.6)
    loss = tf.reduce_sum(keras.losses.categorical_crossentropy(truth, prediction))
    accuracy,accuracy_op = tf.metrics.accuracy(tf.argmax(truth,1),tf.argmax(prediction,1))
    model = keras.Model(inputs=inputs, outputs=prediction)
    
    #model.add_loss(loss)
    #model.compile(optimizer='rmsprop', loss=None)
    #model.summary()
    #train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_op = tf.train.AdamOptimizer(
        learning_rate=0.0001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=True,
        name='Adam'
    ).minimize(
        loss
    )


    #init_op = tf.global_variables_initializer() #bug https://github.com/tensorflow/tensorflow/issues/1045
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_loss = 0
    
    if os.path.exists("model_epoch"+str(epoch-1)+".hdf5"):
        print "loading weights ... model_epoch"+str(epoch-1)+".hdf5"
        model.load_weights("model_epoch"+str(epoch-1)+".hdf5") #use after init_op which initializes random weights!!!
    elif epoch>0:
        print "no weights from previous epoch found"
        sys.exit(1)
        
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            start_time = time.time()

            _, loss_value, accuracy_value = sess.run([train_op, loss,accuracy_op], feed_dict={K.learning_phase(): 1}) #pass 1 for training, 0 for testing
            total_loss+=loss_value
            
            #data = sess.run(trainingBatch)
            #print data
            duration = time.time() - start_time
            if step % 10 == 0:
                print 'Step %d: loss = %.2f, accuracy = %.1f%% (%.3f sec)' % (step, loss_value,accuracy_value*100.,duration)
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
    model.save_weights("model_epoch"+str(epoch)+".hdf5")
    print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
    print "Average loss = ",(total_loss/step)
    loss_mean.append(total_loss/step)
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
        
for i,l in enumerate(loss_mean):
    print i+1,l
    
    
    
    

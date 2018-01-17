import tensorflow as tf
import keras
from keras import backend as K
import os
import numpy
import ROOT
import time
from root_reader import root_reader

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from deepFlavour import model_deepFlavourReference

classificationweights_module = tf.load_op_library('./libClassificationWeights.so')

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass



fileListTrain = []
filePathTrain = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples/rootFiles.raw.txt"
f = open(filePathTrain)
for l in f:
    absPath = os.path.join(filePathTrain.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileListTrain.append(absPath)
f.close()
print "files train ",len(fileListTrain)


    

#fileListTrain = fileListTrain[:6]

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
    },
    
    "gen": {
        "branches":[
            'genLL_decayLength'
        ],
    },
    
    "globals": {
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
        "max":25
    }
}



histsPerClass = {}
weightsPerClass = {}
chain = ROOT.TChain("deepntuplizer/tree")
for f in fileListTrain:
    chain.AddFile(f)

binning = numpy.logspace(1.5,3,num=20)
targetShape = ROOT.TH1F("ptTarget","",len(binning)-1,binning)
for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    print "projecting ... ",branchName
    hist = ROOT.TH1F("pt"+branchName,"",len(binning)-1,binning)
    hist.Sumw2()
    #hist.SetDirectory(0)
    chain.Project(hist.GetName(),"jet_pt","("+branchName+"==1)")
    if hist.Integral()>0:
        hist.Scale(1./hist.Integral())
    else:
        print "no entries found for class: ",branchName
        
    if branchName.find("isFromLLgno")==0:
        targetShape.Add(hist,0.1) #lower impact of LLP
    if branchName.find("isB")==0 or branchName.find("isBB")==0:
        targetShape.Add(hist)
    
    histsPerClass[branchName]=hist
targetShape.Scale(1./targetShape.Integral())

for label in histsPerClass.keys():
    hist = histsPerClass[label]
    if (hist.Integral()>0):
        weight = targetShape.Clone("weight_"+label)
        weight.Scale(1) #can use arbitrary scale here to make weight more reasonable
        weight.Divide(hist)
        weightsPerClass[label]=weight
    else:
        weightsPerClass[label]=hist

cv = ROOT.TCanvas("cv","",800,600)
cv.SetLogx(1)
ymax = max(map(lambda h: h.GetMaximum(),histsPerClass.values()))
axis = ROOT.TH2F("axis",";pt;",50,binning[0],binning[-1],50,0,ymax*1.1)
axis.Draw("AXIS")
targetShape.SetLineWidth(3)
targetShape.SetLineColor(ROOT.kRed)
targetShape.Draw("SameHISTL")
for label in histsPerClass.keys():
    histsPerClass[label].Draw("SameHISTL")
cv.Update()
cv.Print("pt.pdf")

weightFile = ROOT.TFile("weights.root","RECREATE")
cvWeight = ROOT.TCanvas("cv2","",800,600)
cvWeight.SetLogy(1)
cvWeight.SetLogx(1)
axisvWeight = ROOT.TH2F("axis2",";pt;",50,binning[0],binning[-1],50,0.1,150)
axisvWeight.Draw("AXIS")
histNames = []
for label in weightsPerClass.keys():
    weightsPerClass[label].Draw("SameHISTL")
    weightsPerClass[label].Write()
    histNames.append(weightsPerClass[label].GetName())
    
cvWeight.Update()
cvWeight.Print("pt_weight.pdf")
weightFile.Close()


for epoch in range(40):
    epoch_duration = time.time()
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileListTrain, num_epochs=1, shuffle=True)

    #TODO: split 10% off inside root_reader for online testing
    rootreader_op = [
        root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=100).batch() for _ in range(6)
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

    globalvars = keras.layers.Input(tensor=trainingBatch['globals'])
    cpf = keras.layers.Input(tensor=trainingBatch['Cpfcan'])
    npf = keras.layers.Input(tensor=trainingBatch['Npfcan'])
    vtx = keras.layers.Input(tensor=trainingBatch['sv'])
    #gen = keras.layers.Input(tensor=trainingBatch['gen'])
    truth = trainingBatch["truth"]
    #dequeueBatch = trainingBatch['Npfcan'].dequeue()
    
    weights = classificationweights_module.classification_weights(
        trainingBatch["truth"],
        trainingBatch["globals"],
        "weights.root",
        histNames,
        0
    )

    nclasses = truth.shape.as_list()[1]
    inputs = [globalvars,cpf,npf,vtx]
    prediction = model_deepFlavourReference(inputs,nclasses,1,dropoutRate=0.1,momentum=0.6)
    loss = tf.reduce_mean(tf.multiply(keras.losses.categorical_crossentropy(truth, prediction),weights))
    loss_old = tf.reduce_mean(keras.losses.categorical_crossentropy(truth, prediction))
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

            _, loss_value, loss_old_value, accuracy_value = sess.run([train_op, loss,loss_old,accuracy_op], feed_dict={K.learning_phase(): 1}) #pass 1 for training, 0 for testing
            total_loss+=loss_value
            #data = sess.run(trainingBatch)
            #print data
            duration = time.time() - start_time
            if step % 1 == 0:
                print 'Step %d: loss = %.2f (%.2f), accuracy = %.1f%% (%.3f sec)' % (step, loss_value,loss_old_value,accuracy_value*100.,duration)
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
    model.save_weights("model_epoch"+str(epoch)+".hdf5")
    print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
    print "Average loss = ",(total_loss/step)
    f = open("model_epoch.stat","a")
    f.write(str(epoch)+";"+str(total_loss/step)+";"+str(accuracy_value*100.)+"\n")
    f.close()
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    
    
    
    

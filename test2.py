import tensorflow as tf
import os
import time
import ROOT
import sys
import numpy
from root_reader import root_reader
from train_test_splitter import train_test_splitter
from resampler import resampler

classificationweights_module = tf.load_op_library('./libClassificationWeights.so')

fileList = []

#filePath = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples/rootFiles.raw.txt"
filePath = "/vols/cms/mkomm/LLP/samples3_split_train_shuffle.txt"

f = open(filePath)
for l in f:
    absPath = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileList.append(absPath)
f.close()
print len(fileList) 
fileList = fileList[:6]
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
        "max":4
    },

    "truth": {
        "branches":[
            #'isB/UInt_t',
            #'isBB/UInt_t',
            #'isGBB/UInt_t',
            #'isLeptonicB/UInt_t',
            #'isLeptonicB_C/UInt_t',
            'isC/UInt_t',
            'isCC/UInt_t',
            'isGCC/UInt_t',
            'isUD/UInt_t',
            'isS/UInt_t',
            'isG/UInt_t',
            'isFromLLgno/UInt_t',
            #'isUndefined/UInt_t',
            #'isFromLLgno_isB/UInt_t',
            #'isFromLLgno_isBB/UInt_t',
            #'isFromLLgno_isGBB/UInt_t',
            #'isFromLLgno_isLeptonicB/UInt_t',
            #'isFromLLgno_isLeptonicB_C/UInt_t',
            #'isFromLLgno_isC/UInt_t',
            #'isFromLLgno_isCC/UInt_t',
            #'isFromLLgno_isGCC/UInt_t',
            #'isFromLLgno_isUD/UInt_t',
            #'isFromLLgno_isS/UInt_t',
            #'isFromLLgno_isG/UInt_t',
            #'isFromLLgno_isUndefined/UInt_t'
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

}
'''
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
'''

outputFolder = os.getcwd()

histsPerClass = {}
weightsPerClass = {}
chain = ROOT.TChain("deepntuplizer/tree")
for f in fileList:
    chain.AddFile(f)
nEntries = chain.GetEntries()
print "total entries",nEntries


binningPt = numpy.logspace(1.6,3,num=20)
binningEta = numpy.linspace(-2.4,2.4,num=10)
targetShape = ROOT.TH2F("ptetaTarget","",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
branchNameList = []
for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    branchNameList.append(branchName)
    print "projecting ... ",branchName
    hist = ROOT.TH2F("pteta"+branchName,"",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
    hist.Sumw2()
    #hist.SetDirectory(0)
    chain.Project(hist.GetName(),"jet_eta:jet_pt","("+branchName+"==1)")
    
    targetShape.Add(hist)
    if hist.Integral()>0:
        print " -> entries ",hist.GetEntries()
        hist.Scale(1./hist.Integral())
    else:
        print " -> no entries found for class: ",branchName
        
    
    
    histsPerClass[branchName]=hist
targetShape.Scale(1./targetShape.Integral())

for label in branchNameList:
    hist = histsPerClass[label]
    if (hist.Integral()>0):
        weight = targetShape.Clone(label)
        weight.Scale(1) #can use arbitrary scale here to make weight more reasonable
        weight.Divide(hist)
        weightsPerClass[label]=weight
    else:
        weight = targetShape.Clone(label)
        weight.Scale(0)
        weightsPerClass[label]=weight
        
weightFile = ROOT.TFile(os.path.join(outputFolder,"weights.root"),"RECREATE")
for l,h in weightsPerClass.items():
    print h
    h.Write()
weightFile.Close()


for epoch in range(1):
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=True)

    rootreader_op = []
    weights = []
    resamplers = []
    
    for _ in range(1):
        reader = root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=10).batch()
        rootreader_op.append(reader)

        weight = classificationweights_module.classification_weights(
            reader["truth"],
            reader["globals"],
            "weights.root",
            branchNameList,
            [0,1]
        )
        weights.append(weight)
        
        resampled = resampler(
            weight,
            reader
        ).resample()
        resamplers.append(resampled)
        
        
    
    batchSize = 10
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3 * batchSize
    
    #check: tf.contrib.training.stratified_sample
    #for online resampling for equal pt/eta weights
    trainingBatch = tf.train.batch_join(
    #trainingBatch = tf.train.shuffle_batch_join(
        resamplers, 
        batch_size=batchSize, 
        capacity=capacity,
        #min_after_dequeue=minAfterDequeue,
        enqueue_many=True #requires to read examples in batches!
    )
    #trainingBatch["num"]=tf.sign(tf.mod(trainingBatch["num"],tf.constant(10,shape=trainingBatch["num"].get_shape())))
    

    train_test_split = train_test_splitter(
        trainingBatch["num"],
        trainingBatch,
        percentage=20
    )

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    
    sess = tf.Session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    '''
    #print sess.run(dequeue_op)
    for _ in range(100000):
        sess.run(trainingBatch)

    '''
    jet_pt = tf.map_fn(lambda x: x[0], trainingBatch["globals"])
    
    
    steps = 0
    try:
        while(True):
            steps+=1
            t = time.time()
            #result = sess.run([trainingBatch,jet_pt,weights,tf.argmax(trainingBatch["truth"],axis=1)])
            result = sess.run([
                trainingBatch
            ])
            t = time.time()-t
            print "-- step %3i (%8.3fs) --"%(steps,t)
            
            print "train",result
            print
            '''
            if steps%1==0:
                print "-- step %3i (%8.3fs) --"%(steps,t)
                for i in range(len(result[1])):
                    print "%33s:  pt=%6.1f  w=%6.2e"%(histNames[result[3][i]],result[1][i],result[2][i])#,result[0]["globals"][i][0]
                print 
            '''
            
            if steps>20:
                break
            
            #print sess.run(dequeue_op)
    except tf.errors.OutOfRangeError:
        print "done"

    coord.request_stop()
    coord.join(threads)
    
    sess.close()


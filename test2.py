import tensorflow as tf
import os
import time
import ROOT
import sys
from root_reader import root_reader

fileList = []

filePath = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples/rootFiles.raw.txt"
#filePath = "/vols/cms/mkomm/LLP/samples/rootFiles.txt"

f = open(filePath)
for l in f:
    absPath = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileList.append(absPath)
f.close()
print len(fileList)

#fileList = fileList[:20]



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

histsPerClass = {}
weightsPerClass = {}
chain = ROOT.TChain("deepntuplizer/tree")
for f in fileList:
    chain.AddFile(f)

targetShape = ROOT.TH1F("ptTarget","",20,1,3.4)
for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    print "projecting ... ",branchName
    hist = ROOT.TH1F("pt"+branchName,"",20,1,3.4)
    hist.Sumw2()
    #hist.SetDirectory(0)
    chain.Project(hist.GetName(),"TMath::Log10(jet_pt)","("+branchName+"==1)")
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
        weight = targetShape.Clone("weight"+label)
        weight.Scale(0.1) #can use arbitrary scale here to make weight more reasonable
        weight.Divide(hist)
        weightsPerClass[label]=weight
    else:
        weightsPerClass[label]=hist

cv = ROOT.TCanvas("cv","",800,600)
ymax = max(map(lambda h: h.GetMaximum(),histsPerClass.values()))
axis = ROOT.TH2F("axis",";log10(pt);",20,1,3.4,50,0,ymax*1.1)
axis.Draw("AXIS")
targetShape.SetLineWidth(3)
targetShape.SetLineColor(ROOT.kRed)
targetShape.Draw("SameHISTL")
for label in histsPerClass.keys():
    histsPerClass[label].Draw("SameHISTL")
cv.Update()
cv.Print("pt.pdf")



for label in histsPerClass.keys():
    cvWeight = ROOT.TCanvas("cv2","",800,600)
    cvWeight.SetLogy(1)
    axisvWeight = ROOT.TH2F("axis",";log10(pt);",20,1,3.4,50,0.01,100)
    axisvWeight.Draw("AXIS")
    weightsPerClass[label].Draw("SameHISTL")
    fitFct = ROOT.TF1("fit"+label,"pol5",0,4)
    weightsPerClass[label].Fit(fitFct)
    fitFct.Draw("SameL")
    cvWeight.Update()
    cvWeight.Print("pt_weight_"+label+".pdf")
    
sys.exit(1)

for epoch in range(1):
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=False)

    rootreader_op = [
        root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=100).batch() for _ in range(1)
    ]
    
    batchSize = 2
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3 * batchSize
    
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
    #trainingBatch["num"]=tf.sign(tf.mod(trainingBatch["num"],tf.constant(10,shape=trainingBatch["num"].get_shape())))
    print trainingBatch
    '''
    weights = tf.constant([0.1,0.7],shape=[batchSize],dtype=tf.float32)
    print weights
    trainingBatchSampled,resampleRate = tf.contrib.training.weighted_resample(
        [
            trainingBatch["num"],
            trainingBatch["truth"]
        ],
        weights,
        1
    )
    trainingBatchSampledDict = {}
    trainingBatchSampledDict["num"]=trainingBatchSampled[0]
    trainingBatchSampledDict["truth"]=tf.argmax(trainingBatchSampled[1],axis=1)
    '''

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
    steps = 1
    try:
        while(True):
            t = time.time()
            result = sess.run([trainingBatch])
            t = time.time()-t
            print "-- step %3i (%8.3fs) --"%(steps,t)
            print result
            print 
            steps+=1
            if steps>10:
                break
            #print sess.run(dequeue_op)
    except tf.errors.OutOfRangeError:
        print "done"

    coord.request_stop()
    coord.join(threads)


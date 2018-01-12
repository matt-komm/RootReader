import tensorflow as tf
import os
import time
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

fileList = fileList[:20]

print fileList

featureDict = {
    "test": {
        "branches": [
            "nsv"
        ],
        "multiplicity":None
    },
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
    }
}
'''

        "truth": {
        "branches":[
            'isB/i',
            'isBB/i',
            'isGBB/i',
            'isLeptonicB/i',
            'isLeptonicB_C/i',
            'isC/i',
            'isCC/i',
            'isGCC/i',
            'isUD/i',
            'isS/i',
            'isG/i',
            'isUndefined/i',
            'isFromLLgno_isB/i',
            'isFromLLgno_isBB/i',
            'isFromLLgno_isGBB/i',
            'isFromLLgno_isLeptonicB/i',
            'isFromLLgno_isLeptonicB_C/i',
            'isFromLLgno_isC/i',
            'isFromLLgno_isCC/i',
            'isFromLLgno_isGCC/i',
            'isFromLLgno_isUD/i',
            'isFromLLgno_isS/i',
            'isFromLLgno_isG/i',
            'isFromLLgno_isUndefined/i'
        ],
        "multiplicity":None
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
'''

for epoch in range(1):
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=False)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    rootreader_op = [
        root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=100).batch() for _ in range(4)
    ]
    print rootreader_op
    
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
            result = sess.run(trainingBatch)
            print result
            #print len(result["raw"][0])
            t = time.time()-t
            print "step %3i (%8.3fs)"%(steps,t)
            steps+=1
            if steps>10:
                break
            #print sess.run(dequeue_op)
    except tf.errors.OutOfRangeError:
        print "done"

    coord.request_stop()
    coord.join(threads)


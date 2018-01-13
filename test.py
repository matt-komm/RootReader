import tensorflow as tf
import os

rootreader_module = tf.load_op_library('./libRootReader.so')

fileList = []

filePath = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples/rootFiles.raw.txt"
#filePath = "/vols/cms/mkomm/LLP/samples/rootFiles.txt"

f = open(filePath)
for l in f:
    absPath = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    fileList.append(absPath)
f.close()
print len(fileList)

fileList = fileList[:5]

print fileList

def makeReader(queue):
    batch, num = rootreader_module.root_reader(queue, 
    [
        "jet_pt",
        "jet_eta",
        "sv_pt[n_sv,2]",
        "sv_eta[n_sv,2]",
        #"genLL_decayLength"
    ],
    "deepntuplizer/tree",
    naninf=0)
    return batch

for epoch in range(1):
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=2, shuffle=True)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    dequeue_op = fileListQueue.dequeue()

    print fileListQueue.queue_ref

    rootreader_ops = [
        
          makeReader(fileListQueue.queue_ref)
         for _ in range(1)
    ]

    batchSize = 1
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3 * batchSize

    #trainingBatch = tf.train.batch_join(
    trainingBatch = tf.train.shuffle_batch_join(
        rootreader_ops, 
        batch_size=batchSize, 
        capacity=capacity,
        min_after_dequeue=minAfterDequeue,
        enqueue_many=False #requires to read examples in batches!
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
            
            print sess.run(trainingBatch)
            print steps
            steps+=1
            if steps>10:
                break
            #print sess.run(dequeue_op)
    except tf.errors.OutOfRangeError:
        print "done"

    coord.request_stop()
    coord.join(threads)


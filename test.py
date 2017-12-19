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

fileList = fileList[:10]

print fileList

fileListQueue = tf.train.string_input_producer(fileList, num_epochs=2, shuffle=True)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

dequeue_op = fileListQueue.dequeue()

print fileListQueue.queue_ref

rootreader_op = [
    [
        rootreader_module.root_reader(fileListQueue.queue_ref, branches=[
            "jet_pt",
            "jet_eta",
            "genLL_decayLength"
        ])
    ] for _ in range(4)
]

batchSize = 1000
minAfterDequeue = batchSize*2
capacity = minAfterDequeue + 3 * batchSize

#trainingBatch = tf.train.batch_join(
trainingBatch = tf.train.shuffle_batch_join(
    rootreader_op, 
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
try:
    while(True):
        pass
        sess.run(trainingBatch)
        #print sess.run(dequeue_op)
except tf.errors.OutOfRangeError:
    print "done"

coord.request_stop()
coord.join(threads)


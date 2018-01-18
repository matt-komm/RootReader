import tensorflow as tf

traintest_module = tf.load_op_library('./libTrainTestSplitting.so')

class train_test_splitter():       
    def __init__(self,
        nums,
        batch,
        percentage
    ):
        if type(batch)==type(dict()):
            self.trainBatches = {}
            self.testBatches = {}
            for name in sorted(batch.keys()):
                self.trainBatches[name],self.testBatches[name] = traintest_module.train_test_splitting(
                    nums,
                    batch[name],
                    percentage=percentage
                )
        elif type(batch)==type(list()):
            self.trainBatches = []
            self.testBatches = []
            for subBatch in batch.keys():
                trainBatch,testBatch = traintest_module.train_test_splitting(
                    nums,
                    subBatch,
                    percentage=percentage
                )
            self.trainBatches.append(trainBatch)
            self.testBatches.append(testBatch)
                
            
        
    def train(self):
        return self.trainBatches
        
    def test(self):
        return self.testBatches
            

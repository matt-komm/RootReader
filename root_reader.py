import tensorflow as tf

rootreader_module = tf.load_op_library('./libRootReader.so')

class root_reader():
    @staticmethod
    def slice_and_reshape(start,size,shape=None):
        if shape==None:
            return lambda tensor: rootreader_module.batched_transformation(tensor,start,size)
        else:
            #return lambda tensor: tf.map_fn(lambda x: tf.transpose(tf.reshape(tf.slice(x,[start],[size]),shape)),tensor,back_prop=False,infer_shape=True)
            return lambda tensor: rootreader_module.batched_transformation(tensor,start,size,shape=shape,transpose=True)
            
    def __init__(self,
        queue,
        feature_dict,
        treename,
        batch=1,
        naninf=0
    ):
        self._feature_dict = feature_dict
        
        self._branch_list = []
        self._output_formatters = {}
        index = 0
        for feature_name in sorted(self._feature_dict.keys()):
            
            feature_values = self._feature_dict[feature_name]
            if not feature_values.has_key("max"):
                self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                    index,
                    len(feature_values["branches"])
                )
                index+=len(feature_values["branches"])
                self._branch_list.extend(feature_values["branches"])
                
            else:
                self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                    index,
                    len(feature_values["branches"])*feature_values["max"],
                    [len(feature_values["branches"]),feature_values["max"]]
                )
                index+=len(feature_values["branches"])*feature_values["max"]
                for branch_name in feature_values["branches"]:
                    self._branch_list.append(
                        branch_name+"["+str(feature_values["max"])+"]"
                    )
                 
                
        self._op, self._num = rootreader_module.root_reader(
            queue.queue_ref, 
            self._branch_list,
            treename=treename,
            naninf=naninf, 
            batch=batch
        )
        
    def raw(self):
        return {"raw":self._op,"num":self._num}
        
    def batch(self):
        result = {}
        for featureName in sorted(self._output_formatters.keys()):
            result[featureName]=self._output_formatters[featureName](self._op)
        result["num"] = self._num
        return result
        
            

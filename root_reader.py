import tensorflow as tf

rootreader_module = tf.load_op_library('./libRootReader.so')

class root_reader():
    @staticmethod
    def slice_and_reshape(start,size,shape=None):
        if shape==None:
            return lambda tensor: tf.slice(tensor,[start],[size])
        else:
            return lambda tensor: tf.transpose(
                tf.reshape(tf.slice(tensor,[start],[size]),shape)
            )
            
    def __init__(self,
        queue,
        feature_dict,
        naninf=0
    ):
        self._feature_dict = feature_dict
        
        self._branch_list = []
        self._output_formatters = {}
        for feature_name in sorted(self._feature_dict.keys()):
            
            feature_values = self._feature_dict[feature_name]
            if feature_values["multiplicity"]==None:
                self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                    len(self._branch_list),
                    len(feature_values["branches"])
                )
                self._branch_list.extend(feature_values["branches"])
                
            else:
                for branch_name in feature_values["branches"]:
                    self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                        len(self._branch_list),
                        len(feature_values["branches"])*feature_values["max"],
                        #[feature_values["max"],len(feature_values["branches"])]
                    )
                    self._branch_list.append(
                        branch_name+
                        "["+
                        feature_values["multiplicity"]+
                        ","+
                        str(feature_values["max"])+
                        "]"
                    )
                    print self._branch_list[-1]
        self._op = rootreader_module.root_reader(
            queue.queue_ref, 
            naninf=naninf, 
            branches=self._branch_list
        )
        
    def batch(self):
        result = {}
        for featureName in sorted(self._output_formatters.keys()):
            result[featureName]=self._output_formatters[featureName](self._op)
        return result
        
            

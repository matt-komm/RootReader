import tensorflow as tf
import json
import pickle

class Model(object):
    def __init__(self,isTraining):
        self._isTraining = tf.constant(isTraining)
        self._layers = []
        
    def isTraining(self):
        return self._isTraining
        
    def addLayer(self,layer):
        self._layers.append(layer)
        
    def getLayers(self):
        return self._layers
        
    def getVariables(self):
        variables = []
        for layer in self._layers:
            variables.append({"weights":layer.variables})
        return variables
        
    def count_params(self):
        s = 0
        for layer in self._layers:
            s += layer.count_params()
        return s

    def saveVariables(self,filePath,session):
        variableList = self.evalVariables(session)
        output = []
        for ilayer in range(len(variableList)):
            layerDescription = {
                "name": self._layers[ilayer].name,
                "weights":[]
            }
            for iweight, weight in enumerate(variableList[ilayer]["weights"]):
                weightDescription = {
                    "name": self._layers[ilayer].variables[iweight].name,
                    "dtype":str(self._layers[ilayer].variables[iweight].dtype),
                    "shape":list(self._layers[ilayer].variables[iweight].shape.as_list()),
                    "data":pickle.dumps(weight, protocol=0)
                }
                layerDescription["weights"].append(weightDescription)
            output.append(layerDescription)
    
    
        f = open(filePath,"w")
        json.dump({"data":output},f,sort_keys=True, indent=2)
        f.close()
        
    def loadVariables(self,filePath,session):
        assign_ops = []
        
        variables = self.getVariables()
        data = json.load(open(filePath,"r"))["data"]
        if len(data)!=len(variables):
            raise Exception("mismatch in number of saved layers")
            
        for ilayer in range(len(variables)):
            if len(data[ilayer]["weights"])!=len(variables[ilayer]["weights"]):
                raise Exception("mismatch in number of weights",len(data[ilayer]["weights"])," vs ",len(variables[ilayer]["weights"]))
            for ivariable, variable in enumerate(variables[ilayer]["weights"]):
                variableData = pickle.loads(data[ilayer]["weights"][ivariable]["data"])
                #if variableData.dtype!=variable.dtype:
                #    raise Exception("mismatch in dtype of variable")
                assign_ops.append(tf.assign(variable,tf.constant(variableData)))
        session.run(assign_ops)
        
        
    def evalVariables(self,session):
        return session.run(self.getVariables())
        
        
    def assignVariablesFromModel(self,model):
        assign_ops = []
        for ilayer,layer in enumerate(model.getLayers()):
            for ivariable,variable in enumerate(layer.variables):
                if self._layers[ilayer].variables[ivariable].shape != variable.shape:
                    print "mismatching shapes: ",self._layers[ilayer].variables[ivariable].shape,variable.shape
                assign_ops.append(tf.assign(self._layers[ilayer].variables[ivariable],variable))
        return assign_ops
        

class Sort(tf.layers.Layer):
    def __init__(self):
        tf.layers.Layer(self)
        
    def build(self):
        pass
        
    def __call__(self,x):
        pass
        
        

def conv1d(model,x,filters,kernel_size=1,activation=tf.nn.relu,dropout=0.1):
    conv_layer = tf.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        data_format='channels_last',
        dilation_rate=1,
        activation=activation,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.lecun_uniform(),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        trainable=True
    )
    model.addLayer(conv_layer)
    x = conv_layer(x)
    if dropout>0.:
        dropout_layer = tf.layers.Dropout(rate=dropout)
        x = dropout_layer(x,training=model.isTraining())
        model.addLayer(dropout_layer)
    return x

def dense(model,x,units,activation=tf.nn.relu,dropout=0.1,name=None):
    dense_layer = tf.layers.Dense(
        units=units,
        activation=activation,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.lecun_uniform(),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        trainable=True,
        name=None
    )
    model.addLayer(dense_layer)
    x = dense_layer(x)
    if dropout>0.:
        dropout_layer = tf.layers.Dropout(rate=dropout)
        x = dropout_layer(x,training=model.isTraining())
        model.addLayer(dropout_layer)
    return x



def makeModel(noutputs,cpf,npf,sv,event,isTraining):
    model = Model(isTraining=isTraining)
    
    cpf = conv1d(model,cpf,64)
    cpf = conv1d(model,cpf,32)
    cpf = conv1d(model,cpf,32)
    cpf = conv1d(model,cpf,8)

    npf = conv1d(model,npf,32)
    npf = conv1d(model,npf,16)
    npf = conv1d(model,npf,4)

    sv = conv1d(model,sv,32)
    sv = conv1d(model,sv,16)
    sv = conv1d(model,sv,6)

    flatcpf_layer = tf.layers.Flatten()
    #model.addLayer(flatcpf_layer)
    flatcpf = flatcpf_layer(cpf)
    
    flatnpf_layer = tf.layers.Flatten()
    #model.addLayer(flatnpf_layer)
    flatnpf = flatnpf_layer(npf)
    
    flatsv_layer = tf.layers.Flatten()
    #model.addLayer(flatsv_layer)
    flatsv = flatcpf_layer(sv)
    
    flat = tf.concat([flatcpf,flatnpf,flatsv,event],axis=1)
    
    out = dense(model,flat,200)
    out = dense(model,out,100)
    out = dense(model,out,100)
    out = dense(model,out,100)
    out = dense(model,out,100)
    out = dense(model,out,100)
    out = dense(model,out,100)
    
    output = dense(model,out,noutputs,activation=None,dropout=-1,name="output")
    return output,model
    
    

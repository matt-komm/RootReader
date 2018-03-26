import tensorflow as tf
from model import makeModel

featureDict = {

     "sv" : {
        "branches":[
            'sv_pt',
            'sv_deltaR',
            #'sv_mass',
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
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C/UInt_t',
            #'isC||isCC||isGCC/UInt_t',
            #'isUD||isS||isG/UInt_t',
            
            #'isB||isBB||isGBB/UInt_t',
            
            #'isB/UInt_t',
            #'isBB/UInt_t',
            #'isGBB/UInt_t',
            
            #'isLeptonicB||isLeptonicB_C/UInt_t',
            
            #'isLeptonicB/UInt_t',
            #'isLeptonicB_C/UInt_t',
            
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC/UInt_t',
            
            #'isC/UInt_t',
            #'isCC/UInt_t',
            #'isGCC/UInt_t',
            
            'jetorigin_isUD||jetorigin_isS/UInt_t',
            #'isUD/UInt_t',
            #'isS/UInt_t',
            
            
            'jetorigin_isG/UInt_t',
            
            'jetorigin_fromLLP/UInt_t',
            
            
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
            'global_pt',
            'global_eta',
            'ncpf',
            'nnpf',
            'nsv',
            #'npv',
            'csv_trackSumJetEtRatio', 
            'csv_trackSumJetDeltaR', 
            'csv_vertexCategory', 
            'csv_trackSip2dValAboveCharm', 
            'csv_trackSip2dSigAboveCharm', 
            'csv_trackSip3dValAboveCharm', 
            'csv_trackSip3dSigAboveCharm', 
            'csv_jetNSelectedTracks', 
            'csv_jetNTracksEtaRel'
        ],

    },


    "cpf": {
        "branches": [
            'cpf_trackEtaRel',
            'cpf_trackPtRel',
            'cpf_trackPPar',
            'cpf_trackDeltaR',
            'cpf_trackPParRatio',
            'cpf_trackSip2dVal',
            'cpf_trackSip2dSig',
            'cpf_trackSip3dVal',
            'cpf_trackSip3dSig',
            'cpf_trackJetDistVal',

            'cpf_ptrel', 
            'cpf_drminsv',
            'cpf_vertex_association',
            'cpf_puppi_weight',
            'cpf_track_chi2',
            'cpf_track_quality',
            
            'cpf_relIso01',
            'cpf_jetmassdroprel',
            
            
        ],
        "max":25
    },
    "npf": {
        "branches": [
            'npf_ptrel',
            'npf_deltaR',
            'npf_isGamma',
            'npf_hcal_fraction',
            'npf_drminsv',
            'npf_puppi_weight',
            
            'npf_relIso01',
            'npf_jetmassdroprel',
        ],
        "max":25
    }
}

with tf.Session(graph=tf.Graph()) as sess:
    cpf = tf.placeholder('float32',shape=(None,featureDict["cpf"]["max"],len(featureDict["cpf"]["branches"])),name="cpf")
    npf = tf.placeholder('float32',shape=(None,featureDict["npf"]["max"],len(featureDict["npf"]["branches"])),name="npf")
    sv = tf.placeholder('float32',shape=(None,featureDict["sv"]["max"],len(featureDict["sv"]["branches"])),name="sv")
    event = tf.placeholder('float32',shape=(None,len(featureDict["globals"]["branches"])),name="globals")

    output,model = makeModel(
        len(featureDict["truth"]["branches"]),
        cpf,npf,sv,event
    )
    prediction = tf.nn.softmax(output,name="prediction")

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    model.loadVariables("all/model_epoch28.json",sess)
    
    const_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        ["prediction"]
    )
    tf.train.write_graph(const_graph,"all","model_epoch28.pbtxt")
    











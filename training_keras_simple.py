import tensorflow as tf
import keras
from keras import backend as K
import os
import numpy
import ROOT
import time
import math
import random
import sys
from root_reader import root_reader
from train_test_splitter import train_test_splitter
from resampler import resampler

cvscale = 1.0

fontScale = 750./650.

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(1111)
ROOT.gStyle.SetPadTopMargin(0.08)
ROOT.gStyle.SetPadLeftMargin(0.145)
ROOT.gStyle.SetPadRightMargin(0.26)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetStatFontSize(0.025)

ROOT.gStyle.SetOptFit()
ROOT.gStyle.SetOptStat(0)

# For the canvas:
ROOT.gStyle.SetCanvasBorderMode(0)
ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
ROOT.gStyle.SetCanvasDefH(700) #Height of canvas
ROOT.gStyle.SetCanvasDefW(800) #Width of canvas
ROOT.gStyle.SetCanvasDefX(0)   #POsition on screen
ROOT.gStyle.SetCanvasDefY(0)

# For the Pad:
ROOT.gStyle.SetPadBorderMode(0)
# ROOT.gStyle.SetPadBorderSize(Width_t size = 1)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
#ROOT.gStyle.SetPadGridX(True)
#ROOT.gStyle.SetPadGridY(True)
ROOT.gStyle.SetGridColor(ROOT.kBlack)
ROOT.gStyle.SetGridStyle(2)
ROOT.gStyle.SetGridWidth(1)

# For the frame:

ROOT.gStyle.SetFrameBorderMode(0)
ROOT.gStyle.SetFrameBorderSize(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetFrameFillStyle(0)
ROOT.gStyle.SetFrameLineColor(1)
ROOT.gStyle.SetFrameLineStyle(1)
ROOT.gStyle.SetFrameLineWidth(0)

# For the histo:
# ROOT.gStyle.SetHistFillColor(1)
# ROOT.gStyle.SetHistFillStyle(0)
# ROOT.gStyle.SetLegoInnerR(Float_t rad = 0.5)
# ROOT.gStyle.SetNumberContours(Int_t number = 20)

ROOT.gStyle.SetEndErrorSize(2)
#ROOT.gStyle.SetErrorMarker(20)
ROOT.gStyle.SetErrorX(0.)

ROOT.gStyle.SetMarkerStyle(20)
#ROOT.gStyle.SetMarkerStyle(20)

#For the fit/function:
ROOT.gStyle.SetOptFit(1)
ROOT.gStyle.SetFitFormat("5.4g")
ROOT.gStyle.SetFuncColor(2)
ROOT.gStyle.SetFuncStyle(1)
ROOT.gStyle.SetFuncWidth(1)

#For the date:
ROOT.gStyle.SetOptDate(0)
# ROOT.gStyle.SetDateX(Float_t x = 0.01)
# ROOT.gStyle.SetDateY(Float_t y = 0.01)

# For the statistics box:
ROOT.gStyle.SetOptFile(0)
ROOT.gStyle.SetOptStat(0) # To display the mean and RMS:   SetOptStat("mr")
ROOT.gStyle.SetStatColor(ROOT.kWhite)
ROOT.gStyle.SetStatFont(42)
ROOT.gStyle.SetStatFontSize(0.025)
ROOT.gStyle.SetStatTextColor(1)
ROOT.gStyle.SetStatFormat("6.4g")
ROOT.gStyle.SetStatBorderSize(1)
ROOT.gStyle.SetStatH(0.1)
ROOT.gStyle.SetStatW(0.15)

ROOT.gStyle.SetHatchesSpacing(1.3/math.sqrt(cvscale))
ROOT.gStyle.SetHatchesLineWidth(int(2*cvscale))

# ROOT.gStyle.SetStaROOT.TStyle(Style_t style = 1001)
# ROOT.gStyle.SetStatX(Float_t x = 0)
# ROOT.gStyle.SetStatY(Float_t y = 0)


#ROOT.gROOT.ForceStyle(True)
#end modified

# For the Global title:

ROOT.gStyle.SetOptTitle(0)

# ROOT.gStyle.SetTitleH(0) # Set the height of the title box
# ROOT.gStyle.SetTitleW(0) # Set the width of the title box
#ROOT.gStyle.SetTitleX(0.35) # Set the position of the title box
#ROOT.gStyle.SetTitleY(0.986) # Set the position of the title box
# ROOT.gStyle.SetTitleStyle(Style_t style = 1001)
#ROOT.gStyle.SetTitleBorderSize(0)

# For the axis titles:
ROOT.gStyle.SetTitleColor(1, "XYZ")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(35*cvscale*fontScale, "XYZ")
# ROOT.gStyle.SetTitleXSize(Float_t size = 0.02) # Another way to set the size?
# ROOT.gStyle.SetTitleYSize(Float_t size = 0.02)
ROOT.gStyle.SetTitleXOffset(1.2)
#ROOT.gStyle.SetTitleYOffset(1.2)
ROOT.gStyle.SetTitleOffset(1.2, "YZ") # Another way to set the Offset

# For the axis labels:

ROOT.gStyle.SetLabelColor(1, "XYZ")
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelOffset(0.0077, "XYZ")
ROOT.gStyle.SetLabelSize(32*cvscale*fontScale, "XYZ")
#ROOT.gStyle.SetLabelSize(0.04, "XYZ")

# For the axis:

ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetStripDecimals(True)
ROOT.gStyle.SetTickLength(0.03, "Y")
ROOT.gStyle.SetTickLength(0.05, "X")
ROOT.gStyle.SetNdivisions(1005, "X")
ROOT.gStyle.SetNdivisions(506, "Y")

ROOT.gStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
ROOT.gStyle.SetPadTickY(1)

# Change for log plots:
ROOT.gStyle.SetOptLogx(0)
ROOT.gStyle.SetOptLogy(0)
ROOT.gStyle.SetOptLogz(0)

#ROOT.gStyle.SetPalette(1) #(1,0)

# another top group addition

# Postscript options:
#ROOT.gStyle.SetPaperSize(20., 20.)
#ROOT.gStyle.SetPaperSize(ROOT.TStyle.kA4)
#ROOT.gStyle.SetPaperSize(27., 29.7)
#ROOT.gStyle.SetPaperSize(27., 29.7)
ROOT.gStyle.SetPaperSize(8.0*1.6*cvscale,7.0*1.6*cvscale)
ROOT.TGaxis.SetMaxDigits(3)
ROOT.gStyle.SetLineScalePS(2)

# ROOT.gStyle.SetLineStyleString(Int_t i, const char* text)
# ROOT.gStyle.SetHeaderPS(const char* header)
# ROOT.gStyle.SetTitlePS(const char* pstitle)
#ROOT.gStyle.SetColorModelPS(1)

# ROOT.gStyle.SetBarOffset(Float_t baroff = 0.5)
# ROOT.gStyle.SetBarWidth(Float_t barwidth = 0.5)
# ROOT.gStyle.SetPaintTextFormat(const char* format = "g")
# ROOT.gStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
# ROOT.gStyle.SetTimeOffset(Double_t toffset)
# ROOT.gStyle.SetHistMinimumZero(kTRUE)

ROOT.gStyle.SetPaintTextFormat("3.0f")

NRGBs = 6;
NCont = 255;

stops = numpy.array( [0.00, 0.34,0.47, 0.61, 0.84, 1.00] )
red  = numpy.array( [0.5, 0.00,0.1, 1., 1.00, 0.81] )
green = numpy.array( [0.10, 0.71,0.85, 0.70, 0.20, 0.00] )
blue = numpy.array( [0.91, 1.00, 0.12,0.1, 0.00, 0.00] )

colWheelDark = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)

for i in range(NRGBs):
    red[i]=min(1,red[i]*1.1+0.25)
    green[i]=min(1,green[i]*1.1+0.25)
    blue[i]=min(1,blue[i]*1.1+0.25)

colWheel = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
ROOT.gStyle.SetNumberContours(NCont)
ROOT.gRandom.SetSeed(123)

colors=[]
def hex2rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255.0 for i in range(0, lv, lv // 3))

def newColor(red,green,blue):
    newColor.colorindex+=1
    color=ROOT.TColor(newColor.colorindex,red,green,blue)
    colors.append(color)
    return color
    
newColor.colorindex=301

def getDarkerColor(color):
    darkerColor=newColor(color.GetRed()*0.6,color.GetGreen()*0.6,color.GetBlue()*0.6)
    return darkerColor


import keras
from sklearn.metrics import confusion_matrix,roc_auc_score
import llp_model_simple

classificationweights_module = tf.load_op_library('./libClassificationWeights.so')
fakebackground_module = tf.load_op_library('./libFakeBackground.so')

import imp
try:
    if not os.environ.has_key('CUDA_VISIBLE_DEVICES'):
        imp.find_module('setGPU')
        import setGPU
    print "Using GPU: ",os.environ['CUDA_VISIBLE_DEVICES']
except ImportError:
    pass

filePathTrain = "/vols/cms/mkomm/LLP/nanox_ctau_1_train.txt"
filePathTest = "/vols/cms/mkomm/LLP/nanox_ctau_1_test.txt"

fileListTrain = []
fileListTest = []


outputFolder = "nanox_ctau_1_def4_test"
if os.path.exists(outputFolder):
    print "Warning: output folder '%s' already exists!"%outputFolder
else:
    print "Creating output folder '%s'!"%outputFolder
    os.makedirs(outputFolder)

f = open(filePathTrain)
for l in f:
    if len(l.replace("\n","").replace("\r",""))>0:
        absPath = os.path.join(filePathTrain.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    if os.path.exists(absPath):
        rootFile = ROOT.TFile(absPath)
        if not rootFile:
            continue
        tree = rootFile.Get("jets")
        if not tree:
            continue
        fileListTrain.append(absPath)
    else:
        print "WARNING: file '"+absPath+"' does not exists -> skip!"
f.close()

#fileListTrain = fileListTrain[:2]
print "files train ",len(fileListTrain)

f = open(filePathTest)
for l in f:
    if len(l.replace("\n","").replace("\r",""))>0:
        absPath = os.path.join(filePathTest.rsplit('/',1)[0],l.replace("\n","").replace("\r","")+"")
    if os.path.exists(absPath):
        rootFile = ROOT.TFile(absPath)
        if not rootFile:
            continue
        tree = rootFile.Get("jets")
        if not tree:
            continue
        fileListTest.append(absPath)
    else:
        print "WARNING: file '"+absPath+"' does not exists -> skip!"
f.close()
fileListTest = fileListTest[:3]
fileListTrain = fileListTest
print "files test ",len(fileListTest)


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
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',
            #'isC||isCC||isGCC/UInt_t',
            #'isUD||isS||isG/UInt_t',
            
            #'isB||isBB||isGBB/UInt_t',
            
            #'isB/UInt_t',
            #'isBB/UInt_t',
            #'isGBB/UInt_t',
            
            #'isLeptonicB||isLeptonicB_C/UInt_t',
            
            #'isLeptonicB/UInt_t',
            #'isLeptonicB_C/UInt_t',
            
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            
            #'isC/UInt_t',
            #'isCC/UInt_t',
            #'isGCC/UInt_t',
            
            'jetorigin_isUD||jetorigin_isS',
            #'isUD/UInt_t',
            #'isS/UInt_t',
            
            
            'jetorigin_isG',
            
            'jetorigin_fromLLP',
            
            
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
    
    "gen": {
        "branches":[
            "jetorigin_displacement"
        ]
    },
    
    "globalvars": {
        "branches": [
            'global_pt',
            'global_eta',
            'global_rho',
            'ncpf',
            'nnpf',
            'nsv',
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
            'cpf_track_quality'
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
            'npf_puppi_weight'
        ],
        "max":25
    }
}

style = {
    'isB':[newColor(0.8,0.45,0),3,1],
    'isBB':[newColor(0.85,0.42,0),3,2],
    'isGBB':[newColor(0.9,0.39,0),2,1],
    'isLeptonicB':[newColor(0.95,0.36,0),3,2],
    'isLeptonicB_C':[newColor(1,0.33,0),2,1],
    
    'isC':[newColor(0,0.9,0.1),3,2],
    'isCC':[newColor(0,0.8,0.25),2,1],
    'isGCC':[newColor(0,0.7,0.35),3,2],
    
    'isUD':[newColor(0.65,0.65,0.65),3,1],
    'isS':[newColor(0.55,0.55,0.55),3,2],
    'isG':[newColor(0.45,0.45,0.45),3,1],
    'isUndefined':[newColor(0.4,0.4,0.4),3,2],
    
    'isFromLLgno_isB':[newColor(0.0,0.1,1),3,1],
    'isFromLLgno_isBB':[newColor(0.0,0.13,0.95),3,2],
    'isFromLLgno_isGBB':[newColor(0.0,0.16,0.9),2,1],
    'isFromLLgno_isLeptonicB':[newColor(0.0,0.19,0.87),3,1],
    'isFromLLgno_isLeptonicB_C':[newColor(0.0,0.22,0.85),3,2],
    'isFromLLgno_isC':[newColor(0.0,0.25,0.83),2,1],
    'isFromLLgno_isCC':[newColor(0.0,0.28,0.8),3,2],
    'isFromLLgno_isGCC':[newColor(0.0,0.31,0.77),2,1],
    'isFromLLgno_isUD':[newColor(0.0,0.34,0.75),3,2],
    'isFromLLgno_isS':[newColor(0.0,0.37,0.73),2,1],
    'isFromLLgno_isG':[newColor(0.0,0.4,0.7),3,2],
    'isFromLLgno_isUndefined':[newColor(0.0,0.43,0.67),2,1],
}

def drawHists(histDict,branchNameList,legend):
    
    ll = 4
    b = 4
    c = 4
    other = 4
    for label in branchNameList:
        hist = histDict[label]
        legend.AddEntry(hist,label.replace("is",""),"L")
        if label.find("fromLLP")>=0:
            hist.SetLineColor(ROOT.kOrange+7)
            hist.SetLineWidth(ll/3)
            hist.SetLineStyle(ll%3+1)
            ll+=1
        elif label.find("B")>0:
            hist.SetLineColor(ROOT.kAzure-4)
            hist.SetLineWidth(b/2)
            hist.SetLineStyle(b%2+1)
            b+=1
        elif label.find("C")>0:
            hist.SetLineColor(ROOT.kGreen)
            hist.SetLineWidth(c/2)
            hist.SetLineStyle(c%2+1)
            c+=1
        else:
            hist.SetLineColor(ROOT.kMagenta)
            hist.SetLineWidth(other/2)
            hist.SetLineStyle(other%2+1)
            other+=1
        hist.Draw("SameHISTL")

def makePlot(histDict,branchNameList,binning,title,output,taget=None,logx=0,logy=0):
    cv = ROOT.TCanvas("cv","",1100,700)
    cv.SetLogx(logx)
    cv.SetLogy(logy)
    cv.SetRightMargin(0.36)
    ymax = max(map(lambda h: h.GetMaximum(),histDict.values()))
    axis = ROOT.TH2F("axis",title,50,binning[0],binning[-1],50,0,ymax*1.1)
    axis.Draw("AXIS")
    legend = ROOT.TLegend(0.67,0.98,0.99,0.02)
    legend.SetBorderSize(0)
    legend.SetFillColor(ROOT.kWhite)
    legend.SetTextFont(43)
    legend.SetTextSize(22*cvscale*fontScale)
    drawHists(histDict,branchNameList,legend)
    if taget:
        taget.SetLineWidth(3)
        taget.SetLineColor(ROOT.kBlack)
        taget.Draw("SameHISTL")
        legend.AddEntry(taget,"Target","L")
    legend.Draw("Same")
    cv.Update()
    cv.Print(os.path.join(outputFolder,output+".pdf"))


histsPerClass = {}
weightsPerClass = {}
chain = ROOT.TChain("jets")
for f in fileListTrain:
    chain.AddFile(f)
nEntries = chain.GetEntries()
print "total entries",nEntries


binningPt = numpy.linspace(1.3,3,num=30)
binningEta = numpy.linspace(-2.4,2.4,num=10)
targetShape = ROOT.TH2F("ptetaTarget","",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
branchNameList = []
eventsPerLabel = {}
targetEvents = 0
for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    branchNameList.append(branchName)
    print "projecting ... ",branchName
    hist = ROOT.TH2F("pteta"+branchName,"",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
    hist.Sumw2()
    #hist.SetDirectory(0)
    chain.Project(hist.GetName(),"global_eta:global_pt","("+branchName+"==1)")
    
    if label.find("fromLLP")>=0:# or label.find("isB")>=0:
        targetShape.Add(hist)
        targetEvents+=hist.GetEntries()
    if hist.Integral()>0:
        print " -> entries ",hist.GetEntries()
        eventsPerLabel[branchName]=hist.GetEntries()
        hist.Scale(1./hist.Integral())
    else:
        print " -> no entries found for class: ",branchName
    hist.Smooth()
    histsPerClass[branchName]=hist
    
targetShape.Scale(1./targetShape.Integral())
targetShape.Smooth()

for label in branchNameList:
    hist = histsPerClass[label]
    if (hist.Integral()>0):
        weight = targetShape.Clone(label)
        weight.Divide(hist)
        if weight.GetMaximum()>0:
            print "rescale ",label,1./(weight.GetMaximum())
            weight.Scale(1./weight.GetMaximum()) #ensure no crazy oversampling
        
        weightsPerClass[label]=weight
        for ibin in range(hist.GetNbinsX()):
            for jbin in range(hist.GetNbinsY()):
                if weight.GetBinContent(ibin+1,jbin+1)>0:
                    hist.SetBinContent(ibin+1,jbin+1,
                        targetShape.GetBinContent(ibin+1,jbin+1)/weight.GetBinContent(ibin+1,jbin+1)
                    )
                else:
                    hist.SetBinContent(ibin+1,jbin+1,0)
    else:
        weight = targetShape.Clone(label)
        weight.Scale(0)
        weightsPerClass[label]=weight
        
weightFile = ROOT.TFile(os.path.join(outputFolder,"weights.root"),"RECREATE")
for l,h in weightsPerClass.items():
    h.Write()
weightFile.Close()


histsPt = {l: h.ProjectionX() for l, h in histsPerClass.items()}
histsEta = {l: h.ProjectionY() for l, h in histsPerClass.items()}

makePlot(histsPt,branchNameList,binningPt,";Jet pT (GeV);Normalized events","pt",taget=targetShape.ProjectionX())
makePlot(histsEta,branchNameList,binningEta,";Jet #eta;Normalized events","eta",taget=targetShape.ProjectionY())

def divide(n,d):
    r = n.Clone(d.GetName())
    r.Divide(d)
    return r
weightsPt = {l: divide(targetShape.ProjectionX(),h.ProjectionX()) for l, h in histsPerClass.items()}
weightsEta = {l: divide(targetShape.ProjectionY(),h.ProjectionY()) for l, h in histsPerClass.items()}

makePlot(weightsPt,branchNameList,binningPt,";Jet pT (GeV);Weight","weight_pt",logy=1)
makePlot(weightsEta,branchNameList,binningEta,";Jet #eta;Weight","weight_eta",logy=1)



def setupModel(add_summary=False,options={}):
    result = {}
    globalvars = keras.layers.Input(shape=(len(featureDict["globalvars"]["branches"]),))
    cpf = keras.layers.Input(shape=(featureDict["cpf"]["max"],len(featureDict["cpf"]["branches"])))#tensor=batch['Cpfcan'])
    npf = keras.layers.Input(shape=(featureDict["npf"]["max"],len(featureDict["npf"]["branches"])))#tensor=batch['Npfcan'])
    sv = keras.layers.Input(shape=(featureDict["sv"]["max"],len(featureDict["sv"]["branches"])))#tensor=batch['sv'])

    print globalvars
    print cpf
    print npf
    print sv

    nclasses = len(featureDict["truth"]["branches"])
    print "Nclasses = ",nclasses
    conv_prediction,lstm1_prediction,full_prediction = llp_model_simple.model(
        globalvars,cpf,npf,sv,
        nclasses,
        options=options
    )
    w = 0.2*0.85**epoch
    print "Weighting loss: ",w
    prediction = keras.layers.Lambda(lambda x: (x[0]+x[1])*0.5*w+x[2]*(1-w))([conv_prediction,lstm1_prediction,full_prediction])
    #prediction = full_prediction
    return keras.Model(inputs=[globalvars,cpf,npf,sv], outputs=prediction)
    
    
    
    
def input_pipeline(files,batchSize):
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

        rootreader_op = []
        resamplers = []
        for _ in range(min(len(fileListTrain)-1,6)):
            reader = root_reader(fileListQueue, featureDict,"jets",batch=200).batch() 
            rootreader_op.append(reader)
            
            weight = classificationweights_module.classification_weights(
                reader["truth"],
                reader["globalvars"],
                os.path.join(outputFolder,"weights.root"),
                branchNameList,
                [0,1]
            )
            resampled = resampler(
                weight,
                reader
            ).resample()
            
            isSignal = resampled["truth"][:,4]>0.5 #index 4 is LLP
            resampled["gen"] = fakebackground_module.fake_background(resampled["gen"],isSignal,0)
            print resampled["gen"]
            
            resamplers.append(resampled)
            
        
        minAfterDequeue = batchSize*2
        capacity = minAfterDequeue + 3*batchSize
        
        batch = tf.train.shuffle_batch_join(
            #rootreader_op, 
            resamplers,
            batch_size=batchSize, 
            capacity=capacity,
            min_after_dequeue=minAfterDequeue,
            enqueue_many=True #requires to read examples in batches!
        )
        return batch

    
def getROC(signal,background):
    N=signal.GetNbinsX()+2
    sigHistContent=numpy.zeros(N)
    bgHistContent=numpy.zeros(N)
    
    for i in range(N):
        sigHistContent[i]=signal.GetBinContent(i)
        bgHistContent[i]=background.GetBinContent(i)

    sigN=sum(sigHistContent)
    bgN=sum(bgHistContent)

    sigEff=[]
    bgRej=[]
    bgEff=[]
    for ibin in range(N):
        sig_integral=0.0
        bg_integral=0.0
        for jbin in range(ibin,N):
            sig_integral+=sigHistContent[jbin]
            bg_integral+=bgHistContent[jbin]
        sigEff.append(sig_integral/sigN)
        bgRej.append(1-bg_integral/bgN)
        bgEff.append(bg_integral/bgN)
    return sigEff,bgRej,bgEff
    
def getAUC(sigEff,bgRej):
    integral=0.0
    for i in range(len(sigEff)-1):
        w=math.fabs(sigEff[i+1]-sigEff[i])
        h=0.5*(bgRej[i+1]+bgRej[i])
        x=(sigEff[i+1]+sigEff[i])*0.5
        integral+=w*math.fabs(h-(1-x))
    return math.fabs(integral)
    
def drawROC(name,sigEff,bgEff,signalName="Signal",backgroundName="Background",auc=None,style=1):

    cv = ROOT.TCanvas("cv_roc"+str(random.random()),"",800,600)
    cv.SetPad(0.0, 0.0, 1.0, 1.0)
    cv.SetFillStyle(4000)

    cv.SetBorderMode(0)
    #cv.SetGridx(True)
    #cv.SetGridy(True)

    #For the frame:
    cv.SetFrameBorderMode(0)
    cv.SetFrameBorderSize(1)
    cv.SetFrameFillColor(0)
    cv.SetFrameFillStyle(0)
    cv.SetFrameLineColor(1)
    cv.SetFrameLineStyle(1)
    cv.SetFrameLineWidth(1)

    # Margins:
    cv.SetLeftMargin(0.163)
    cv.SetRightMargin(0.03)
    cv.SetTopMargin(0.08)
    cv.SetBottomMargin(0.175)

    # For the Global title:
    cv.SetTitle("")

    # For the axis:
    cv.SetTickx(1)  # To get tick marks on the opposite side of the frame
    cv.SetTicky(1)

    cv.SetLogy(1)

    axis=ROOT.TH2F("axis"+str(random.random()),";"+signalName+" efficiency;"+backgroundName+" efficiency",50,0,1.0,50,0.0008,1.0)
    axis.GetYaxis().SetNdivisions(508)
    axis.GetXaxis().SetNdivisions(508)
    axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
    axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
    #axis.GetYaxis().SetNoExponent(True)
    axis.Draw("AXIS")

    #### draw here
    graphF = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphF.SetLineWidth(0)
    graphF.SetFillColor(ROOT.kOrange+10)
    #graphF.Draw("SameF")

    graphL = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphL.SetLineColor(ROOT.kOrange+7)
    graphL.SetLineWidth(3)
    graphL.SetLineStyle(style)
    graphL.Draw("SameL")

    ROOT.gPad.RedrawAxis()
    
    pCMS=ROOT.TPaveText(cv.GetLeftMargin(),0.94,cv.GetLeftMargin(),0.94,"NDC")
    pCMS.SetFillColor(ROOT.kWhite)
    pCMS.SetBorderSize(0)
    pCMS.SetTextFont(63)
    pCMS.SetTextSize(30*cvscale*fontScale)
    pCMS.SetTextAlign(11)
    pCMS.AddText("CMS")
    pCMS.Draw("Same")

    pPreliminary=ROOT.TPaveText(cv.GetLeftMargin()+0.095,0.94,cv.GetLeftMargin()+0.095,0.94,"NDC")
    pPreliminary.SetFillColor(ROOT.kWhite)
    pPreliminary.SetBorderSize(0)
    pPreliminary.SetTextFont(53)
    pPreliminary.SetTextSize(30*cvscale*fontScale)
    pPreliminary.SetTextAlign(11)
    pPreliminary.AddText("Simulation")
    pPreliminary.Draw("Same")
    
    
    if auc:
        pAUC=ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
        pAUC.SetFillColor(ROOT.kWhite)
        pAUC.SetBorderSize(0)
        pAUC.SetTextFont(43)
        pAUC.SetTextSize(32*cvscale*fontScale)
        pAUC.SetTextAlign(31)
        pAUC.AddText("AUC: % 4.1f %%" % (auc*100.0))
        pAUC.Draw("Same")

    cv.Update()
    cv.Print(name+".pdf")
    #cv.Print(name+".png")
    cv.WaitPrimitive()
    
    
learning_rate_val = 0.005
epoch = 0
previous_train_loss = 1000

while (epoch<61):

    epoch_duration = time.time()
    print "epoch",epoch+1
    
    
    train_batch = input_pipeline(fileListTrain,1000)
    test_batch = input_pipeline(fileListTest,1000)
    '''
    if os.environ.has_key('CUDA_VISIBLE_DEVICES'):
        modelTrain = setupModel(options={"GPULSTM":True})
        modelTest = setupModel()
    else:
    '''
    modelTrain = setupModel()
    modelTest = setupModel()
    
    opt = keras.optimizers.Adam(lr=learning_rate_val, beta_1=0.9, beta_2=0.999)
    modelTrain.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
    modelTest.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
    modelTrain.summary()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()
    sess.run(init_op)
    
    #summary_writer = tf.summary.FileWriter(os.path.join(outputFolder,"log"+str(epoch)), sess.graph)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    if os.path.exists(os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5")):
        print "loading weights ... ",os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5")
        modelTrain.load_weights(os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5"))
    elif epoch>0:
        print "no weights from previous epoch found"
        sys.exit(1)
        
    total_loss_train = 0
    total_loss_test = 0
    
    nTrain = 0
    nTest = 0
    start_time = time.time()

    try:
        step = 0
        while not coord.should_stop():
            train_batch_value = sess.run(train_batch)
            train_outputs = modelTrain.train_on_batch([
                    train_batch_value['globalvars'],
                    train_batch_value['cpf'],
                    train_batch_value['npf'],
                    train_batch_value['sv']
                ],
                train_batch_value["truth"]
            )
            step += 1
            nTrainBatch = train_batch_value["truth"].shape[0]
            
            nTrain += nTrainBatch
            
            if nTrainBatch>0:
                total_loss_train+=train_outputs[0]*nTrainBatch
                         
            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Training step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % (
                    step,
                    train_outputs[0],
                    train_outputs[1]*100.,
                    duration
                )
                
                start_time = time.time()
                
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
        
    modelTrain.save_weights(os.path.join(outputFolder,"model_epoch"+str(epoch)+".hdf5"))
    modelTest.set_weights(modelTrain.get_weights())
        
    hists = []
    histsScaled = []
    
    scores = []
    truths = []
    
    for branches1 in featureDict["truth"]["branches"]:
        disName = branches1.replace("||","_").replace("is","").replace("from","")
        histsPerDis = []
        histsPerDisScaled = []
        for branches2 in featureDict["truth"]["branches"]:
            probName = branches2.replace("||","_").replace("is","").replace("from","")
            
            h = ROOT.TH1F(disName+probName,probName,10000,0,1)
            histsPerDis.append(h)
            
            h = ROOT.TH1F(disName+probName+"scaled",probName,10000,0,1)
            histsPerDisScaled.append(h)
        hists.append(histsPerDis)
        histsScaled.append(histsPerDisScaled)
        
    try:
        step = 0
        while not coord.should_stop():
            test_batch_value = sess.run(test_batch)
            test_outputs = modelTest.test_on_batch([
                    test_batch_value['globalvars'],
                    test_batch_value['cpf'],
                    test_batch_value['npf'],
                    test_batch_value['sv']
                ],
                test_batch_value["truth"]
            )
            test_prediction = modelTest.predict_on_batch([
                    test_batch_value['globalvars'],
                    test_batch_value['cpf'],
                    test_batch_value['npf'],
                    test_batch_value['sv']
                ]
            )
            step += 1
            nTestBatch =test_batch_value["truth"].shape[0]
            
            for ibatch in range(test_batch_value["truth"].shape[0]):
                truthclass = numpy.argmax(test_batch_value["truth"][ibatch])
                predictedclass = numpy.argmax(test_prediction[ibatch])
                
                truths.append(truthclass)
                scores.append(predictedclass)
                
                maxProb = test_prediction[ibatch][predictedclass]
                for idis in range(len(featureDict["truth"]["branches"])):
                    hists[idis][truthclass].Fill(test_prediction[ibatch][idis])
                    histsScaled[idis][truthclass].Fill(test_prediction[ibatch][idis]/maxProb)
                
            nTest += nTestBatch
            
            if nTestBatch>0:
                total_loss_test+=test_outputs[0]*nTestBatch
                         
            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Testing step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % (
                    step,
                    test_outputs[0],
                    test_outputs[1]*100.,
                    duration
                )
                
                start_time = time.time()
                
            
    except tf.errors.OutOfRangeError:
        print('Done testing for %d steps.' % (step))
        

    avgLoss_train = total_loss_train/nTrain
    avgLoss_test = total_loss_test/nTest

    print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
    print "Training/Testing = %i/%i, Testing rate = %4.1f%%"%(nTrain,nTest,100.*nTest/(nTrain+nTest))
    print "Average loss = %.4f (%.4f)"%(avgLoss_train,avgLoss_test)
    print "Learning rate = %.4e"%(learning_rate_val)
    
    
    names = [
        "b jet",
        "c jet",
        "uds jet",
        "g jet",
        "LLP jet"
    ]
    
    
    for idis1 in range(len(featureDict["truth"]["branches"])):
        signalHist = hists[idis1][idis1]
        signalHistScaled = histsScaled[idis1][idis1]
        rocs = []
        aucs = []
        name = []
        for idis2 in range(len(featureDict["truth"]["branches"])):
            if idis2==idis1:
                continue
            
            bkgHist = hists[idis1][idis2]
            bkgHistScaled = histsScaled[idis1][idis2]
            sigEff,bgRej,bgEff = getROC(signalHist,bkgHist)
            sigEffScaled,bgRejScaled,bgEffScaled = getROC(signalHistScaled,bkgHistScaled)
            auc = getAUC(sigEff,bgRej)
            aucScaled = getAUC(sigEffScaled,bgRejScaled)
            print names[idis1],names[idis2],auc,aucScaled
            
            graph = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
            graph.SetLineWidth(3)
            graph.SetLineStyle(1+idis2%2)
            graph.SetLineColor(int(colWheelDark+250.*idis2/(len(featureDict["truth"]["branches"])-1)))
            rocs.append(graph)
            aucs.append(auc)
            name.append(names[idis2])
            
        cv = ROOT.TCanvas("cv_roc"+str(idis1),"",800,600)
        cv.SetRightMargin(0.25)
        cv.SetBottomMargin(0.18)
        cv.SetLeftMargin(0.16)
        cv.SetLogy(1)
        axis=ROOT.TH2F("axis"+str(random.random()),";"+names[idis1]+" efficiency;Background efficiency",50,0,1.0,50,0.0008,1.0)
        axis.GetYaxis().SetNdivisions(508)
        axis.GetXaxis().SetNdivisions(508)
        axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
        axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
        #axis.GetYaxis().SetNoExponent(True)
        axis.Draw("AXIS")
        legend = ROOT.TLegend(0.76,0.9,0.99,0.25)
        legend.SetBorderSize(0)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        
        for i,roc in enumerate(rocs):
            roc.Draw("SameL")
            legend.AddEntry(roc,name[i],"L")
            legend.AddEntry("","AUC %.1f%%"%(aucs[i]*100.),"")
        legend.Draw("Same")
        
        cv.Print(os.path.join(outputFolder,"roc "+names[idis1]+" epoch"+str(epoch)+".pdf"))
        
        
            
    rootOutput = ROOT.TFile(os.path.join(outputFolder,"report_epoch"+str(epoch)+".root"),"RECREATE")
    
    for idis in range(len(featureDict["truth"]["branches"])):
        cv = ROOT.TCanvas("cv"+str(idis),"",800,600)
        cv.SetRightMargin(0.25)
        cv.SetBottomMargin(0.18)
        cv.SetLeftMargin(0.16)
        
        ymax = 0.
        ymin = 1000
        for h in hists[idis]:
            h.Rebin(200)
            if h.Integral()>0:
                h.SetDirectory(rootOutput)
                h.Scale(1./h.Integral())
                ymax = max(ymax,h.GetMaximum())
                ymin = max(10**-5.5,min(ymin,h.GetMinimum()))
        disName = names[idis]
        axis = ROOT.TH2F("axis"+str(idis),";Prob("+disName+");Normalized events",50,0,1,50,ymin*0.6,ymax*1.2)
        axis.Draw("AXIS")
        cv.SetLogy(1)
        legend = ROOT.TLegend(0.76,0.9,0.99,0.35)
        legend.SetBorderSize(0)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        
        for iprob in range(len(featureDict["truth"]["branches"])):
            hists[idis][iprob].SetLineColor(int(colWheelDark+250.*iprob/(len(featureDict["truth"]["branches"])-1)))
            hists[idis][iprob].SetLineWidth(3)
            hists[idis][iprob].SetLineStyle(1+iprob%2)
            hists[idis][iprob].Draw("HISTSame")
            legend.AddEntry(hists[idis][iprob],names[iprob],"L")
        legend.Draw("Same")
        cv.Print(os.path.join(outputFolder,disName+" epoch"+str(epoch)+".pdf"))
        #cv.Print(os.path.join(outputFolder,disName+" epoch"+str(epoch)+".png"))
        
        
    for idis in range(len(featureDict["truth"]["branches"])):
        cv = ROOT.TCanvas("cv_scaled"+str(idis),"",800,600)
        cv.SetRightMargin(0.25)
        cv.SetBottomMargin(0.18)
        cv.SetLeftMargin(0.16)
        
        ymax = 0.
        ymin = 1000
        for h in histsScaled[idis]:
            h.Rebin(200)
            if h.Integral()>0:
                h.SetDirectory(rootOutput)
                h.Scale(1./h.Integral())
                ymax = max(ymax,h.GetMaximum())
                ymin = max(10**-5.5,min(ymin,h.GetMinimum()))
        disName = names[idis]
        axis = ROOT.TH2F("axisscaled"+str(idis),";Prob("+disName+") scaled;Normalized events",50,0,1,50,ymin*0.6,ymax*1.2)
        axis.Draw("AXIS")
        cv.SetLogy(1)
        legend = ROOT.TLegend(0.76,0.9,0.99,0.35)
        legend.SetBorderSize(0)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        
        for iprob in range(len(featureDict["truth"]["branches"])):
            histsScaled[idis][iprob].SetLineColor(int(colWheelDark+250.*iprob/(len(featureDict["truth"]["branches"])-1)))
            histsScaled[idis][iprob].SetLineWidth(3)
            histsScaled[idis][iprob].SetLineStyle(1+iprob%2)
            histsScaled[idis][iprob].Draw("HISTSame")
            legend.AddEntry(histsScaled[idis][iprob],names[iprob],"L")
        legend.Draw("Same")
        cv.Print(os.path.join(outputFolder,disName+" scaled epoch"+str(epoch)+".pdf"))
        
        
    conf_matrix = confusion_matrix(
        y_true=numpy.array(truths,dtype=int), 
        y_pred=numpy.array(scores,dtype=int),
        labels=range(len(featureDict["truth"]["branches"]))
    )
    
    
    conf_matrix_norm = numpy.zeros(conf_matrix.shape)
    for itruth in range(len(featureDict["truth"]["branches"])):
        total = 0.0
        for ipred in range(len(featureDict["truth"]["branches"])):
            total += conf_matrix[itruth][ipred]
        for ipred in range(len(featureDict["truth"]["branches"])):
            conf_matrix_norm[itruth][ipred] = 1.*conf_matrix[itruth][ipred]/total
      
    hist_conf = ROOT.TH2F("conf_hist","",
        len(featureDict["truth"]["branches"]),0,len(featureDict["truth"]["branches"]),
        len(featureDict["truth"]["branches"]),0,len(featureDict["truth"]["branches"])
    )
    hist_conf.SetDirectory(rootOutput)
    for itruth in range(len(featureDict["truth"]["branches"])):
        hist_conf.GetYaxis().SetBinLabel(itruth+1,"Pred. "+names[itruth])
        hist_conf.GetXaxis().SetBinLabel(itruth+1,"True "+names[itruth])
        for ipred in range(len(featureDict["truth"]["branches"])):
            hist_conf.SetBinContent(itruth+1,ipred+1,conf_matrix_norm[itruth][ipred]*100.)
    hist_conf.GetZaxis().SetTitle("Accuracy (%)")
    hist_conf.GetXaxis().SetLabelOffset(0.02)
    hist_conf.GetZaxis().SetTitleOffset(1.25)
    hist_conf.SetMarkerSize(1.8)
    cv = ROOT.TCanvas("conf","",900,700)
    cv.SetRightMargin(0.22)
    cv.SetBottomMargin(0.18)
    cv.SetLeftMargin(0.25)
    hist_conf.Draw("colztext")
    cv.Print(os.path.join(outputFolder,"confusion epoch"+str(epoch)+".pdf"))
    f = open(os.path.join(outputFolder,"model_epoch.stat"),"a")
    f.write(str(epoch)+";"+str(learning_rate_val)+";"+str(avgLoss_train)+";"+str(avgLoss_test)+"\n")
    f.close()
    rootOutput.Write()
    rootOutput.Close()
    if epoch>2 and previous_train_loss<avgLoss_train:
        learning_rate_val = learning_rate_val*0.9
        print "Decreasing learning rate to %.4e"%(learning_rate_val)
    previous_train_loss = avgLoss_train
 
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    
    epoch+=1

    

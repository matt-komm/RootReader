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

NRGBs = 5;
NCont = 255;

stops = numpy.array( [0.00, 0.34, 0.61, 0.84, 1.00] )
red  = numpy.array( [0.00, 0.00, 0.87, 1.00, 0.51] )
green = numpy.array( [0.00, 0.81, 1.00, 0.20, 0.00] )
blue = numpy.array( [0.51, 1.00, 0.12, 0.00, 0.00] )

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


from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D,LSTM,Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from deepFlavour import model_deepFlavourReference
from model import makeModel

classificationweights_module = tf.load_op_library('./libClassificationWeights.so')


import imp
try:
    if not os.environ.has_key('CUDA_VISIBLE_DEVICES'):
        imp.find_module('setGPU')
        import setGPU
    print "Using GPU: ",os.environ['CUDA_VISIBLE_DEVICES']
except ImportError:
    pass



fileListTrain = []
#filePathTrain = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples/rootFiles.raw.txt"
#filePathTrain = "/vols/cms/mkomm/LLP/samples/rootFiles_stripped2.txt"
#filePathTrain = "/vols/cms/mkomm/LLP/samples3_b_train_shuffle.txt"

#filePathTrain = "/vols/cms/mkomm/LLP/samples4_train_ttbar.txt"
#filePathTrain = "/vols/cms/mkomm/LLP/samples4_train_ctau1_red.txt"
#filePathTrain = "/vols/cms/mkomm/LLP/samples4_train_ctau10.txt"
#filePathTrain = "/vols/cms/mkomm/LLP/samples4_train_ctau100.txt"

#filePathTrain = "/vols/cms/mkomm/LLP/samples_nanox_train.txt"
filePathTrain = "/media/matthias/HDD/matthias/Analysis/LLP/training/samples_nanox_train.txt"

outputFolder = "all_v13"
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
        fileListTrain.append(absPath)
    else:
        print "WARNING: file '"+absPath+"' does not exists -> skip!"
f.close()
print "files train ",len(fileListTrain)

#fileListTrain = fileListTrain[:20]

#print fileList

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
        legend.AddEntry(hist,label.replace("is","").replace("jetorigin","").replace("_",""),"L")
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


binningPt = numpy.linspace(1.3,3.0,num=30)
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
            weight.Scale(1.2/weight.GetMaximum()) #ensure no crazy oversampling
        
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

makePlot(histsPt,branchNameList,binningPt,";Jet log(pT/1 GeV);Normalized events","pt",taget=targetShape.ProjectionX())
makePlot(histsEta,branchNameList,binningEta,";Jet #eta;Normalized events","eta",taget=targetShape.ProjectionY())

def divide(n,d):
    r = n.Clone(d.GetName())
    r.Divide(d)
    return r
weightsPt = {l: divide(targetShape.ProjectionX(),h.ProjectionX()) for l, h in histsPerClass.items()}
weightsEta = {l: divide(targetShape.ProjectionY(),h.ProjectionY()) for l, h in histsPerClass.items()}

makePlot(weightsPt,branchNameList,binningPt,";Jet log(pT/1 GeV);Weight","weight_pt",logy=1)
makePlot(weightsEta,branchNameList,binningEta,";Jet #eta;Weight","weight_eta",logy=1)



def setupModel(batch,isTraining,add_summary=False):
    result = {}
    
    truth = batch["truth"]
    nclasses = truth.shape.as_list()[1]
    output,model = makeModel(
        nclasses,
        batch["cpf"],
        batch["npf"],
        batch["sv"],
        batch["globals"],
        isTraining=isTraining
    )
    result["model"] = model
    
    prediction = tf.nn.softmax(output)

    '''
    bigtruth = tf.stack([tf.reduce_max(truth[:,0:nclasses-1],axis=1),truth[:,nclasses-1]],axis=1)
    bigoutput = tf.stack([tf.reduce_sum(output[:,0:nclasses-1],axis=1),output[:,nclasses-1]],axis=1)
    bigprediction =  keras.layers.Activation('softmax')(bigoutput)
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth,logits=output)
    loss = tf.reduce_mean(cross_entropy)
    
    #bigcross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=bigtruth,logits=bigoutput)
    #bigloss = tf.reduce_mean(bigcross_entropy)
    
    
    #bigcross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth,logits=output)
    #bigloss = tf.reduce_mean(cross_entropy)
    result["prediction"] = prediction
    result["truth"] = truth
    
    
    #result["bigprediction"] = bigprediction
    #result["bigtruth"] = bigtruth
    
    accuracy,accuracy_op = tf.metrics.accuracy(tf.argmax(truth,1),tf.argmax(prediction,1))
    result["accuracy"] = accuracy_op
    
    
    #print "Extra loss in model from Keras:", model.losses
    result["loss"] = loss
    #result["bigloss"] = bigloss
    #result["extraloss"] = tf.reduce_sum(model.losses)
    result["mimloss"] = loss#+tf.reduce_sum(model.losses) #bigloss+loss+tf.reduce_sum(model.losses)
    
    return result
    
global_step = 0

scanned_learning_rate = -100

isScanning = False

#learning_rate_val = 0.0005
learning_rate_val = 0.001
previous_losses = [1000,1000,1000]
epoch = 0
while (epoch<61):
    #if epoch%5==0:
    #    isScanning = not isScanning
    
    epoch_duration = time.time()
    print "epoch",epoch+1," isScanning =",isScanning
    
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(fileListTrain, num_epochs=1, shuffle=True)

        rootreader_op = []
        resamplers = []
        for _ in range(min(len(fileListTrain)-1,6)):
            reader = root_reader(fileListQueue, featureDict,"jets",batch=200).batch() 
            rootreader_op.append(reader)
            
            weight = classificationweights_module.classification_weights(
                reader["truth"],
                reader["globals"],
                os.path.join(outputFolder,"weights.root"),
                branchNameList,
                [0,1]
            )
            resampled = resampler(
                weight,
                reader
            ).resample()
            resamplers.append(resampled)
            
        
        batchSize = 10000
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
        train_test_split = train_test_splitter(
            batch["num"],
            batch,
            percentage=10
        )
        train_batch = train_test_split.train()
        test_batch = train_test_split.test()

    model_train = setupModel(train_batch,isTraining=True)
    print "Model parameters: ",model_train["model"].count_params()
    #print model_train["model"].getVariables()
    
    with tf.name_scope("test_input"):
        placeholder_test = {}
        for l in train_batch.keys():
            placeholder_test[l]=tf.placeholder(
                train_batch[l].dtype,
                train_batch[l].shape,
                name="test_"+l
            )
    
    #TODO: try to reuse variables
    model_test = setupModel(placeholder_test,isTraining=False)
    assign_varsToTest = model_test["model"].assignVariablesFromModel(model_train["model"])

    #model.add_loss(loss)
    #model.compile(optimizer='rmsprop', loss=None)
    #model.summary()
    #train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    learning_rate = tf.placeholder("float")
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope("minimizer"):
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-06,
            use_locking=True,
            name='Adam'
        ).minimize(
            model_train["mimloss"]
        )
    tf.summary.scalar('validation_loss', model_test["loss"])
    tf.summary.scalar('validation_acc', model_test["accuracy"])
    
    predictionsPerClass = tf.multiply(placeholder_test["truth"],model_test["prediction"])
    fightPerClass = tf.argmax(tf.multiply(1-placeholder_test["truth"],model_test["prediction"]),axis=1)
    
    for i,label in enumerate(featureDict["truth"]["branches"]):
        tf.summary.histogram('validation_prob_'+label.split('/')[0],predictionsPerClass[:,i])
        tf.summary.histogram('validation_fight_'+label.split('/')[0],fightPerClass)
    
    #summary_op = tf.summary.merge_all()

    #init_op = tf.global_variables_initializer() #bug https://github.com/tensorflow/tensorflow/issues/1045
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()
    
    sess.run(init_op)
    
    #summary_writer = tf.summary.FileWriter(os.path.join(outputFolder,"log"+str(epoch)), sess.graph)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    if epoch>0:
        modelPath = os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".json")
        if os.path.exists(modelPath):
            print "loading weights ... ",modelPath
            #use after init_op which initializes random weights!!!
            model_train["model"].loadVariables(modelPath,sess)
        else:
            print "no weights from previous epoch found"
            sys.exit(1)
    
        
    total_loss_train = 0
    total_loss_test = 0
    #total_bigloss_train = 0
    #total_bigloss_test = 0
    
    nTrain = 0
    nTest = 0
    start_time = time.time()

    NSCAN=100
    if isScanning:
        learning_rate_scan = numpy.zeros(NSCAN)
        loss_scan = numpy.zeros(NSCAN)
    
    predictions_per_class = numpy.zeros((model_test["prediction"].shape.as_list()[1],100,2))
    
    
    try:
        step = 0
        while not coord.should_stop():
            if isScanning:
                if step>=NSCAN:
                    break
                learning_rate_val = 10**(-7.+6.7*step/(NSCAN-1.))
                learning_rate_scan[step]=learning_rate_val
            else:
                #learning_rate_val = scanned_learning_rate
                #learning_rate_val = scanned_learning_rate*(0.1**(epoch/10.))
                #learning_rate_val = 0.0001*(0.1**(epoch/10.))+0.00001*(0.1**(epoch/20.))
                pass
            #loss is calculated before weights are updated
            #train_weights = model_train["model"].get_weights()
            #print train_weights
            #model_test["model"].set_weights(train_weights) 
            
            sess.run(assign_varsToTest)

            _,loss_train, accuracy_train,prediction_train_val,train_batch_value,test_batch_value = sess.run([
                    train_op,
                    model_train["loss"],model_train["accuracy"],
                    model_train["prediction"],train_batch,test_batch,
                ], 
                    feed_dict={learning_rate:learning_rate_val}
            )
            
            
            
            feed_dict = {learning_rate:learning_rate_val}
            for l in placeholder_test.keys():
                feed_dict[placeholder_test[l]]=test_batch_value[l]
                
            loss_test,accuracy_test,prediction_test_val = sess.run([
                    model_test["loss"],model_test["accuracy"],model_test["prediction"]
                ], 
                    feed_dict=feed_dict
            )
            if isScanning:
                loss_scan[step]=loss_test
            
            step += 1
            global_step+=1
            
            #account for dynamic batch size
            nTestBatch = len(test_batch_value["num"])
            nTest+=nTestBatch
            nTrain+=batchSize-nTestBatch
            if (batchSize-nTestBatch)>0:
                total_loss_train+=loss_train*(batchSize-nTestBatch)
            if (nTestBatch)>0:
                total_loss_test+=loss_test*nTestBatch
                    
                 
            labelIndices_train = numpy.argmax(train_batch_value["truth"],axis=1)
            labelIndices_test = numpy.argmax(test_batch_value["truth"],axis=1)
            for ibatch in range(len(labelIndices_train)):
                predictions_per_class[labelIndices_train[ibatch]][min(int(prediction_train_val[ibatch][labelIndices_train[ibatch]]*100.),99)][0] += 1
            for ibatch in range(len(labelIndices_test)):
                predictions_per_class[labelIndices_test[ibatch]][min(int(prediction_test_val[ibatch][labelIndices_test[ibatch]]*100.),99)][1] += 1
            
            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Step %d/~%d: loss = %.3f (%.3f), accuracy = %.2f%% (%.2f%%), time = %.3f sec' % (
                    step,
                    math.floor(1.*nEntries/batchSize),
                    loss_train,#bigloss_train,extraloss_train,
                    loss_test,#bigloss_test,extraloss_test,
                    accuracy_train*100.,
                    accuracy_test*100.,duration
                )
                start_time = time.time()
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
        
    if isScanning:
        
        loss_scan[0] = loss_scan[1] # this cannot be used since at beginning weights haven't been optimized
        loss_scan[-1] = loss_scan[-1]*0.5+loss_scan[-2]*0.5
        for i in range(1,NSCAN-1):
            loss_scan[i] = loss_scan[i-1]*0.25+loss_scan[i]*0.5+loss_scan[i+1]*0.25
        
        loss_error_scan = numpy.zeros(NSCAN)
        for i in range(1,NSCAN-1):
            loss_error_scan[i]=numpy.std([loss_scan[i-1],loss_scan[i],loss_scan[i+1]])
        loss_error_scan[0] = loss_error_scan[1]
        loss_error_scan[-1] = loss_error_scan[-2]
        
        cv = ROOT.TCanvas("cv"+str(epoch)+str(random.random()),"",750,650)
        cv.SetLogx(1)
        axis = ROOT.TH2F("axis"+str(epoch)+str(random.random()),";learning rate; loss",
            50,learning_rate_scan[0],learning_rate_scan[-1],
            50,numpy.min(loss_scan)*0.8,numpy.mean(loss_scan[0:10])+(numpy.mean(loss_scan[0:10])-0.8*numpy.min(loss_scan))
        )
        axis.Draw("AXIS")
        #graph = ROOT.TGraphErrors(NSCAN,learning_rate_scan,loss_scan,numpy.zeros(NSCAN),loss_error_scan)
        graph = ROOT.TGraph(NSCAN,learning_rate_scan,loss_scan)
        graph.SetLineColor(ROOT.kAzure-4)
        graph.SetLineWidth(2)
        graph.Draw("SameL")
        
        
        minLR = learning_rate_scan[numpy.argmin(loss_scan)]
        fit_range = learning_rate_scan[numpy.argmin(loss_scan)+int(NSCAN/20.)]
        fct = ROOT.TF1(
            "fct"+str(epoch)+str(random.random()),
            "[0]-[1]*TMath::Exp(-1./(2*[2]*[2])*TMath::Power(TMath::Log(x)-TMath::Log([3]),2))",
            learning_rate_scan[int(NSCAN/20.)],
            fit_range
        )
        fct.SetParameter(0,numpy.mean(loss_scan[0:10]))
        fct.SetParameter(1,numpy.mean(loss_scan[0:10]-numpy.min(loss_scan)))
        fct.SetParameter(2,0.3)
        fct.SetParameter(3,minLR)
        fct.SetLineColor(ROOT.kBlack)
        fct.SetLineWidth(2)
        graph.Fit(fct,"R")
        
        fct.Draw("SameL")
        
        #inflection points for exp(-0.5*x**2) are at x=-1/1
        x_opt = math.exp(math.log(fct.GetParameter(3))-fct.GetParameter(2))
        marker = ROOT.TMarker(x_opt,fct.Eval(x_opt),20)
        marker.SetMarkerSize(1.2)
        marker.Draw("SameP")
        cv.Print(os.path.join(outputFolder,"lr_scan_epoch"+str(epoch)+".pdf"))
        
        rootFile =ROOT.TFile(os.path.join(outputFolder,"lr_scan_epoch"+str(epoch)+".root"),"RECREATE")
        graph.SetName("graph")
        graph.Write()
        fct.SetName("fct")
        fct.Write()
        cv.SetName("cv")
        cv.Write()
        rootFile.Close()
        
        scanned_learning_rate = max([10**-7,x_opt])
        print "Set learning rate to %.4e"%(scanned_learning_rate)
        

    else:
        model_train["model"].saveVariables(os.path.join(outputFolder,"model_epoch"+str(epoch)+".json"),sess)
        #model_train["model"].save_weights(os.path.join(outputFolder,"model_epoch"+str(epoch)+".hdf5"))
        print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
        avgLoss_train = total_loss_train/nTrain
        avgLoss_test = total_loss_test/nTest
        #avgbigLoss_train = total_bigloss_train/nTrain
        #avgbigLoss_test = total_bigloss_test/nTest
        
        
        
        print "Training/Testing = %i/%i, Testing rate = %4.1f%%"%(nTrain,nTest,100.*nTest/(nTrain+nTest))
        print "Average loss = %.4f (%.4f)"%(avgLoss_train,avgLoss_test)
        print "Learning rate = = %.4e"%(learning_rate_val)
        threshold = 0#0.01/(1.+0.1*epoch)
        print "Improvement = %5.3e (threshold: %5.3e)"%(1.-1.*avgLoss_train/sum(previous_losses)*len(previous_losses),threshold)
        
        previous_losses[epoch%len(previous_losses)] = avgLoss_train
        
        if (1.-1.*avgLoss_train/sum(previous_losses)*len(previous_losses))<threshold:
            print "Optimizing learning rate"
            learning_rate_val = learning_rate_val*0.8
            previous_losses[epoch%len(previous_losses)] = 1.5*avgLoss_train/len(previous_losses) #prevents immediate decrease in next epochs
          
        
        
        for ilabel in range(predictions_per_class.shape[0]):
            sumEntries=numpy.sum(predictions_per_class[ilabel],axis=0)
            for ibin in range(predictions_per_class.shape[1]):
                predictions_per_class[ilabel][ibin][0]/=sumEntries[0]
                predictions_per_class[ilabel][ibin][1]/=sumEntries[1]
            numpy.savetxt(
                os.path.join(outputFolder,"predictions_epoch"+str(epoch)+"_"+branchNameList[ilabel].replace("||","_")+".np"),
                predictions_per_class[ilabel]
            )
            
        f = open(os.path.join(outputFolder,"model_epoch.stat"),"a")
        f.write(str(epoch)+";"+str(learning_rate_val)+";"+str(avgLoss_train)+";"+str(avgLoss_test)+";"+str(accuracy_train*100.)+";"+str(accuracy_test*100.)+"\n")
        f.close()
        
        epoch+=1
        
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    '''
    f = open(os.path.join(outputFolder,"scan_lr_epoch"+str(epoch)+".dat"),"w")
    for i in range(len(learning_rates)):
        f.write("%8.5e;%8.5e\n"%(learning_rates[i],loss_values[i]))
    f.close()
    '''
    

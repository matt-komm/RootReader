#!/usr/bin/python

import ROOT
import numpy
import random
import math
import os
import re
from optparse import OptionParser

cvscale = 1.0

fontScale = 750./650.

ROOT.gROOT.Reset()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
ROOT.gROOT.Reset()
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




fileList = []

filePath = "/vols/cms/mkomm/LLP/samples/rootFiles_test_stripped.txt"

f = open(filePath)
for l in f:
    fileName = os.path.join(filePath.rsplit('/',1)[0],l.replace("\n","").replace("\r",""))
    if (not os.path.exists(fileName+".friend")):
        #print "warning file '",fileName+".friend","' not found"
        continue
        fileList.append([fileName])
    else:
        fileList.append([fileName,fileName+".friend"])
f.close()
#fileList=fileList[:1]

print fileList
print "files",len(fileList)

rootObj = []
def makeHistogram(var,weight,binning):
    hist = ROOT.TH1F("hist"+str(random.random()),";"+var+";Events",len(binning)-1,binning)
    rootObj.append(hist)
    hist.SetDirectory(0)
    for f in fileList:
        rootFile = ROOT.TFile(f[0])
        tree = rootFile.Get("deepntuplizer/tree")
        if (tree):
            for fExtra in f[1:]:
                tree.AddFriend("evaluated",fExtra)
            histTemp = hist.Clone()
            histTemp.Scale(0)
            tree.Project(histTemp.GetName(),var,weight)
            histTemp.SetDirectory(0)
            hist.Add(histTemp)
        else:
            print "No tree found in file '",f[0],"'"
        rootFile.Close()
        
    return hist
    
    
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
    
def drawROC(name,sigEff,bgRej):

    cv = ROOT.TCanvas("cv_roc"+str(random.random()),"",800,700)
    cv.SetPad(0.0, 0.0, 1.0, 1.0)
    cv.SetFillStyle(4000)

    cv.SetBorderMode(0)
    cv.SetGridx(False)
    cv.SetGridy(False)

    #For the frame:
    cv.SetFrameBorderMode(0)
    cv.SetFrameBorderSize(1)
    cv.SetFrameFillColor(0)
    cv.SetFrameFillStyle(0)
    cv.SetFrameLineColor(1)
    cv.SetFrameLineStyle(1)
    cv.SetFrameLineWidth(1)

    # Margins:
    cv.SetLeftMargin(0.145)
    cv.SetRightMargin(0.03)
    cv.SetTopMargin(0.08)
    cv.SetBottomMargin(0.16)

    # For the Global title:
    cv.SetTitle("")

    # For the axis:
    cv.SetTickx(1)  # To get tick marks on the opposite side of the frame
    cv.SetTicky(1)

    #cv.SetLogy(1)

    axis=ROOT.TH2F("axis"+str(random.random()),";Signal efficiency;Background rejection",50,0,1.0,50,0.0001,1.0)
    axis.GetYaxis().SetNdivisions(508)
    axis.GetXaxis().SetNdivisions(508)
    axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
    axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
    #axis.GetYaxis().SetNoExponent(True)
    axis.Draw("AXIS")

    #### draw here
    graphF = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgRej))
    graphF.SetLineWidth(0)
    graphF.SetFillColor(ROOT.kAzure-4)
    graphF.Draw("SameF")

    graphL = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgRej))
    graphL.SetLineColor(ROOT.kAzure-5)
    graphL.SetLineWidth(2)
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
    
    
    pAUC=ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
    pAUC.SetFillColor(ROOT.kWhite)
    pAUC.SetBorderSize(0)
    pAUC.SetTextFont(43)
    pAUC.SetTextSize(32*cvscale*fontScale)
    pAUC.SetTextAlign(31)
    pAUC.AddText("AUC: % 4.1f %%" % (getAUC(sigEff,bgRej)*100.0))
    pAUC.Draw("Same")

    cv.Update()
    cv.Print(name+".pdf")
    cv.Print(name+".png")
    cv.WaitPrimitive()
    
def getAUC(sigEff,bgRej):
    integral=0.0
    for i in range(len(sigEff)-1):
        w=math.fabs(sigEff[i+1]-sigEff[i])
        h=0.5*(bgRej[i+1]+bgRej[i])
        x=(sigEff[i+1]+sigEff[i])*0.5
        integral+=w*math.fabs(h-(1-x))
    return math.fabs(integral)
    
def makeFlag(varList,isEval=False):
    ret = "(0"
    if isEval:
        for var in varList:
            ret+="+eval_"+var
    else:
        for var in varList:
            ret+="||("+var+"==1)"
    ret +=")"
    return ret

bFlags = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C']
cFlags = ['isC','isCC','isGCC']
lFlags = ['isUD','isS','isG','isUndefined']
llpbFlags = ['isFromLLgno_isB','isFromLLgno_isBB','isFromLLgno_isGBB','isFromLLgno_isLeptonicB','isFromLLgno_isLeptonicB_C']
llpcFlags = ['isFromLLgno_isC','isFromLLgno_isCC','isFromLLgno_isGCC']
llplFlags = ['isFromLLgno_isUD','isFromLLgno_isS','isFromLLgno_isG','isFromLLgno_isUndefined']

bFlagsAll = bFlags+llpbFlags
cFlagsAll = cFlags+llpcFlags
lFlagsAll = lFlags+llplFlags
llpFlagsAll = llpbFlags+llpcFlags+llplFlags
flagsAll = bFlags+llpbFlags+cFlags+llpcFlags+lFlags+llplFlags
    
for probVar in [
    ["probllp","Prob(LLP jet)",makeFlag(llpFlagsAll,True)],
    #["probb","Prob(b jet)",makeFlag(bFlags,True)],
    #["probc","Prob(c jet)",makeFlag(cFlags,True)],
    #["probl","Prob(udgs jet)",makeFlag(lFlags,True)],
]:
    for selection in [
        ["none","","1"],
        ["pt20_60","p#lower[0.3]{#scale[0.7]{T}}#in#kern[-0.55]{ }[20;60]#kern[-0.5]{ }GeV","(jet_pt>20)*(jet_pt<60)"],
        ["pt60_100","p#lower[0.3]{#scale[0.7]{T}}#in#kern[-0.55]{ }[60;100]#kern[-0.5]{ }GeV","(jet_pt>60)*(jet_pt<100)"],
        ["pt100_200","p#lower[0.3]{#scale[0.7]{T}}#in#kern[-0.55]{ }[100;200]#kern[-0.5]{ }GeV","(jet_pt>100)*(jet_pt<200)"],
        ["pt200_400","p#lower[0.3]{#scale[0.7]{T}}#in#kern[-0.55]{ }[200;400]#kern[-0.5]{ }GeV","(jet_pt>200)*(jet_pt<400)"],
        ["pt400","p#lower[0.3]{#scale[0.7]{T}}>400#kern[-0.5]{ }GeV","(jet_pt>400)"],
        ["prompt","L#lower[0.3]{#scale[0.7]{Lab}}#in#kern[-0.55]{ }[10#lower[-0.7]{#scale[0.7]{-5}};10#lower[-0.7]{#scale[0.7]{-2}}]#kern[-0.5]{ }cm","(genLL_decayLength>-5)*(genLL_decayLength<-2)"],
        ["blike","L#lower[0.3]{#scale[0.7]{Lab}}#in#kern[-0.55]{ }[10#lower[-0.7]{#scale[0.7]{-2}};10#lower[-0.7]{#scale[0.7]{1}}]#kern[-0.5]{ }cm","(genLL_decayLength>-2)*(genLL_decayLength<1)"],
        ["displaced","L#lower[0.3]{#scale[0.7]{Lab}}#in#kern[-0.55]{ }[10#lower[-0.7]{#scale[0.7]{1}};10#lower[-0.7]{#scale[0.7]{5}}]#kern[-0.5]{ }cm","(genLL_decayLength>1)*(genLL_decayLength<5)"],
        ["nosv","#SV=0","(nsv==0)"],
        ["wsv","#SV>0","(nsv>0)"],
    ]:
        cv = ROOT.TCanvas("cv","",800,700)
        legend = ROOT.TLegend(0.75,1-cv.GetTopMargin(),0.99,1-cv.GetTopMargin()-7*0.09)
        legend.SetBorderSize(0)
        legend.SetFillColor(ROOT.kWhite)
        legend.SetTextFont(43)
        legend.SetTextSize(32*cvscale*fontScale)

        histLL = makeHistogram(probVar[2],makeFlag(llpFlagsAll)+"*"+selection[2]+"*(jet_pt>20.)*(fabs(jet_eta)<2.4)",numpy.linspace(0,1,num=501))
        histLL.SetLineColor(ROOT.kOrange+7)
        histLL.SetLineWidth(3)
        histB = makeHistogram(probVar[2],makeFlag(bFlags)+"*"+selection[2]+"*(jet_pt>20.)*(fabs(jet_eta)<2.4)",numpy.linspace(0,1,num=501))
        histB.SetLineColor(ROOT.kAzure-4)
        histB.SetLineWidth(3)
        histC = makeHistogram(probVar[2],makeFlag(cFlags)+"*"+selection[2]+"*(jet_pt>20.)*(fabs(jet_eta)<2.4)",numpy.linspace(0,1,num=501))
        histC.SetLineColor(ROOT.kGreen+1)
        histC.SetLineWidth(4)
        histC.SetLineStyle(2)
        histL = makeHistogram(probVar[2],makeFlag(lFlags)+"*"+selection[2]+"*(jet_pt>20.)*(fabs(jet_eta)<2.4)",numpy.linspace(0,1,num=501))
        histL.SetLineColor(ROOT.kViolet)
        histL.SetLineWidth(4)
        histL.SetLineStyle(2)

        sigEff,bgRej,bgEff = getROC(histLL,histB)
        auc_b = getAUC(sigEff,bgRej)
        drawROC(probVar[0]+"_"+selection[0]+"_b_roc",sigEff,bgRej)
        
        sigEff,bgRej,bgEff = getROC(histLL,histC)
        auc_c = getAUC(sigEff,bgRej)
        drawROC(probVar[0]+"_"+selection[0]+"_c_roc",sigEff,bgRej)
        
        sigEff,bgRej,bgEff = getROC(histLL,histL)
        auc_l = getAUC(sigEff,bgRej)
        drawROC(probVar[0]+"_"+selection[0]+"_l_roc",sigEff,bgRej)
        

        histLL.Rebin(20)
        histB.Rebin(20)
        histC.Rebin(20)
        histL.Rebin(20)

        histLL.Scale(1./histLL.Integral())
        histB.Scale(1./histB.Integral())
        histC.Scale(1./histC.Integral())
        histL.Scale(1./histL.Integral())
        
        legend.AddEntry(histLL,"LLP jet","L")
        legend.AddEntry(histB,"b jet","L")
        legend.AddEntry("","(%4.1f%%)"%(auc_b*100),"")
        legend.AddEntry(histC,"c jet","L")
        legend.AddEntry("","(%4.1f%%)"%(auc_c*100),"")
        legend.AddEntry(histL,"udsg jet","L")
        legend.AddEntry("","(%4.1f%%)"%(auc_l*100),"")



        cv.SetLogy(1)
        axis = ROOT.TH2F("axis"+str(random.random()),";"+probVar[1]+";a.u.",50,0,1,50,0.0001,max(map(lambda x: x.GetMaximum(), [histLL,histB,histC,histL]))*1.2)
        axis.Draw("AXIS")

        histL.Draw("HISTSame")
        histC.Draw("HISTSame")
        histB.Draw("HISTSame")
        histLL.Draw("HISTSame")

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
        
        
        pSel=ROOT.TPaveText(1-cv.GetRightMargin()-0.1,0.84,1-cv.GetRightMargin()-0.1,0.84,"NDC")
        pSel.SetFillColor(ROOT.kWhite)
        pSel.SetBorderSize(0)
        pSel.SetTextFont(43)
        pSel.SetTextSize(30*cvscale*fontScale)
        pSel.SetTextAlign(31)
        pSel.AddText(selection[1])
        pSel.Draw("Same")


        legend.Draw("Same")
        cv.Update()
        cv.Print(probVar[0]+"_"+selection[0]+".pdf")
        cv.Print(probVar[0]+"_"+selection[0]+".png")
        cv.WaitPrimitive()



    


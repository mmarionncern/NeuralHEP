from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, LSTM, ZeroPadding2D
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from keras.utils.data_utils import get_file
from keras.backend import spatial_2d_padding, temporal_padding
import tensorflow as tf
#import simplejson
import numpy as np
import random
import sys
import math
import h5py
#import ROOT

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

import socket
host=socket.gethostname()
scratch=""
if host=="mmarionn-eth-laptop":
    scratch="/home/mmarionn/Documents/CMS/Computing"
    userDir="/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt"
    fileLoc=userDir+"/dataFiles"
else:
    import os
    scratch=os.environ['SCRATCH']
    userDir="/users/mmarionn/PhysProjects/TTV"
    fileLoc=scratch
    
sys.path.append(scratch+"/NeuralHEP/utils/")
#sys.path.append("/home/mmarionn/Documents/CMS/Computing/NeuralHEP/utils/")
from KerasSuperLayers import *
from NNFunctions import assignFlag#, storeOutput, evaluateModel
if host=="mmarionn-eth-laptop":
    #sys.path.append("/home/mmarionn/Documents/CMS/Computing/NeuralHEP/utils/")
    from NNFunctions import storeOutput


useROOT=False
ext=".h5"
if useROOT:
    import ROOT
    ext=".root"
    
def phi(x, y):
    phi_ =math.atan2(y, x)
    return phi_ if (phi_>=0) else phi_ + 2*np.pi

def dPhi(phi1, phi2):
    phi1_= phi( math.cos(phi1), math.sin(phi1) )
    phi2_= phi( math.cos(phi2), math.sin(phi2) )
    dphi_= phi1_-phi2_;
    if dphi_> np.pi: dphi_-=2*np.pi
    if dphi_<-np.pi: dphi_+=2*np.pi
    return dphi_


class LorentzVector():
    def __init__(self):
        self.isCart=None
        self.px=0
        self.py=0
        self.pz=0
        self.E=0
        self.pt=0
        self.eta=0
        self.theta=0
        self.phi=0
        self.m=0
        self.p=0
        self.px2=0
        self.py2=0
        self.pz2=0
        self.pt2=0
        self.p2=0
        self.m2=0
        self.E2=0
    def setPxPyPzE(self,input=[]):
        self.isCart=True
        self.px=input[0]
        self.py=input[1]
        self.pz=input[2]
        self.E=input[3]
        self.px2=self.px*self.px
        self.py2=self.py*self.py
        self.pz2=self.pz*self.pz
        self.pt2=self.px2+self.py2
        self.p2=self.px2+self.py2+self.pz2
        self.E2=self.E*self.E
        self.pt=self.getPtFromCart()
        self.eta=self.getEtaFromCart()
        self.phi=self.getPhiFromCart()
        self.m=self.getMFromCart()
        self.p=math.sqrt(self.p2)       
    def setPtEtaPhiM(self,input=[]):
        self.isCart=False
        self.pt=input[0]
        self.eta=input[1]
        self.phi=input[2]
        self.m=input[3]
        self.pt2=self.pt*self.pt
        self.m2=self.m*self.m
        self.px=self.getPxFromPEPM()
        self.py=self.getPyFromPEPM()
        self.pz=self.getPzFromPEPM()
        self.E=self.getEFromPEPM()       
    def getTheta(self):
        theta=2.0*math.atan(math.exp(-self.eta))
        if theta<0: theta+=2*np.pi
        return theta
    def getEta(self):
        if math.sin(self.theta/2.)==0: return 10000.*math.cos(self.theta/2.)
        else: return -1*math.log(math.tan(self.theta/2.))
    def getPtFromCart(self):
        return math.sqrt(self.pt2)
    def getEtaFromCart(self):
        if self.pt==0: return -10000 if self.pz<0 else 1000
        theta=np.pi/2.;
        if math.fabs(self.pz)>0:
            theta -= math.atan( self.pz/self.pt );
        self.theta=theta
        return self.getEta()
    def getPhiFromCart(self):
        phi=math.atan2(self.py, self.px)
        return phi
    def getMFromCart(self):
        return math.sqrt(self.E2-self.p2)
    def getPxFromPEPM(self):
        return self.pt*math.cos(self.phi)
    def getPyFromPEPM(self):
        return self.pt*math.sin(self.phi)
    def getPzFromPEPM(self):
        self.theta=self.getTheta()
        self.p=self.pt/math.sin(self.theta)
        return self.p*math.cos(self.theta)
    def getEFromPEPM(self):
        self.p2=self.p*self.p
        return math.sqrt(self.p2 + self.m2)
    def __add__(self,other):
        s=LorentzVector()
        s.setPxPyPzE([self.px + other.px,
                     self.py + other.py,
                     self.pz + other.pz,
                     self.E + other.E])
        return s

def buildZCandidate(leps):
    l1=LorentzVector()
    l2=LorentzVector()

    l1.setPtEtaPhiM(leps[0])
    l2.setPtEtaPhiM(leps[1])
    Z=l1+l2
    return [Z.pt,Z.eta,Z.phi,Z.m,0,0]

def buildWCandidate(lep,met):
    l=LorentzVector()
    m=LorentzVector()
    l.setPtEtaPhiM(lep)
    m.setPtEtaPhiM([met[0],0,met[1],0])
    dphi=dPhi(l.phi,m.phi)
    W=l+m
    Wmt=np.sqrt(2*lep[0]*met[0]*(1-np.cos(dphi)))
    output=[W.pt, 0, W.phi, Wmt,0,0]
    return output

#def buildZCandidate(leps):
#    
#    lep1=ROOT.TLorentzVector()
#    lep2=ROOT.TLorentzVector()
#    lep1.SetPtEtaPhiM(leps[0][0],leps[0][1],leps[0][2],0.0005 if abs(leps[0][3])==11 else 0.105)
#    lep2.SetPtEtaPhiM(leps[1][0],leps[1][1],leps[1][2],0.0005 if abs(leps[1][3])==11 else 0.105)
#
#    Z=lep1+lep2
#
#    output=[Z.Pt(),Z.Eta(),Z.Phi(),Z.M(),0,0]
#    
#    return output

#def buildWCandidate(lep,met):
#    lep4v=ROOT.TLorentzVector()
#    met4v=ROOT.TLorentzVector()
#    lep4v.SetPtEtaPhiM(lep[0],lep[1],lep[2],0.0005 if abs(lep[3])==11 else 0.105)
#    met4v.SetPtEtaPhiM(met[0],0,met[1],0)

#    WPt=(lep4v+met4v).Pt()
#    WPhi=(lep4v+met4v).Phi()
#    dPhi=lep4v.DeltaPhi(met4v)
#    WMT=np.sqrt(2*lep[0]*met[0]*(1-np.cos(dPhi)))

#    output=[WPt,0,WPhi,WMT,0,0]
    
#    return output 


def loadEventDatasetFromROOT(fname,nJMax,n=1000,
                     jecUnc=0, btagVar=0) :
    
    f = ROOT.TFile.Open(fname, "read")
    t = f.Get("tree")

    t.SetBranchStatus("*",0)
    t.SetBranchStatus("nJet",1)
    t.SetBranchStatus("Jet_pt",1)
    t.SetBranchStatus("Jet_eta",1)
    t.SetBranchStatus("Jet_phi",1)
    t.SetBranchStatus("Jet_mass",1)
    t.SetBranchStatus("Jet_btagDeepCSV",1)
    t.SetBranchStatus("Jet_qgl",1)
    t.SetBranchStatus("jetIdx",1)
    t.SetBranchStatus("Jet_corr_JECUp",1)
    t.SetBranchStatus("Jet_corr_JECDown",1)

    t.SetBranchStatus("lepWIdx",1)
    t.SetBranchStatus("lepZIdx",1)
    t.SetBranchStatus("LepGood_pt",1)
    t.SetBranchStatus("LepGood_eta",1)
    t.SetBranchStatus("LepGood_phi",1)
    t.SetBranchStatus("LepGood_pdgId",1)

    t.SetBranchStatus("met_pt")
    t.SetBranchStatus("met_phi")
    
    print('Loading data '+fname+' ...')
    events=[]
    nmax=min(n,t.GetEntries())
    n=min(n,t.GetEntries())
    for event in t:
        lepsZ=[]
        lepW=[]
        met=[]
        jets=[]
      

        #jets
        #nJets=len(event.jetIdx)
        nJets=sum([1 for i in event.jetIdx if i!=-1])
        nLep=sum([1 for i in event.lepZIdx if i!=-1]) 
        if nJets<2 or nLep!=2:
            continue       
        #print "new event",fname, n
        for k in range(0,min(nJets,nJMax)):
            j=event.jetIdx[k]
            corr=1
            if jecUnc!=0:
                corr=event.Jet_corr_JECUp[j] if jecUnc==1 else event.Jet_corr_JECDown[j]
            pt=event.Jet_pt[j]*corr
            jet=[pt,event.Jet_eta[j],
                 event.Jet_phi[j],event.Jet_mass[j],event.Jet_btagCSV[j]*(1+btagVar),event.Jet_qgl[j] ]
            jets.append(jet)

        if len(jets)<nJMax:
            for j in range(len(jets),nJMax):
                jets.append([0,0,0,0,0,0]) 
            
        lep1=event.lepZIdx[0]
        lep2=event.lepZIdx[1]
        lepsZ.append([event.LepGood_pt[lep1], event.LepGood_eta[lep1],
                     event.LepGood_phi[lep1], 0.0005 if abs(event.LepGood_pdgId[lep1]) else 0.105,
                     0,0])
        lepsZ.append([event.LepGood_pt[lep2], event.LepGood_eta[lep2],
                     event.LepGood_phi[lep2], 0.0005 if abs(event.LepGood_pdgId[lep2]) else 0.105,
                     0,0])
        lw=event.lepWIdx
        lepW.extend([event.LepGood_pt[lw], event.LepGood_eta[lw],
                     event.LepGood_phi[lw], 0.0005 if abs(event.LepGood_pdgId[lw]) else 0.105,
                     0,0])
        met.extend([ event.met_pt, 0, event.met_phi, 0, 0, 0 ])
        
        ZCand=buildZCandidate(lepsZ)
        WCand=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsZ[0],met)
        WCand3=buildWCandidate(lepsZ[1],met)

        
        evt=[lepsZ[0], lepsZ[1], lepW, met]
        evt.extend(jets)
        evt.extend([ZCand, WCand, WCand2, WCand3 ])
        
        events.append(evt)
        n-=1
        #sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        #sys.stdout.flush()
        if n==0: break

    return events



def loadEventDatasetFromHDF5(fname,nJMax,n=1000,
                     jecUnc=0, btagVar=0) :
    
    f = h5py.File(fname, 'r')
    a_group_key = list(f.keys())[0]
    data = f[a_group_key]
    
    kNJet=0
    kNLepGood=1
    kLepWIdx=2
    kMetPt=3
    kMetPhi=4
    kLepPdgId=5
    kJetIdx=6
    kLepZIdx=7
    kJetBTag=8
    kJetJECDo=9
    kJetJECUp=10
    kJetEta=11
    kJetMass=12
    kJetPhi=13
    kJetPt=14
    kJetQGL=15
    kLepEta=16
    kLepPhi=17
    kLepPt=18
    
    print('Loading data '+fname+' ...')
    events=[]
    nData=len(data)
    nmax=min(n,nData)
    n=min(n,nData)
    for event in data:
        lepsZ=[]
        lepW=[]
        met=[]
        jets=[]
      
        #jets
        #nJets=len(event.jetIdx)
        nJets=sum([1 for i in event[kJetIdx] if i!=-1])
        nLep=sum([1 for i in event[kLepZIdx] if i!=-1]) 
        if nJets<2 or nLep!=2:
            continue       
        #print "new event",fname, n
        for k in range(0,min(nJets,nJMax)):
            j=event[kJetIdx][k]
            corr=1
            if jecUnc!=0:
                corr=event[kJetJECUp][j] if jecUnc==1 else event[kJetJECDo][j]
            pt=event[kJetPt][j]*corr
            jet4v=LorentzVector()
            jet4v.setPtEtaPhiM([pt,event[kJetEta][j],event[kJetPhi][j],event[kJetMass][j]])
            jet=[jet4v.pt,jet4v.eta,jet4v.phi,jet4v.m,
                 event[kJetBTag][j]*(1+btagVar),event[kJetQGL][j]]
            jets.append(jet)

        if len(jets)<nJMax:
            for j in range(len(jets),nJMax):
                jets.append([0,0,0,0,0,0]) 
            
        lep1=event[kLepZIdx][0]
        lep2=event[kLepZIdx][1]
        lepsZ.append([event[kLepPt][lep1], event[kLepEta][lep1],
                     event[kLepPhi][lep1], 0.0005 if abs(event[kLepPdgId][lep1]) else 0.105,
                     0,0])
        lepsZ.append([event[kLepPt][lep2], event[kLepEta][lep2],
                     event[kLepPhi][lep2], 0.0005 if abs(event[kLepPdgId][lep2]) else 0.105,
                     0,0])
        lw=event[kLepWIdx]
        lepW.extend([event[kLepPt][lw], event[kLepEta][lw],
                     event[kLepPhi][lw], 0.0005 if abs(event[kLepPdgId][lw]) else 0.105,
                     0,0])
        met.extend([ event[kMetPt], 0, event[kMetPhi], 0, 0, 0 ])
        
        ZCand=buildZCandidate(lepsZ)
        WCand=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsZ[0],met)
        WCand3=buildWCandidate(lepsZ[1],met)

        evt=[lepsZ[0], lepsZ[1], lepW, met]
        evt.extend(jets)
        evt.extend([ZCand, WCand, WCand2, WCand3 ])
        
        events.append(evt)
        n-=1
        #sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        #sys.stdout.flush()
        if n==0: break

    return events



def loadEventDataset(fname,nJMax,n=1000,
                     jecUnc=0, btagVar=0):

    if useROOT:
        return loadEventDatasetFromROOT(fname, nJMax, n, jecUnc, btagVar)
    else:
        return loadEventDatasetFromHDF5(fname, nJMax, n, jecUnc, btagVar)


def makeModelV1(nJMax,nOut,version="1.0", f=2):

    nJets=nJMax
    
    inputVals=Input(shape=(8+nJets,6))

    inputRawLeps=cropData(inputVals, ymin=0, ymax=3, xmin=0, xmax=3)
    #inputLepZ1=cropData(inputVals, ymin=0, ymax=0, xmin=0, xmax=3)
    #inputLepZ2=cropData(inputVals, ymin=1, ymax=1, xmin=0, xmax=3)
    #inputLepW=cropData(inputVals, ymin=2, ymax=2, xmin=0, xmax=3)
    #inputMet=cropData(inputVals, ymin=3, ymax=3, xmin=0, xmax=3)
    inputJets=cropData(inputVals,ymin=4,ymax=4+(nJets-1), xmin=0, xmax=5)
    #print inputJets._keras_shape
    nOffset=4+nJets
    inputZ=cropData(inputVals,ymin=nOffset,ymax=nOffset, xmin=0, xmax=3)
    inputW=cropData(inputVals,ymin=nOffset+1,ymax=nOffset+1, xmin=0, xmax=3)
    #inputWCart=Convert4VBlocks(inputW,cartToPEPM=False).get()
    
    #inputW1=cropData(inputVals,ymin=nOffset+2,ymax=nOffset+2, xmin=0, xmax=3)
    #inputW2=cropData(inputVals,ymin=nOffset+3,ymax=nOffset+3, xmin=0, xmax=3)
    inputEWK=cropData(inputVals,ymin=nOffset,ymax=nOffset+3, xmin=0, xmax=3)
    
    #jet doublets =====================================================================
    jetHadWDoublets=BuildCombinationsDim1(inputJets,2).get()
    jetHadWDoubletsKin=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=0, xmax=3)
    jetHadWDoubletsBTag=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=4, xmax=4)
    jetHadWDoubletsQG=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=5, xmax=5)
    
    #jet triplets =====================================================================
    jetTriplets=BuildCombinationsDim1(inputJets,3).get()
    jetTripletsKin=cropData(jetTriplets, ymin=0, ymax=-1, xmin=0, xmax=3)
    jetTripletsBTag=cropData(jetTriplets, ymin=0, ymax=-1, xmin=4, xmax=4)
    jetTripletsQG=cropData(jetTriplets, ymin=0, ymax=-1, xmin=5, xmax=5)

    #ordering of scalars
    #jetTripletsBTag=SequentialSort(k=3, colIdx=0)(jetTripletsBTag)
    #jetTripletsQG=SequentialSort(k=3, colIdx=0)(jetTripletsQG)
    
    #jet doublets inside the triplets =================================================
    #jetDoublets=BuildSequentialCombinationsDim1(jetTriplets,k=2,step=3).get()
    #jetDoubletsKin=cropData(jetDoublets, ymin=0, ymax=-1, xmin=0, xmax=3)
    #jetDoubletsBTag=cropData(jetDoublets, ymin=0, ymax=-1, xmin=4, xmax=4)
    #jetDoubletsQG=cropData(jetDoublets, ymin=0, ymax=-1, xmin=5, xmax=5)
    #jetDoubletsDr=DeltaR()(jetDoubletsKin)

    #ordering of scalars
    #jetDoubletsBTag=SequentialSort(k=2, colIdx=0)(jetDoubletsBTag)
    #jetDoubletsQG=SequentialSort(k=2, colIdx=0)(jetDoubletsQG)
    #jetDoubletsDr=SequentialSort(k=2, colIdx=0, reverse=True)(jetDoubletsDr)

    #top 4 vectors, sorted by top mass ================================================

    #hadTopsKin=Convert4VBlocks(jetTripletsKin,cartToPEPM=False).get()
    hadTopsKin=SumCombinatorial4VBlocks(jetTripletsKin, 3).get()
    #hadTopsKin=Convert4VBlocks(hadTopsKin, cartToPEPM=True).get()
    hadTopsBTagMax=SequentialReduceOperation(operation="max",k=3)(jetTripletsBTag)
    hadTopsBTagAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsBTag)
    hadTopsQGAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsQG)

    tmpJetTriplet=SequentialSort(k=3, colIdx=4)(jetTriplets)
    tmpJetTriplet=Reshape((-1,3,6))(tmpJetTriplet)
    tmpWs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=1,xmax=2))
    tmpWs=cropData(tmpWs,ymin=0, ymax=-1, xmin=0, xmax=3)

    #tmpWs=Convert4VBlocks(tmpWs, cartToPEPM=False).get()
    tmpWs=SumCombinatorial4VBlocks(tmpWs,2).get()
    #tmpWs=Convert4VBlocks( tmpWs , cartToPEPM=True).get()

    tmpBs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=0,xmax=0))
    tmpBs=cropData(tmpBs,ymin=0, ymax=-1, xmin=0, xmax=3)
    tmpComb=Concatenate(axis=2)([tmpWs,tmpBs])
    tmpComb=Reshape((-1,4))(tmpComb)
    hadTopDrMaxBW=DeltaR()(tmpComb)
    
    hadTops=Concatenate()([hadTopsKin,hadTopsBTagMax,hadTopsBTagAvg,hadTopsQGAvg,hadTopDrMaxBW])
    hadTops=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(hadTops)

    #do not uncomment!!!
    ##doublets 4 vectors, sorted by W mass =============================================
    ##hadTWsKin=SumCombinatorial4VBlocks(jetDoubletsKin,2).get()
    ##hadTWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetDoubletsBTag)
    ##hadTWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsBTag)
    ##hadTWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetDoubletsBTag)
    ##hadTWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsQG)
    ##hadTWsDr=DeltaR()(jetDoubletsKin)
    ##hadTWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg, hadWsDr])
    ##hadTWs=SequentialSort(k=3,colIdx=3,reverse=True, sortByClosestValue=True, target=80.4)(hadWs)

    #hadronic Ws 4 vector, sorted by W mass ===============================================
    
    #hadWsKin=Convert4VBlocks(jetHadWDoubletsKin,cartToPEPM=False).get()
    hadWsKin=SumCombinatorial4VBlocks(jetHadWDoubletsKin,2).get()
    #hadWsKin=Convert4VBlocks(hadWsKin, cartToPEPM=True).get()
    hadWsDr=DeltaR()(jetHadWDoubletsKin)
    hadWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetHadWDoubletsBTag)
    hadWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsQG)
    hadWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg,hadWsDr])
    hadWs=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=80.4)(hadWs)
    
    #semileptonic tops ================================================================
    multiWs=UpSampling1D(nJets)(inputW)
    #multiWs=UpSampling1D(nJets)(inputWCart)
    jetsForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=0,xmax=3)
    bScoreForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=4,xmax=4)
    semiLepTops=Concatenate(axis=2)([multiWs,jetsForsemiLepTops])
    semiLepTops=Reshape((-1,4))(semiLepTops)
    semiLepTopsDr=DeltaR()(semiLepTops)

    #semiLepTops=Convert4VBlocks(semiLepTops, cartToPEPM=False).get()
    semiLepTops=SumCombinatorial4VBlocks(semiLepTops,2).get()
    #semiLepTops=Convert4VBlocks(semiLepTops, cartToPEPM=True).get()
    
    semiLepTops=Concatenate()([semiLepTops, semiLepTopsDr, bScoreForsemiLepTops ])
    semiLepTops=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(semiLepTops)
    
    #dR between Z and tops ===========================================================
    multiZsSemiLep=cropData(inputZ,ymin=0,xmin=0,xmax=3)
    multiZsSemiLep=UpSampling1D(nJets)(multiZsSemiLep)
    redSLTops=cropData(semiLepTops,ymin=0,xmin=0,xmax=3)
    ZSLTops=Concatenate(axis=2)([multiZsSemiLep,redSLTops])
    ZSLTops=Reshape((-1,4))(ZSLTops)
    ZSLTopsDr=DeltaR()(ZSLTops)

    multiZsHad=cropData(inputZ,ymin=0,xmin=0,xmax=3)
    multiZsHad=UpSampling1D(hadTops._keras_shape[1])(multiZsHad)
    redHTops=cropData(hadTops,ymin=0,xmin=0,xmax=3)
    ZHTops=Concatenate(axis=2)([multiZsHad,redHTops])
    ZHTops=Reshape((-1,4))(ZHTops)
    ZHTopsDr=DeltaR()(ZHTops)

    #concatenation of the dR and the tops =============================================
    #print hadTops._keras_shape, ZHTopsDr._keras_shape
    hadTops=Concatenate()([hadTops, ZHTopsDr])
    semiLepTops=Concatenate()([semiLepTops, ZSLTopsDr])

    #jet btagging
    inputJetsBTag=cropData(inputVals,ymin=4,ymax=4+(nJets-1), xmin=4, xmax=4)
    inputJetsBTag=Sort(colIdx=0, reverse=False, sortByClosestValue=False, target=None)(inputJetsBTag)
    
    ##=======================================
    ## Now active layers ====================
    ##=======================================

    ##hadronic tops
    hadTopPart=superLSTMLayer(hadTops,"htop_lstm1",64/f, return_sequence=True)
    hadTopPart=superLSTMLayer(hadTopPart,"htop_lstm2",64/f)

    ##semi leptonic tops
    semiLepTopPart=superLSTMLayer(semiLepTops,"sltop_lstm1",64/f, return_sequence=True)
    semiLepTopPart=superLSTMLayer(semiLepTopPart,"sltop_lstm2",64/f)

    ##hadronic Ws
    hadWsPart=superLSTMLayer(hadWs,"hW_lstm1",64/f, return_sequence=True)
    hadWsPart=superLSTMLayer(hadWsPart,"hW_lstm2",64/f)

    ##all jets
    jetPart=superLSTMLayer(inputJets,"jet_lstm1",64/f, return_sequence=True)
    jetPart=superLSTMLayer(jetPart,"jet_lstm2",64/f)

    jetBTagPart=superLSTMLayer(inputJetsBTag,"btag_lstm1",64/f, return_sequence=True)
    jetBTagPart=superLSTMLayer(jetBTagPart,"btag_lstm2",64/f)
    
    ##all leptons
    leptonPart=superLSTMLayer(inputRawLeps,"lep_lstm1",64/f, return_sequence=True)
    leptonPart=superLSTMLayer(leptonPart,"lep_lstm2",64/f)

    ##all leptons
    ewkPart=superLSTMLayer(inputEWK,"ewk_lstm1",64/f, return_sequence=True)
    ewkPart=superLSTMLayer(ewkPart,"ewk_lstm2",64/f)

    ##general concatenation
    mergingAll=Concatenate()([hadTopPart,semiLepTopPart,hadWsPart,jetPart,jetBTagPart,leptonPart,ewkPart])
    
    dense=superDenseLayer(mergingAll, "d1", nNodes=256/f)
    dense=superDenseLayer(dense, "d2", nNodes=128/f)
    dense=superDenseLayer(dense, "d3", nNodes=128/f)
    dense=superDenseLayer(dense, "d4", nNodes=128/f)
    
    finalOutput=Dense(nOut, activation='softmax')(dense)
    
    #model = Model(inputs=inputVals, outputs=hadWsKin)
    model = Model(inputs=inputVals,outputs=finalOutput)   
    #model.summary()

    return model



def makeModelV2(nJMax,nOut, f=2, prepareData=False):

    nJets=nJMax
    
    inputVals=Input(shape=(8+nJets,6))

    inputRawLeps=cropData(inputVals, ymin=0, ymax=3, xmin=0, xmax=3)
    #inputLepZ1=cropData(inputVals, ymin=0, ymax=0, xmin=0, xmax=3)
    #inputLepZ2=cropData(inputVals, ymin=1, ymax=1, xmin=0, xmax=3)
    #inputLepW=cropData(inputVals, ymin=2, ymax=2, xmin=0, xmax=3)
    #inputMet=cropData(inputVals, ymin=3, ymax=3, xmin=0, xmax=3)
    inputJets=cropData(inputVals,ymin=4,ymax=4+(nJets-1), xmin=0, xmax=5)
    #print inputJets._keras_shape
    nOffset=4+nJets
    inputZ=cropData(inputVals,ymin=nOffset,ymax=nOffset, xmin=0, xmax=3)
    inputW=cropData(inputVals,ymin=nOffset+1,ymax=nOffset+1, xmin=0, xmax=3)
    #inputWCart=Convert4VBlocks(inputW,cartToPEPM=False).get()
    
    #inputW1=cropData(inputVals,ymin=nOffset+2,ymax=nOffset+2, xmin=0, xmax=3)
    #inputW2=cropData(inputVals,ymin=nOffset+3,ymax=nOffset+3, xmin=0, xmax=3)
    inputEWK=cropData(inputVals,ymin=nOffset,ymax=nOffset+3, xmin=0, xmax=3)
    
    #jet doublets =====================================================================
    jetHadWDoublets=BuildCombinationsDim1(inputJets,2).get()
    jetHadWDoubletsKin=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=0, xmax=3)
    jetHadWDoubletsBTag=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=4, xmax=4)
    jetHadWDoubletsQG=cropData(jetHadWDoublets, ymin=0, ymax=-1, xmin=5, xmax=5)
    
    #jet triplets =====================================================================
    jetTriplets=BuildCombinationsDim1(inputJets,3).get()
    jetTripletsKin=cropData(jetTriplets, ymin=0, ymax=-1, xmin=0, xmax=3)
    jetTripletsBTag=cropData(jetTriplets, ymin=0, ymax=-1, xmin=4, xmax=4)
    jetTripletsQG=cropData(jetTriplets, ymin=0, ymax=-1, xmin=5, xmax=5)

    #ordering of scalars
    #jetTripletsBTag=SequentialSort(k=3, colIdx=0)(jetTripletsBTag)
    #jetTripletsQG=SequentialSort(k=3, colIdx=0)(jetTripletsQG)
    
    #jet doublets inside the triplets =================================================
    #jetDoublets=BuildSequentialCombinationsDim1(jetTriplets,k=2,step=3).get()
    #jetDoubletsKin=cropData(jetDoublets, ymin=0, ymax=-1, xmin=0, xmax=3)
    #jetDoubletsBTag=cropData(jetDoublets, ymin=0, ymax=-1, xmin=4, xmax=4)
    #jetDoubletsQG=cropData(jetDoublets, ymin=0, ymax=-1, xmin=5, xmax=5)
    #jetDoubletsDr=DeltaR()(jetDoubletsKin)

    #ordering of scalars
    #jetDoubletsBTag=SequentialSort(k=2, colIdx=0)(jetDoubletsBTag)
    #jetDoubletsQG=SequentialSort(k=2, colIdx=0)(jetDoubletsQG)
    #jetDoubletsDr=SequentialSort(k=2, colIdx=0, reverse=True)(jetDoubletsDr)

    #top 4 vectors, sorted by top mass ================================================
    #hadTopsKin=SumCombinatorial4VBlocks(jetTripletsKin, 3).get()
    hadTopsKin=Convert4VBlocks(jetTripletsKin,cartToPEPM=False).get()
    hadTopsKin=SumCombinatorial4VBlocks(hadTopsKin, 3).get()
    hadTopsKin=Convert4VBlocks(hadTopsKin, cartToPEPM=True).get()
    hadTopsBTagMax=SequentialReduceOperation(operation="max",k=3)(jetTripletsBTag)
    hadTopsBTagAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsBTag)
    hadTopsQGAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsQG)

    tmpJetTriplet=SequentialSort(k=3, colIdx=4)(jetTriplets)
    tmpJetTriplet=Reshape((-1,3,6))(tmpJetTriplet)
    tmpWs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=1,xmax=2))
    tmpWs=cropData(tmpWs,ymin=0, ymax=-1, xmin=0, xmax=3)

    #tmpWs=SumCombinatorial4VBlocks(tmpWs,2).get()
    tmpWs=Convert4VBlocks(tmpWs, cartToPEPM=False).get()
    tmpWs=SumCombinatorial4VBlocks(tmpWs,2).get()
    tmpWs=Convert4VBlocks( tmpWs , cartToPEPM=True).get()

    tmpBs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=0,xmax=0))
    tmpBs=cropData(tmpBs,ymin=0, ymax=-1, xmin=0, xmax=3)
    tmpComb=Concatenate(axis=2)([tmpWs,tmpBs])
    tmpComb=Reshape((-1,4))(tmpComb)
    hadTopDrMaxBW=DeltaR()(tmpComb)
    
    hadTops=Concatenate()([hadTopsKin,hadTopsBTagMax,hadTopsBTagAvg,hadTopsQGAvg,hadTopDrMaxBW])
    hadTops=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(hadTops)

    #do not uncomment!!!
    ##doublets 4 vectors, sorted by W mass =============================================
    ##hadTWsKin=SumCombinatorial4VBlocks(jetDoubletsKin,2).get()
    ##hadTWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetDoubletsBTag)
    ##hadTWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsBTag)
    ##hadTWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetDoubletsBTag)
    ##hadTWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsQG)
    ##hadTWsDr=DeltaR()(jetDoubletsKin)
    ##hadTWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg, hadWsDr])
    ##hadTWs=SequentialSort(k=3,colIdx=3,reverse=True, sortByClosestValue=True, target=80.4)(hadWs)

    #hadronic Ws 4 vector, sorted by W mass ===============================================
    #hadWsKin=SumCombinatorial4VBlocks(jetHadWDoubletsKin,2).get()
    hadWsKin=Convert4VBlocks(jetHadWDoubletsKin,cartToPEPM=False).get()
    hadWsKin=SumCombinatorial4VBlocks(hadWsKin,2).get()
    hadWsKin=Convert4VBlocks(hadWsKin, cartToPEPM=True).get()
    hadWsDr=DeltaR()(jetHadWDoubletsKin)
    hadWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetHadWDoubletsBTag)
    hadWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsQG)
    hadWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg,hadWsDr])
    hadWs=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=80.4)(hadWs)
    
    #semileptonic tops ================================================================
    multiWs=UpSampling1D(nJets)(inputW)
    #multiWs=UpSampling1D(nJets)(inputWCart)
    jetsForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=0,xmax=3)
    bScoreForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=4,xmax=4)
    semiLepTops=Concatenate(axis=2)([multiWs,jetsForsemiLepTops])
    semiLepTops=Reshape((-1,4))(semiLepTops)
    semiLepTopsDr=DeltaR()(semiLepTops)

    #semiLepTops=SumCombinatorial4VBlocks(semiLepTops,2).get()
    semiLepTops=Convert4VBlocks(semiLepTops, cartToPEPM=False).get()
    semiLepTops=SumCombinatorial4VBlocks(semiLepTops,2).get()
    semiLepTops=Convert4VBlocks(semiLepTops, cartToPEPM=True).get()
    
    semiLepTops=Concatenate()([semiLepTops, semiLepTopsDr, bScoreForsemiLepTops ])
    semiLepTops=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(semiLepTops)
    
    #dR between Z and tops ===========================================================
    multiZsSemiLep=cropData(inputZ,ymin=0,xmin=0,xmax=3)
    multiZsSemiLep=UpSampling1D(nJets)(multiZsSemiLep)
    redSLTops=cropData(semiLepTops,ymin=0,xmin=0,xmax=3)
    ZSLTops=Concatenate(axis=2)([multiZsSemiLep,redSLTops])
    ZSLTops=Reshape((-1,4))(ZSLTops)
    ZSLTopsDr=DeltaR()(ZSLTops)

    multiZsHad=cropData(inputZ,ymin=0,xmin=0,xmax=3)
    multiZsHad=UpSampling1D(hadTops._keras_shape[1])(multiZsHad)
    redHTops=cropData(hadTops,ymin=0,xmin=0,xmax=3)
    ZHTops=Concatenate(axis=2)([multiZsHad,redHTops])
    ZHTops=Reshape((-1,4))(ZHTops)
    ZHTopsDr=DeltaR()(ZHTops)

    #concatenation of the dR and the tops =============================================
    #print hadTops._keras_shape, ZHTopsDr._keras_shape
    hadTops=Concatenate()([hadTops, ZHTopsDr])
    semiLepTops=Concatenate()([semiLepTops, ZSLTopsDr])

    #jet btagging
    inputJetsBTag=cropData(inputVals,ymin=4,ymax=4+(nJets-1), xmin=4, xmax=4)
    inputJetsBTag=Sort(colIdx=0, reverse=False, sortByClosestValue=False, target=None)(inputJetsBTag)


    if prepareData:
        dataHadTops=Reshape((120,9,1))(hadTops)
        dataSemiLepTops=ZeroPadding2D( ((0,110),(0,2)) )(Reshape((10,7,1))(semiLepTops))
        dataHadWs=ZeroPadding2D( ((0,75),(0,0)) )(Reshape((45,9,1))(hadWs))
        dataJets=ZeroPadding2D( ((0,110),(0,3)) )(Reshape((10,6,1))(inputJets))
        dataJetsBTag=ZeroPadding2D( ((0,110),(0,8)) )(Reshape((10,1,1))(inputJetsBTag))
        dataRawLeps=ZeroPadding2D( ((0,116),(0,5)) )(Reshape((4,4,1))(inputRawLeps))
        dataEWK=ZeroPadding2D( ((0,116),(0,5)) )(Reshape((4,4,1))(inputEWK))
        
        mergedPreComp=Reshape((840,9))(Concatenate(axis=1)([dataHadTops,
                                                            dataSemiLepTops,
                                                            dataHadWs,
                                                            dataJets,
                                                            dataJetsBTag,
                                                            dataRawLeps,
                                                            dataEWK]))
        #print "-->>",mergedPreComp._keras_shape
        model = Model(inputs=inputVals,outputs=mergedPreComp)
        return model
    #sys.exit(0)
    
    ##=======================================
    ## Now active layers ====================
    ##=======================================

    ##hadronic tops
    hadTopPart=superLSTMLayer(hadTops,"htop_lstm1",64/f, return_sequence=True)
    hadTopPart=superLSTMLayer(hadTopPart,"htop_lstm2",64/f)

    ##semi leptonic tops
    semiLepTopPart=superLSTMLayer(semiLepTops,"sltop_lstm1",64/f, return_sequence=True)
    semiLepTopPart=superLSTMLayer(semiLepTopPart,"sltop_lstm2",64/f)

    ##hadronic Ws
    hadWsPart=superLSTMLayer(hadWs,"hW_lstm1",64/f, return_sequence=True)
    hadWsPart=superLSTMLayer(hadWsPart,"hW_lstm2",64/f)

    ##all jets
    jetPart=superLSTMLayer(inputJets,"jet_lstm1",64/f, return_sequence=True)
    jetPart=superLSTMLayer(jetPart,"jet_lstm2",64/f)

    jetBTagPart=superLSTMLayer(inputJetsBTag,"btag_lstm1",64/f, return_sequence=True)
    jetBTagPart=superLSTMLayer(jetBTagPart,"btag_lstm2",64/f)
    
    ##all leptons
    leptonPart=superLSTMLayer(inputRawLeps,"lep_lstm1",64/f, return_sequence=True)
    leptonPart=superLSTMLayer(leptonPart,"lep_lstm2",64/f)

    ##all leptons
    ewkPart=superLSTMLayer(inputEWK,"ewk_lstm1",64/f, return_sequence=True)
    ewkPart=superLSTMLayer(ewkPart,"ewk_lstm2",64/f)

    ##general concatenation
    mergingAll=Concatenate()([hadTopPart,semiLepTopPart,hadWsPart,jetPart,jetBTagPart,leptonPart,ewkPart])
    
    dense=superDenseLayer(mergingAll, "d1", nNodes=256/f)
    dense=superDenseLayer(dense, "d2", nNodes=256/f)
    dense=superDenseLayer(dense, "d3", nNodes=128/f)
    dense=superDenseLayer(dense, "d4", nNodes=128/f)
    
    finalOutput=Dense(nOut, activation='softmax')(dense)
    
    #model = Model(inputs=inputVals, outputs=hadWsKin)
    model = Model(inputs=inputVals,outputs=finalOutput)   
    #model.summary()

    return model


def prepareMergedH5File(nJMax, background="WZ"):
    print("prepare file merging")
    bkg=None
    nEvtSig=200000
    class_weights={}
    if background=="WZ": #200000
        bkg=loadEventDataset(fileLoc+"/WZTo3LNu_amcatnlo"+ext,nJMax=nJMax, n=130000,jecUnc=0)
        nEvtSig=170000
    if background=="tZq":
        bkg=loadEventDataset(fileLoc+"/tZq_ll"+ext,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=170000
    if background=="multi":
        bkg1=loadEventDataset(fileLoc+"/WZTo3LNu_amcatnlo"+ext   ,nJMax=nJMax, n=130000,jecUnc=0)
        bkg2=loadEventDataset(fileLoc+"/tZq_ll"+ext              ,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=170000

    eventsTTZ=loadEventDataset(fileLoc+"/TTZToLLNuNu"+ext,nJMax=nJMax, n=nEvtSig,jecUnc=0) #up to 150k

    if background=="test":
        bkg=loadEventDataset(fileLoc+"/WZTo3LNu_amcatnlo"+ext,nJMax=nJMax, n=1,jecUnc=0)
        nEvtSig=1

    print("Assigning flags and shuffling...")
    if background!="multi":
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg], True,0.9) #eventsWZ,eventsTTH,
        emax=float(max([len(eventsTTZ), len(bkg) ]))
        class_weights={0:emax/len(eventsTTZ),
                       1:emax/len(bkg)}
    else:
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg1,bkg2], True,0.9) #,bkg3,bkg4
        emax=float(max([len(eventsTTZ), len(bkg1), len(bkg2) ])) #, len(bkg3), len(bkg4)
        class_weights={0:emax/len(eventsTTZ),
                       1:emax/len(bkg1),
                       2:emax/len(bkg2),
                       #3:emax/len(bkg3),
                       #4:emax/len(bkg4)
        }
    print("Data ready...")    
        
    with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/Fall17_TTZvs"+background+'.h5', 'w') as h5f:
        h5f.create_dataset('data', data=train, compression="gzip")
        h5f.create_dataset('decs', data=decTrain, compression="gzip")
        g = h5f.create_group('weights')
        g.create_dataset("idxs",data=[int(k) for k in class_weights.keys()])
        g.create_dataset("values",data=[class_weights[k] for k in class_weights.keys()])
    with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/Fall17_TTZvs"+background+'_test.h5', 'w') as h5f:
        h5f.create_dataset('data', data=test, compression="gzip")
        h5f.create_dataset('decs', data=decTest, compression="gzip")
        g = h5f.create_group('weights')
        g.create_dataset("idxs",data=[int(k) for k in class_weights.keys()])
        g.create_dataset("values",data=[class_weights[k] for k in class_weights.keys()])
        #for k in class_weights.keys():
        #    h5f.create_dataset(str(k), data=[class_weights[k]], compression="gzip")

    #print decTrain.shape, train.shape
    #sys.exit(0)
    #now pre-conmputing the data inputs
  
    ### preprocessed data =========================================================
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    model=makeModelV2(nJMax, len(decTrain[0]),2,prepareData=True)
    #with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/preProcessed_Fall17_TTZvs"+background+'.h5', 'w') as h5f:
    #    x=np.reshape(train[0], (1, train.shape[1], train.shape[2]))
    #    u=model.predict(x)
    #    #print u
    #    h5f.create_dataset('data', data=u, maxshape=(None, 840, 9), compression="gzip")
    #    h5f.create_dataset('decs', data=decTrain, compression="gzip")
    #    g = h5f.create_group('weights')
    #    g.create_dataset("idxs",data=[int(k) for k in class_weights.keys()])
    #    g.create_dataset("values",data=[class_weights[k] for k in class_weights.keys()])

            
    batchSize=5000
    for j,b in enumerate(batch(train, batchSize)):
        sys.stdout.write("\r{} / {} writing (train)".format(j, len(train) // batchSize))
        sys.stdout.flush()
        data=[]
        for k,i in enumerate(b):
            #print k
            x=np.reshape(i, (1, train.shape[1], train.shape[2]))
            data.extend(model.predict(x))
        #print data[0].shape
        data=np.array(data)
        #print data.shape
        if j==0:
            with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/preProcessed_Fall17_TTZvs"+background+'.h5', 'w') as h5f:
                h5f.create_dataset('data', data=data, maxshape=(None, 840, 9), compression="gzip")
                h5f.create_dataset('decs', data=decTrain, compression="gzip")
                g = h5f.create_group('weights')
                g.create_dataset("idxs",data=[int(k) for k in class_weights.keys()])
                g.create_dataset("values",data=[class_weights[k] for k in class_weights.keys()])
        else:
            with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/preProcessed_Fall17_TTZvs"+background+'.h5', 'a') as h5f:
                h5f["data"].resize((h5f["data"].shape[0] + data.shape[0]), axis = 0)
                h5f["data"][-data.shape[0]:] = data

    #test
    for j,b in enumerate(batch(test, batchSize)):
        sys.stdout.write("\r{} / {} writing (test)".format(j, len(test) // batchSize))
        sys.stdout.flush()
        data=[]
        for k,i in enumerate(b):
            #print k
            x=np.reshape(i, (1, test.shape[1], test.shape[2]))
            data.extend(model.predict(x))
        #print data[0].shape
        data=np.array(data)
        #print data.shape
        if j==0:
            with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/preProcessed_Fall17_TTZvs"+background+'_test.h5', 'w') as h5f:
                h5f.create_dataset('data', data=data, maxshape=(None, 840, 9), compression="gzip")
                h5f.create_dataset('decs', data=decTest, compression="gzip")
                g = h5f.create_group('weights')
                g.create_dataset("idxs",data=[int(k) for k in class_weights.keys()])
                g.create_dataset("values",data=[class_weights[k] for k in class_weights.keys()])
        else:
            with h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/preProcessed_Fall17_TTZvs"+background+'_test.h5', 'a') as h5f:
                h5f["data"].resize((h5f["data"].shape[0] + data.shape[0]), axis = 0)
                h5f["data"][-data.shape[0]:] = data


  
def generate_batches_from_hdf5_file(filepath, batchsize):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    #dimensions = (batchsize, 28, 28, 1) # 28x28 pixel, one channel
    while 1:
        f = h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/"+filepath+'.h5', "r")
        filesize = len(f['decs'])

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (filesize - batchsize):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f['data'][n_entries : n_entries + batchsize]
            #xs = np.reshape(xs, dimensions).astype('float32')

            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
            ys = f['decs'][n_entries:n_entries+batchsize]
            #ys = np.array(np.zeros((batchsize, 2))) # data with 2 different classes (e.g. dog or cat)

            # Select the labels that we want to use, e.g. is dog/cat
            #for c, y_val in enumerate(y_values):
            #    ys[c] = encode_targets(y_val, class_type='dog_vs_cat') # returns categorical labels [0,1], [1,0]

            # we have read one more batch from this file
            n_entries += batchsize
            #print "  ",n_entries, n_entries + batchsize#xs, ys
            yield (xs, ys)
        f.close()

        
def fit(nJMax,background,f,batchSize=2000):

    class_weights={}
    ftrain = h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/Fall17_TTZvs"+background+'.h5', "r")
    weights = ftrain["weights"]

    for i,key in enumerate(weights['idxs']):
        class_weights[key]=weights['values'][i]

    sample_size=len(ftrain['decs'])
    print(class_weights)

    ftest = h5py.File("/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt/dataFiles/Fall17_TTZvs"+background+'_test.h5', "r")
    sample_size_test=len(ftrain['decs'])
    #sys.exit(0)
    
    model = makeModelV2(nJMax=nJMax, nOut=len(ftrain['decs'][0]), f=f)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.01)
    
    model.fit_generator(#generator(10),
        generate_batches_from_hdf5_file("Fall17_TTZvs"+background,batchSize),
        steps_per_epoch = sample_size // batchSize , epochs = 100, verbose=1,
        callbacks=[early_stop,reduce_lr],
        validation_data=generate_batches_from_hdf5_file("Fall17_TTZvs"+background+"_test",batchSize),
        validation_steps=sample_size_test // batchSize,
        max_queue_size=1,
        class_weight=class_weights,
        workers=1,
        use_multiprocessing=False,
        shuffle=False,
        #initial_epoch=i
    )

        
        
def main(nJMax, v="1.1", background="WZ", f=2):

    bkg=None
    nEvtSig=200000
    class_weights={}
    if background=="WZ": #200000
        bkg=loadEventDataset(fileLoc+"/WZTo3LNu_amcatnlo"+ext,nJMax=nJMax, n=130000,jecUnc=0)
        #bkg1=loadEventDataset(fileLoc+"/WZTo3LNu"+ext,nJMax=nJMax, n=100000,jecUnc=0)
        #bkg.extend(bkg1)
        nEvtSig=170000
    if background=="ttH":
        bkg=loadEventDataset(fileLoc+"/TTHnobb_mWCutfix_ext"+ext,nJMax=nJMax, n=21000,jecUnc=0)
        nEvtSig=21000
    if background=="tZq":
        bkg=loadEventDataset(fileLoc+"/tZq_ll"+ext,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=170000
    if background=="tZW":
        bkg=loadEventDataset(fileLoc+"/tZW_ll_ext"+ext,nJMax=nJMax, n=100000,jecUnc=0)
        nEvtSig=200000
        #nEvtSig=4000
    if background=="multi":
        bkg1=loadEventDataset(fileLoc+"/WZTo3LNu_amcatnlo"+ext   ,nJMax=nJMax, n=130000,jecUnc=0)
        bkg2=loadEventDataset(fileLoc+"/tZq_ll"+ext              ,nJMax=nJMax, n=150000,jecUnc=0)
        #bkg2=loadEventDataset(fileLoc+"/TTHnobb_mWCutfix_ext"+ext,nJMax=nJMax, n=21000,jecUnc=0)
        #bkg4=loadEventDataset(fileLoc+"/tZW_ll_ext"+ext          ,nJMax=nJMax, n=80000,jecUnc=0)
        nEvtSig=170000

    eventsTTZ=None
    if background!="tZW":
        eventsTTZ=loadEventDataset(fileLoc+"/TTZToLLNuNu"+ext,nJMax=nJMax, n=nEvtSig,jecUnc=0) #up to 150k
    else:
        eventsTTZ=loadEventDataset(fileLoc+"/TTZ_LO"+ext,nJMax=nJMax, n=nEvtSig,jecUnc=0) #up to 150k
        
    #print np.array(eventsTTZ)
    #print np.array(eventsTTZ).shape

    print("total ttZ events : "+str(len(eventsTTZ)))
    if bkg!=None:
        print("total bkg events : "+str(len(bkg)))
    #sys.exit(0)
    
    print("Assigning flags and shuffling...")
   
    if background!="multi":
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg], True,0.9) #eventsWZ,eventsTTH,
        emax=float(max([len(eventsTTZ), len(bkg) ]))
        class_weights={0:emax/len(eventsTTZ),
                       1:emax/len(bkg)}
        
    else:
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg1,bkg2], True,0.9) #,bkg3,bkg4
        emax=float(max([len(eventsTTZ), len(bkg1), len(bkg2) ])) #, len(bkg3), len(bkg4)
        class_weights={0:emax/len(eventsTTZ),
                       1:emax/len(bkg1),
                       2:emax/len(bkg2),
                       #3:emax/len(bkg3),
                       #4:emax/len(bkg4)
        }
    print("Data ready...")

    print(class_weights)
    #sys.exit(0)
    #weights=np.zeros(len(train))
    #for i,d in enumerate(decTrain):
    #    if d[0]==0:
    #        weights[i]=len(decTrain)/len(eventsTTZ)
    #    else:
    #        weights[i]=len(decTrain)/len(bkg)
        
   
    model=makeModelV2(nJMax=nJMax, nOut=len(decTest[0]), f=f)
    #model=makeModelV1(nJMax=nJMax, nOut=len(decTest[0]),version=v, f=f)
    #from keras.utils import plot_model
    #plot_model(model,show_shapes=True, to_file='RNNTTZ_'+background+'_v'+v+'.png')

    print("model ready ====================================================")
    #print(model.get_config())
    #print("================================================================")

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.01)
    history=model.fit(train,decTrain,validation_data=(test,decTest),
                      #sample_weight=weights,
                      class_weight=class_weights if len(class_weights)!=0 else None,
                      shuffle=True,
                      verbose=1,epochs=100,batch_size=2000,callbacks=[early_stop,reduce_lr])
    
    model.save(userDir+'/trainings/TiamattZ_'+background+'_v'+v+".h5")

    model.save_weights(userDir+'/trainings/TiamattZ_'+background+'_v'+v+"_weights.h5")
    model_json = model.to_json()
    with open(userDir+'/trainings/TiamattZ_'+background+'_v'+v+".json", "w") as json_file:
        json_file.write(model_json)

    host=socket.gethostname()
    if host=="mmarionn-eth-laptop":
        storeOutput('TiamattZ_'+background+'_v'+v, model, history, test, decTest)
        storeOutput('TiamattZtrainOutput_'+background+'_v'+v, model, history, train, decTrain)

def test():

    nJMax=6
    eventsTTZ=loadEventDataset(fileLoc+"/WZTo3LNu_ext"+ext,nJMax=nJMax, n=20,jecUnc=0)
    train,test,decTrain,decTest = assignFlag([eventsTTZ,eventsTTZ], False,0.999)
    model=makeModelV1(nJMax=nJMax, nOut=len(decTest[0]),version="", f=8)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

    print decTrain.shape
    sys.exit(0)
    #print "======================="
    for k,i in enumerate(train):
        x=np.reshape(i, (1, train.shape[1], train.shape[2]))
        u=model.predict(x)
        #u=evaluateModel(train, model)
        #print k,x,"===",u

        
if __name__ == "__main__":
  

    import os, itertools
    from argparse import ArgumentParser
    from optparse import OptionParser
    #argparser = ArgumentParser(description='')
    #argparser.add_argument('mode', type="string", help='running mode (training or format)')
    parser = OptionParser(usage="%prog [options] <TREE_DIR> <OUT>")
    parser.add_option("-b", "--background", dest="bkg",  type="string", default="WZ", help="background to be trained against");
    parser.add_option("-v", "--version", dest="version",  type="string", default="3.0", help="tiamatt version");
    parser.add_option("-n", "--nJMax", dest="nJMax",  type="int", default=10, help="maximal number of jets");
    parser.add_option("-f", "--factor", dest="factor",  type="int", default=2, help="Process only this dataset (or dataset if specified multiple times): REGEXP");
    (options, args) = parser.parse_args()
    #(vals, argsvals) = argparser.parse_args()

    if len(args)!=1:
        print("please provide the running mode (one arg)")
    if args[0]=="training":
        main(options.nJMax, options.version, options.bkg, options.factor)
    elif args[0]=="format":
        prepareMergedH5File(options.nJMax, options.bkg)
    elif args[0]=="test":
        model=makeModelV2(options.nJMax,2,options.factor, True)
        model.save(userDir+'/trainings/TiamattZ_preprocessing.h5')
        #fit(options.nJMax, options.bkg, options.factor)
    
    #main(10, "3.0", background="ttH",f=2)
    #main(10, "3.0", background="tZq",f=2)
    #main(10, "3.0", background="tZW",f=2)
    #main(10, "3.0", background="WZ",f=2)

    #main(10, "2.0", background="multi",f=2)

    #main(12, "nJet7", background="tZW",f=2)

    #test()
    #makeModelV1(nJMax=10, nOut=2,version="", f=2)
    #eventsTTZ=loadEventDataset(fileLoc+"/TTZToLLNuNu_ext2.h5",nJMax=3, n=2,jecUnc=0)

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, LSTM
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from keras.utils.data_utils import get_file
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
local=False
if host=="mmarionn-eth-laptop":
    local=True
    scratch="/home/mmarionn/Documents/CMS/Computing"
    userDir="/home/mmarionn/Documents/CMS/SMStudies/TTV/NNSofttt"
    fileLoc=userDir+"/dataFiles/tiamattWFiles"
else:
    local=False
    import os
    scratch=os.environ['SCRATCH']
    userDir="/users/mmarionn/PhysProjects/TTV"
    fileLoc=scratch
    
sys.path.append(scratch+"/NeuralHEP/utils/")
#sys.path.append("/home/mmarionn/Documents/CMS/Computing/NeuralHEP/utils/")
from KerasSuperLayers import *
from NNFunctions import assignFlag#, storeOutput, evaluateModel

useROOT=True
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
    t.SetBranchStatus("Jet_btagCSV",1)
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
        lepsW=[]
        #lepW=[]
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
        lepsW.append([event.LepGood_pt[lep1], event.LepGood_eta[lep1],
                     event.LepGood_phi[lep1], 0.0005 if abs(event.LepGood_pdgId[lep1]) else 0.105,
                     0,0])
        lepsW.append([event.LepGood_pt[lep2], event.LepGood_eta[lep2],
                     event.LepGood_phi[lep2], 0.0005 if abs(event.LepGood_pdgId[lep2]) else 0.105,
                     0,0])
        met.extend([ event.met_pt, 0, event.met_phi, 0, 0, 0 ])
        
        ZCand=buildZCandidate(lepsW)
        #WCand=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsW[0],met)
        WCand3=buildWCandidate(lepsW[1],met)

        
        evt=[lepsW[0], lepsW[1], met]
        evt.extend(jets)
        evt.extend([ZCand, WCand2, WCand3 ]) # WCand,
        
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
        lepsW=[]
        #lepW=[]
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
        lepsW.append([event[kLepPt][lep1], event[kLepEta][lep1],
                     event[kLepPhi][lep1], 0.0005 if abs(event[kLepPdgId][lep1]) else 0.105,
                     0,0])
        lepsW.append([event[kLepPt][lep2], event[kLepEta][lep2],
                     event[kLepPhi][lep2], 0.0005 if abs(event[kLepPdgId][lep2]) else 0.105,
                     0,0])
        met.extend([ event[kMetPt], 0, event[kMetPhi], 0, 0, 0 ])
        
        ZCand=buildZCandidate(lepsW)
        #WCand=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsW[0],met)
        WCand3=buildWCandidate(lepsW[1],met)

        evt=[lepsW[0], lepsW[1], lepW, met]
        evt.extend(jets)
        evt.extend([ZCand, WCand2, WCand3 ])
        
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
    
    inputVals=Input(shape=(6+nJets,6))

    inputRawLeps=cropData(inputVals, ymin=0, ymax=2, xmin=0, xmax=3)
    #inputLepW1=cropData(inputVals, ymin=0, ymax=0, xmin=0, xmax=3)
    #inputLepW2=cropData(inputVals, ymin=1, ymax=1, xmin=0, xmax=3)
    #inputMet=cropData(inputVals, ymin=2, ymax=2, xmin=0, xmax=3)
    inputJets=cropData(inputVals,ymin=3,ymax=3+(nJets-1), xmin=0, xmax=5)
    
    nOffset=3+nJets
    inputZ=cropData(inputVals,ymin=nOffset,ymax=nOffset, xmin=0, xmax=3)
    inputW1=cropData(inputVals,ymin=nOffset+1,ymax=nOffset+1, xmin=0, xmax=3)
    inputW2=cropData(inputVals,ymin=nOffset+2,ymax=nOffset+2, xmin=0, xmax=3)
    
    #inputW1=cropData(inputVals,ymin=nOffset+2,ymax=nOffset+2, xmin=0, xmax=3)
    #inputW2=cropData(inputVals,ymin=nOffset+3,ymax=nOffset+3, xmin=0, xmax=3)
    inputEWK=cropData(inputVals,ymin=nOffset,ymax=nOffset+2, xmin=0, xmax=3)
    
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
    multiWs1=UpSampling1D(nJets)(inputW1)
    multiWs2=UpSampling1D(nJets)(inputW2)
    jetsForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=0,xmax=3)
    bScoreForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=4,xmax=4)

    semiLepTops1=Concatenate(axis=2)([multiWs1,jetsForsemiLepTops])
    semiLepTops1=Reshape((-1,4))(semiLepTops1)
    semiLepTops1Dr=DeltaR()(semiLepTops1)
    semiLepTops1=Convert4VBlocks(semiLepTops1, cartToPEPM=False).get()
    semiLepTops1=SumCombinatorial4VBlocks(semiLepTops1,2).get()
    semiLepTops1=Convert4VBlocks(semiLepTops1, cartToPEPM=True).get()
    semiLepTops1=Concatenate()([semiLepTops1, semiLepTops1Dr, bScoreForsemiLepTops ])
    semiLepTops1=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(semiLepTops1)

    semiLepTops2=Concatenate(axis=2)([multiWs2,jetsForsemiLepTops])
    semiLepTops2=Reshape((-1,4))(semiLepTops2)
    semiLepTops2Dr=DeltaR()(semiLepTops2)
    semiLepTops2=Convert4VBlocks(semiLepTops2, cartToPEPM=False).get()
    semiLepTops2=SumCombinatorial4VBlocks(semiLepTops2,2).get()
    semiLepTops2=Convert4VBlocks(semiLepTops2, cartToPEPM=True).get()
    semiLepTops2=Concatenate()([semiLepTops2, semiLepTops2Dr, bScoreForsemiLepTops ])
    semiLepTops2=Sort(colIdx=3,reverse=False, sortByClosestValue=True, discardZeros=True, target=172)(semiLepTops2)
    
    #dR between Z and tops ===========================================================
    redSLTops1=cropData(semiLepTops1,ymin=0,xmin=0,xmax=3)
    WSLTops1=Concatenate(axis=2)([multiWs2,redSLTops1])
    WSLTops1=Reshape((-1,4))(WSLTops1)
    WSLTops1Dr=DeltaR()(WSLTops1)

    redSLTops2=cropData(semiLepTops2,ymin=0,xmin=0,xmax=3)
    WSLTops2=Concatenate(axis=2)([multiWs1,redSLTops2])
    WSLTops2=Reshape((-1,4))(WSLTops2)
    WSLTops2Dr=DeltaR()(WSLTops2)
    
    redHTops=cropData(hadTops,ymin=0,xmin=0,xmax=3)
    multiWs1Had=UpSampling1D(hadTops._keras_shape[1])(inputW1)
    multiWs2Had=UpSampling1D(hadTops._keras_shape[1])(inputW2)
    WHTops1=Concatenate(axis=2)([multiWs1Had,redHTops])
    WHTops1=Reshape((-1,4))(WHTops1)
    WHTops1Dr=DeltaR()(WHTops1)
    WHTops2=Concatenate(axis=2)([multiWs2Had,redHTops])
    WHTops2=Reshape((-1,4))(WHTops2)
    WHTops2Dr=DeltaR()(WHTops2)

    #concatenation of the dR and the tops =============================================
    hadTops=Concatenate()([hadTops, WHTops1Dr, WHTops2Dr])
    semiLepTops1=Concatenate()([semiLepTops1, WSLTops1Dr])
    semiLepTops2=Concatenate()([semiLepTops2, WSLTops2Dr])

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
    semiLepTop1Part=superLSTMLayer(semiLepTops1,"sltop1_lstm1",64/f, return_sequence=True)
    semiLepTop1Part=superLSTMLayer(semiLepTop1Part,"sltop1_lstm2",64/f)

    semiLepTop2Part=superLSTMLayer(semiLepTops2,"sltop2_lstm1",64/f, return_sequence=True)
    semiLepTop2Part=superLSTMLayer(semiLepTop2Part,"sltop2_lstm2",64/f)

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
    mergingAll=Concatenate()([hadTopPart,semiLepTop1Part,semiLepTop2Part,hadWsPart,
                              jetPart,jetBTagPart,leptonPart,ewkPart])
    
    dense=superDenseLayer(mergingAll, "d1", nNodes=256/f)
    dense=superDenseLayer(dense, "d2", nNodes=128/f)
    dense=superDenseLayer(dense, "d3", nNodes=128/f)
    dense=superDenseLayer(dense, "d4", nNodes=128/f)
    
    finalOutput=Dense(nOut, activation='softmax')(dense)
    
    model = Model(inputs=inputVals,outputs=finalOutput)   
    model.summary()

    return model


def main(nJMax, v="1.1", background="WZ", f=2):

    bkg=None
    nEvtSig=200000
    class_weights={}
    if background=="WZ": #200000
        bkg=loadEventDataset(fileLoc+"/WZTo3LNu_ext"+ext,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=200000
    if background=="ttH":
        bkg=loadEventDataset(fileLoc+"/TTHnobb_pow"+ext,nJMax=nJMax, n=10000,jecUnc=0)
        nEvtSig=10000
    if background=="tZq":
        bkg=loadEventDataset(fileLoc+"/tZq_ll"+ext,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=200000
    if background=="tZW":
        bkg=loadEventDataset(fileLoc+"/tZW_ll_ext"+ext,nJMax=nJMax, n=200000,jecUnc=0)
        nEvtSig=200000
        #nEvtSig=4000
    if background=="multi":
        bkg1=loadEventDataset(fileLoc+"/WZTo3LNu_ext"+ext        ,nJMax=nJMax, n=100000,jecUnc=0)
        bkg2=loadEventDataset(fileLoc+"/TTHnobb_mWCutfix_ext"+ext,nJMax=nJMax, n=21000,jecUnc=0)
        bkg3=loadEventDataset(fileLoc+"/tZq_ll"+ext              ,nJMax=nJMax, n=150000,jecUnc=0)
        bkg4=loadEventDataset(fileLoc+"/tZW_ll_ext"+ext          ,nJMax=nJMax, n=80000,jecUnc=0)
        nEvtSig=150000
                
    eventsTTW=loadEventDataset(fileLoc+"/TTWToLNu_ext2_part1"+ext,nJMax=nJMax, n=nEvtSig,jecUnc=0) #up to 150k
        
    #print np.array(eventsTTZ)
    #print np.array(eventsTTZ).shape

    print("total ttZ events : "+str(len(eventsTTW)))
    if bkg!=None:
        print("total bkg events : "+str(len(bkg)))
    #sys.exit(0)
    
    print("Assigning flags and shuffling...")
   
    if background!="multi":
        train,test,decTrain,decTest = assignFlag([eventsTTW,bkg], True,0.7) #eventsWZ,eventsTTH,
        emax=float(max([len(eventsTTW), len(bkg) ]))
        class_weights={0:emax/len(eventsTTW),
                       1:emax/len(bkg)}
        
    else:
        train,test,decTrain,decTest = assignFlag([eventsTTW,bkg1,bkg2,bkg3,bkg4], True,0.7) #eventsWZ,eventsTTH,
        emax=float(max([len(eventsTTW), len(bkg1), len(bkg2), len(bkg3), len(bkg4) ]))
        class_weights={0:emax/len(eventsTTW),
                       1:emax/len(bkg1),
                       2:emax/len(bkg2),
                       3:emax/len(bkg3),
                       4:emax/len(bkg4)}
    print("Data ready...")

    print(class_weights)
    #sys.exit(0)
    #weights=np.zeros(len(train))
    #for i,d in enumerate(decTrain):
    #    if d[0]==0:
    #        weights[i]=len(decTrain)/len(eventsTTZ)
    #    else:
    #        weights[i]=len(decTrain)/len(bkg)
        
   
    model=makeModelV1(nJMax=nJMax, nOut=len(decTest[0]),version=v, f=f)
    from keras.utils import plot_model
    plot_model(model,show_shapes=True, to_file='RNNTTZ_'+background+'_v'+v+'.png')

    print("model ready ====================================================")
    #print(model.get_config())
    #print("================================================================")

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.01)
    history=model.fit(train,decTrain,validation_data=(test,decTest),
                      #sample_weight=weights,
                      class_weight=class_weights if len(class_weights)!=0 else None,
                      verbose=1,epochs=200,batch_size=100,callbacks=[early_stop,reduce_lr])
    
    model.save(userDir+'/trainings/TiamattW_'+background+'_v'+v+".h5")

    model.save_weights(userDir+'/trainings/TiamattW_'+background+'_v'+v+"_weights.h5")
    model_json = model.to_json()
    with open(userDir+'/trainings/TiamattW_'+background+'_v'+v+".json", "w") as json_file:
        json_file.write(model_json)

    if local and useROOT:
        storeOutput('TiamattW_'+background+'_v'+v, model, history, test, decTest)
    

def test():

    nJMax=6
    eventsTTZ=loadEventDataset(fileLoc+"/WZTo3LNu_ext"+ext,nJMax=nJMax, n=100000,jecUnc=0)
    train,test,decTrain,decTest = assignFlag([eventsTTZ], False,0.999)
    model=makeModelV1(nJMax=nJMax, nOut=len(decTest[0]),version="", f=8)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

    #print train
    #print "======================="
    for k,i in enumerate(train):
        x=np.reshape(i, (1, train.shape[1], train.shape[2]))
        u=model.predict(x)
        #u=evaluateModel(train, model)
        #print k,x,"===",u
    
if __name__ == "__main__":
  

    import os, itertools
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] <TREE_DIR> <OUT>")
    parser.add_option("-b", "--background", dest="bkg",  type="string", default="WZ", help="background to be trained against");
    parser.add_option("-v", "--version", dest="version",  type="string", default="3.0", help="tiamatt version");
    parser.add_option("-n", "--nJMax", dest="nJMax",  type="int", default=10, help="maximal number of jets");
    parser.add_option("-f", "--factor", dest="factor",  type="int", default=2, help="Process only this dataset (or dataset if specified multiple times): REGEXP");
    (options, args) = parser.parse_args()


    main(options.nJMax, options.version, options.bkg, options.factor)

    #main(10, "3.0", background="ttH",f=2)
    #main(10, "3.0", background="tZq",f=2)
    #main(10, "3.0", background="tZW",f=2)
    #main(10, "3.0", background="WZ",f=2)

    #main(10, "2.0", background="multi",f=2)

    #main(12, "nJet7", background="tZW",f=2)

    #test()
    #makeModelV1(nJMax=10, nOut=2,version="", f=2)
    #eventsTTZ=loadEventDataset(fileLoc+"/TTZToLLNuNu_ext2.h5",nJMax=3, n=2,jecUnc=0)

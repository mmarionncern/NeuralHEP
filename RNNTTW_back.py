from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, LSTM
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from keras.utils.data_utils import get_file
import tensorflow as tf
import simplejson
import numpy as np
import random
import sys
import ROOT

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

sys.path.append("/home/mmarionn/Documents/CMS/Computing/NeuralHEP/utils/")
from KerasSuperLayers import *
from NNFunctions import assignFlag, storeOutput


def buildZCandidate(leps):
    
    lep1=ROOT.TLorentzVector()
    lep2=ROOT.TLorentzVector()
    lep1.SetPtEtaPhiM(leps[0][0],leps[0][1],leps[0][2],0.0005 if abs(leps[0][3])==11 else 0.105)
    lep2.SetPtEtaPhiM(leps[1][0],leps[1][1],leps[1][2],0.0005 if abs(leps[1][3])==11 else 0.105)

    Z=lep1+lep2

    output=[Z.Pt(),Z.Eta(),Z.Phi(),Z.M(),0,0]
    
    return output

def buildWCandidate(lep,met):
    lep4v=ROOT.TLorentzVector()
    met4v=ROOT.TLorentzVector()
    lep4v.SetPtEtaPhiM(lep[0],lep[1],lep[2],0.0005 if abs(lep[3])==11 else 0.105)
    met4v.SetPtEtaPhiM(met[0],0,met[1],0)

    WPt=(lep4v+met4v).Pt()
    WPhi=(lep4v+met4v).Phi()
    dPhi=lep4v.DeltaPhi(met4v)
    WMT=np.sqrt(2*lep[0]*met[0]*(1-np.cos(dPhi)))

    output=[WPt,0,WPhi,WMT,0,0]
    
    return output 


def loadEventDataset(fname,nJMax,n=1000,
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
        nJets=len(event.jetIdx)
        if nJets<2:
            continue       
        #print "new event",fname, n
        for k in range(0,min(nJets,nJMax)):
            j=event.jetIdx[k]
            corr=1
            if jecUnc!=0:
                corr=event.Jet_corr_JECUp[j] if jecUnc==1 else event.Jet_corr_JECDown[j]
            pt=event.Jet_pt[j]*corr
            #print pt
            jet=[pt,event.Jet_eta[j],
                 event.Jet_phi[j],event.Jet_mass[j],event.Jet_btagCSV[j]*(1+btagVar),event.Jet_qgl[j] ]
            jets.append(jet)

        if len(jets)<nJMax:
            for j in range(len(jets),nJMax):
                jets.append([0,0,0,0,0,0]) 
            
        lep1=event.lepWIdx[0]
        lep2=event.lepWIdx[1]
        lepsW.append([event.LepGood_pt[lep1], event.LepGood_eta[lep1],
                     event.LepGood_phi[lep1], 0.0005 if abs(event.LepGood_pdgId[lep1]) else 0.105,
                     0,0])
        lepsW.append([event.LepGood_pt[lep2], event.LepGood_eta[lep2],
                     event.LepGood_phi[lep2], 0.0005 if abs(event.LepGood_pdgId[lep2]) else 0.105,
                     0,0])
        #lw=event.lepWIdx
        #lepW.extend([event.LepGood_pt[lw], event.LepGood_eta[lw],
        #             event.LepGood_phi[lw], 0.0005 if abs(event.LepGood_pdgId[lw]) else 0.105,
        #             0,0])
        met.extend([ event.met_pt, 0, event.met_phi, 0, 0, 0 ])
        
        ZCand=buildZCandidate(lepsW)
        #WCand=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsZ[0],met)
        WCand3=buildWCandidate(lepsZ[1],met)

        
        evt=[lepsZ[0], lepsZ[1],  met] # lepW,
        evt.extend(jets)
        evt.extend([ZCand, WCand2, WCand3 ]) # WCand,
        
        events.append(evt)
        n-=1
        sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        sys.stdout.flush()
        if n==0: break

    return events


def makeModelV1(nJMax,nOut,version="1.0", f=2):

    nJets=nJMax
    
    inputVals=Input(shape=(8+nJets,6))

    inputRawLeps=cropData(inputVals, ymin=0, ymax=2, xmin=0, xmax=3)
    #inputLepW1=cropData(inputVals, ymin=0, ymax=0, xmin=0, xmax=3)
    #inputLepW2=cropData(inputVals, ymin=1, ymax=1, xmin=0, xmax=3)
    #inputMet=cropData(inputVals, ymin=2, ymax=2, xmin=0, xmax=3)
    inputJets=cropData(inputVals,ymin=3,ymax=3+(nJets-1), xmin=0, xmax=5)
    print inputJets._keras_shape
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
    hadTopsKin=SumCombinatorial4VBlocks(jetTripletsKin,3).get()
    hadTopsBTagMax=SequentialReduceOperation(operation="max",k=3)(jetTripletsBTag)
    hadTopsBTagAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsBTag)
    hadTopsQGAvg=SequentialReduceOperation(operation="avg",k=3)(jetTripletsQG)

    tmpJetTriplet=SequentialSort(k=3, colIdx=4)(jetTriplets)
    tmpJetTriplet=Reshape((-1,3,6))(tmpJetTriplet)
    tmpWs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=1,xmax=2))
    tmpWs=cropData(tmpWs,ymin=0, ymax=-1, xmin=0, xmax=3)
    tmpWs=SumCombinatorial4VBlocks(tmpWs,2).get()
    tmpBs=Reshape((-1,6))(cropData2D(tmpJetTriplet,ymin=0, ymax=-1, xmin=0,xmax=0))
    tmpBs=cropData(tmpBs,ymin=0, ymax=-1, xmin=0, xmax=3)
    tmpComb=Concatenate(axis=2)([tmpWs,tmpBs])
    tmpComb=Reshape((-1,4))(tmpComb)
    hadTopDrMaxBW=DeltaR()(tmpComb)
    
    hadTops=Concatenate()([hadTopsKin,hadTopsBTagMax,hadTopsBTagAvg,hadTopsQGAvg,hadTopDrMaxBW])
    hadTops=Sort(colIdx=3,reverse=True, sortByClosestValue=True, target=172)(hadTops)
    
    #doublets 4 vectors, sorted by W mass =============================================
    #hadTWsKin=SumCombinatorial4VBlocks(jetDoubletsKin,2).get()
    #hadTWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetDoubletsBTag)
    #hadTWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsBTag)
    #hadTWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetDoubletsBTag)
    #hadTWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetDoubletsQG)
    #hadTWsDr=DeltaR()(jetDoubletsKin)
    #hadTWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg, hadWsDr])
    #hadTWs=SequentialSort(k=3,colIdx=3,reverse=True, sortByClosestValue=True, target=80.4)(hadWs)

    #hadronic Ws 4 vector, sorted by W mass ===============================================
    hadWsKin=SumCombinatorial4VBlocks(jetHadWDoubletsKin,2).get()
    hadWsDr=DeltaR()(jetHadWDoubletsKin)
    hadWsBTagMax=SequentialReduceOperation(operation="max",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsBTag)
    hadWsBTagAbsDiff=SequentialReduceOperation(operation="-abs",k=2)(jetHadWDoubletsBTag)
    hadWsQGAvg=SequentialReduceOperation(operation="avg",k=2)(jetHadWDoubletsQG)
    hadWs=Concatenate()([hadWsKin,hadWsBTagMax,hadWsBTagAvg,hadWsBTagAbsDiff,hadWsQGAvg,hadWsDr])
    hadWs=Sort(colIdx=3,reverse=True, sortByClosestValue=True, target=80.4)(hadWs)
    
    #semileptonic tops ================================================================
    multiWs1=UpSampling1D(nJets)(inputW1)
    multiWs2=UpSampling1D(nJets)(inputW2)
    jetsForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=0,xmax=3)
    bScoreForsemiLepTops=cropData(inputJets,ymin=0,ymax=-1,xmin=4,xmax=4)
    semiLepTops1=Concatenate(axis=2)([multiWs1,jetsForsemiLepTops])
    semiLepTops1=Reshape((-1,4))(semiLepTops1)
    semiLepTops1Dr=DeltaR()(semiLepTops1)
    semiLepTops1=SumCombinatorial4VBlocks(semiLepTops1,2).get()
    semiLepTops1=Concatenate()([semiLepTops1, semiLepTops1Dr, bScoreForsemiLepTops ])
    semiLepTops1=Sort(colIdx=3,reverse=True, sortByClosestValue=True, target=172)(semiLepTops1)

    semiLepTops2=Concatenate(axis=2)([multiWs2,jetsForsemiLepTops])
    semiLepTops2=Reshape((-1,4))(semiLepTops2)
    semiLepTops2Dr=DeltaR()(semiLepTops2)
    semiLepTops2=SumCombinatorial4VBlocks(semiLepTops2,2).get()
    semiLepTops2=Concatenate()([semiLepTops2, semiLepTops2Dr, bScoreForsemiLepTops ])
    semiLepTops2=Sort(colIdx=3,reverse=True, sortByClosestValue=True, target=172)(semiLepTops2)
    
    
    #dR between Ws and tops ===========================================================
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
    #print hadTops._keras_shape, ZHTopsDr._keras_shape
    hadTops=Concatenate()([hadTops, WHTops1Dr, WHTops2Dr])
    #print semiLepTops._keras_shape, ZSLTopsDr._keras_shape
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
    nEvtSig=100000
    if background=="WZ":
        bkg=loadEventDataset("dataFiles/ttV_v2/WZTo3LNu_ext.root",nJMax=nJMax, n=100000,jecUnc=0)
        nEvtSig=100000
    if background=="ttH":
        bkg=loadEventDataset("dataFiles/ttV_v2/TTHnobb_mWCutfix_ext.root",nJMax=nJMax, n=21000,jecUnc=0)
        nEvtSig=21000
    if background=="tZq":
        bkg=loadEventDataset("dataFiles/ttV_v2/tZq_ll.root",nJMax=nJMax, n=150000,jecUnc=0)
        nEvtSig=150000
    if background=="tZW":
        bkg=loadEventDataset("dataFiles/ttV_v2/tZW_ll_ext.root",nJMax=nJMax, n=80000,jecUnc=0)
        nEvtSig=80000
        #nEvtSig=4000
    if background=="multi":
        bkg1=loadEventDataset("dataFiles/ttV_v2/WZTo3LNu_ext.root"        ,nJMax=nJMax, n=100000,jecUnc=0)
        bkg2=loadEventDataset("dataFiles/ttV_v2/TTHnobb_mWCutfix_ext.root",nJMax=nJMax, n=21000,jecUnc=0)
        bkg3=loadEventDataset("dataFiles/ttV_v2/tZq_ll.root"              ,nJMax=nJMax, n=150000,jecUnc=0)
        bkg4=loadEventDataset("dataFiles/ttV_v2/tZW_ll_ext.root"          ,nJMax=nJMax, n=80000,jecUnc=0)
        nEvtSig=150000
                
    #eventsTTZ=loadEventDataset("dataFiles/ttV_v2/TTZToLLNuNu_ext2.root",nJMax=nJMax, n=nEvtSig,jecUnc=0)
    eventsTTW=loadEventDataset("dataFiles/ttV_v2/",nJMax=nJMax, n=nEvtSig, jecUnc=0)
        
    #print np.array(eventsTTZ)

    print("total ttZ events : "+str(len(eventsTTZ)))
    if bkg!=None:
        print("total bkg events : "+str(len(bkg)))
    #sys.exit(0)
    
    print("Assigning flags and shuffling...")
    class_weights={}
    if background!="multi":
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg], True,0.7) #eventsWZ,eventsTTH,
    else:
        train,test,decTrain,decTest = assignFlag([eventsTTZ,bkg1,bkg2,bkg3,bkg4], True,0.7) #eventsWZ,eventsTTH,
        emax=float(max([len(eventsTTZ), len(bkg1), len(bkg2), len(bkg3), len(bkg4) ]))
        class_weights={0:emax/len(eventsTTZ),
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
                      verbose=1,epochs=200,batch_size=2000,callbacks=[early_stop,reduce_lr])
    
    model.save('trainings/TiamattZ_'+background+'_v'+v+".h5")

    #model.save('TiamattZ_'+background+'_v'+v+".h5")
    model.save_weights('trainings/TiamattZ_'+background+'_v'+v+"_weights.h5")
    model_json = model.to_json()
    with open('trainings/TiamattZ_'+background+'_v'+v+".json", "w") as json_file:
        json_file.write(model_json)

    storeOutput('TiamattZ_'+background+'_v'+v, model, history, test, decTest)
    

    
if __name__ == "__main__":
  
    #main(10, "2.0", background="ttH",f=2)
    #main(10, "2.0", background="tZq",f=2)
    #main(10, "2.0", background="tZW",f=2)
    #main(10, "2.0", background="WZ",f=2)

    #main(10, "2.0", background="multi",f=2)

    #main(12, "nJet7", background="tZW",f=2)
    makeModelV1(10,2)

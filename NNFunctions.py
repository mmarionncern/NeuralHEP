from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from keras.utils.data_utils import get_file
#import simplejson
import numpy as np
import random
import sys
import socket
host=socket.gethostname()
if host=="mmarionn-eth-laptop":
    import ROOT
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def evaluateModel(event, model):
    x=np.reshape(event, (1, event.shape[0], event.shape[1]))
    prediction=model.predict(x,verbose=0)
    return prediction

def loadModel(weightFile, customLayers):


    model = load_model(weightFile, custom_objects=customLayers)
    return model
    

def loadModelFromJSON(theFile,customLayers):

    # load json and create model
    json_file = open(theFile+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects=customLayers)

    # load weights into new model
    loaded_model.load_weights(theFile+"_weights.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    return loaded_model


    
def assignFlag(samples, shuffling=True, frac=0.6):

    trainSizes=[int(len(s)*frac) for s in samples]
    testSizes=[int(len(s)-trainSizes[i]) for i,s in enumerate(samples)]
    #print trainSizes, testSizes
    #assign the decision flag
    allEventsTrain=[]
    allEventsTest=[]    
    stdDec=[]
    for i in range(0,len(samples)):
        stdDec.append(0.)

    for i,s in enumerate(samples):
        dec=list(stdDec)
        dec[i]=1.
        #print i,"-------",s,'--------------------',s[0:trainSizes[i]]
        for e in s[0:trainSizes[i]]:
            #print i,'----',dec,"----",e
            allEventsTrain.append([e,dec])
        for e in s[trainSizes[i]:]:
            allEventsTest.append([e,dec])
            
    sampleArray=[]
    decisionArray=[]
    if len(allEventsTrain)!=0:
        schuffled=np.array(allEventsTrain)
        if shuffling:
            np.random.seed(10)
            print("Shuffling...")
            np.random.shuffle(schuffled)
            print("Shuffled...")
        k=False
        nmax=len(schuffled)
        n=nmax
        for evt in schuffled:
            #sys.stdout.write("\r%d%% decision addition remaining (train)" % (n*100/nmax))
            #sys.stdout.flush()
            #print '---------------->>>',evt[0],'----',evt[1],'-----',sampleArray
            #print '---------------->>>',type(evt[0]),'----',type(evt[1]),'-----',type(sampleArray)
            if type(evt[0]).__module__ == np.__name__:
                sampleArray.append(evt[0].tolist())
                decisionArray.append(evt[1].tolist())
            else:
                sampleArray.append(evt[0])
                decisionArray.append(evt[1])
            
            n-=1

       
    train=np.array(sampleArray)
    decisionTrain=np.array(decisionArray)

    #print "========================="
    #print sampleArray
    #print "========================="
    #print train
    
    sampleArrayTest=[]
    decisionArrayTest=[]
    if len(allEventsTest)!=0:
        schuffledTest=np.array(allEventsTest)
        np.random.shuffle(schuffledTest)
        nmax=len(schuffledTest)
        n=nmax
        for evt in schuffledTest:
            #print evt
            #sys.stdout.write("\r%d%% decision addition remaining (test)" % (n*100/nmax))
            #sys.stdout.flush()
            if type(evt[0]).__module__ == np.__name__:
                sampleArrayTest.append(evt[0].tolist())
                decisionArrayTest.append(evt[1].tolist())
            else:  
                sampleArrayTest.append(evt[0])
                decisionArrayTest.append(evt[1])
            n-=1
        
    test=np.array(sampleArrayTest)
    decisionTest=np.array(decisionArrayTest)
        

    return train, test, decisionTrain, decisionTest
"""
#def deltaR(j1,j2):
#
#    jet1=ROOT.TLorentzVector()
#    jet2=ROOT.TLorentzVector()
#    jet1.SetPtEtaPhiM(j1[0],j1[1],j1[2],j1[3])
#    jet2.SetPtEtaPhiM(j2[0],j2[1],j2[2],j2[3])

#    return jet1.DeltaR(jet2)


#def jetPairing(jets,refMass=80.4,istop=False,jetInfos=False,addW=False,nPMax=4):
#    #nPMax=
#    pairs=[]
#  
#    for i1,j1 in enumerate(jets):
#        for i2,j2 in enumerate(jets):
#              if i2<=i1: continue
#              if j1[0]==0 or j2[0]==0: continue
#              jet1=ROOT.TLorentzVector()
#              jet2=ROOT.TLorentzVector()
#              jet1.SetPtEtaPhiM(j1[0],j1[1],j1[2],j1[3])
              jet2.SetPtEtaPhiM(j2[0],j2[1],j2[2],j2[3])
              if istop:
                  if i2<=i1: continue
                  if j1[0]==0 or j2[0]==0: continue
                  for i3,j3 in enumerate(jets):
                      jet3=ROOT.TLorentzVector()
                      jet3.SetPtEtaPhiM(j3[0],j3[1],j3[2],j3[3])
                      if i3<=i2: continue
                      if j3[0]==0: continue
                      thePair=[jet1+jet2+jet3]
                      if jetInfos:
                          order=[j1,j2,j3]
                          order.sort(key= lambda j:1.-j[5])
                          if addW: # adding the best W
                              ws=jetPairing(order,refMass=80.4,istop=False,jetInfos=True,nPMax=3)
                              #print ">>>",np.array(order)
                              order.insert(0,ws[0][:4]+[ws[0][9],ws[0][15]])
                              #print ">>",np.array(order)
                          thePair.append(order)
                      pairs.append(thePair)
              else:
                  thePair=[jet1+jet2]
                  if jetInfos:
                      order=[j1,j2]
                      order.sort(key= lambda j:1.-j[5])
                      #print "<<",np.array(order), thePair[0].Pt(), thePair[0].M() 
                      thePair.append(order)
                  pairs.append(thePair)
    
    pairs.sort(key=lambda x: abs(x[0].M()-refMass))

    #convert the pairs to list of attributes
    cPairs=[]
    for i,p in enumerate(pairs):
        #print ">>",p
        cPairs.append([p[0].Pt(),p[0].Eta(),p[0].Phi(),p[0].M()])
        #print "-",np.array(cPairs)
        if istop and addW:
            cPairs[-1].extend(p[1][0])
            #print "--",np.array(cPairs)
        if jetInfos:
            cPairs[-1].extend(p[1][int(istop and addW)])
            cPairs[-1].extend(p[1][int(istop and addW)+1])
            #print "---",np.array(cPairs)
            if istop:
                cPairs[-1].extend(p[1][-1])
                #print "----",np.array(cPairs)
            
    #if len(cPairs)<nPMax:
    #    for p in range(len(cPairs),nPMax):
    #        cPairs.append([0,0,0,0])
    #        if jetInfos:
    #            cPairs[-1].extend([0,0,0,0,0,0])
    #            cPairs[-1].extend([0,0,0,0,0,0])
    #            if istop:
    #                cPairs[-1].extend([0,0,0,0,0,0])
                    
        
    return cPairs[:nPMax]


def loadDilepton(fname,n=1000, addZ=False ):
    
    f = ROOT.TFile.Open(fname, "read")
    t = f.Get("tree")

    t.SetBranchStatus("*",0)
  
    t.SetBranchStatus("lepWIdx",1)
    t.SetBranchStatus("lepZIdx",1)
    t.SetBranchStatus("LepGood_pt",1)
    t.SetBranchStatus("LepGood_eta",1)
    t.SetBranchStatus("LepGood_phi",1)
    t.SetBranchStatus("LepGood_pdgId",1)
    
    print('Loading data '+fname+' ...')
    events=[]
    
    nmax=min(n,t.GetEntries())
    n=min(n,t.GetEntries())
    for event in t :
        lepsZ=[]
        lep1=event.lepZIdx[0]
        lep2=event.lepZIdx[1]
        lepsZ.append([event.LepGood_pt[lep1], event.LepGood_eta[lep1],
                      event.LepGood_phi[lep1], float(event.LepGood_pdgId[lep1]) ])
        lepsZ.append([event.LepGood_pt[lep2], event.LepGood_eta[lep2],
                      event.LepGood_phi[lep2],  float(event.LepGood_pdgId[lep2]) ])
        #lw=event.lepWIdx
        #lepW.extend([event.LepGood_pt[lw], event.LepGood_eta[lw],
        #             event.LepGood_phi[lw], event.LepGood_pdgId[lw],
        #             0,0])
        #met.extend([ event.met_pt, event.met_phi ])

        evt=[lepsZ[0],lepsZ[1]]
        #events.append(lepsZ)
        if addZ:
            ZCand=buildZCandidate(lepsZ)
            evt.append(ZCand[:4])
            

        events.append(evt)    
        n-=1
        sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        sys.stdout.flush()
        if n==0: break

        
    print "\nData loaded"
    print "number of events loaded:", len(events)
    return events


def loadDataset(fname,n=1000, jetInfos=False,
                jecUnc=0,addW=True,
                nPMax=4, nTMax=4 ): # wPairing=False, topPairing=False, wJetInfos=False, topJetInfos=False,

    #if wJetInfos or topJetInfos:
    #    jetInfos=True
    
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

    t.SetBranchStatus("nGenPart",1)
    t.SetBranchStatus("GenPart_pdgId",1)
    t.SetBranchStatus("GenPart_pt",1)
    t.SetBranchStatus("GenPart_eta",1)
    t.SetBranchStatus("GenPart_phi",1)
    t.SetBranchStatus("GenPart_mass",1)
    t.SetBranchStatus("GenPart_motherId",1)
    
    print('Loading data '+fname+' ...')
    events=[]
    eventsPairs=[]
    eventsTriplets=[]
    nJMax=10
    #nPMax=nPMax if nPMax!=-1 else (sum(i for i in range(0,nJMax-0) ))
    nPMax=nPMax if nPMax!=-1 else (sum(i for i in range(0,nJMax-1) ))
    #print "==>>> ",nPMax
    nmax=min(n,t.GetEntries())
    n=min(n,t.GetEntries())
    for event in t :
        jets=[]
        nJets=len(event.jetIdx)
        #print nJets,"=====================",event.nJet
        #for k in range(0,event.nJet):
        #    print k,event.Jet_pt[k],event.Jet_eta[k],event.Jet_phi[k],event.Jet_mass[k]
        if nJets<3:
            continue        

        for k in range(0,min(nJets,nJMax)):
            j=event.jetIdx[k]
            corr=1
            if jecUnc!=0:
                corr=event.Jet_corr_JECUp[j] if jecUnc==1 else event.Jet_corr_JECDown[j]
            pt=event.Jet_pt[j]*corr
            jet=[pt,event.Jet_eta[j],
                 event.Jet_phi[j],event.Jet_mass[j] ]
            if jetInfos:
                jet.extend([event.Jet_qgl[j],event.Jet_btagCSV[j] ])
            jets.append(jet)
            
        #events.append(jets)

        #for i in range(0,event.nGenPart):
        #    if abs(event.GenPart_pdgId[i])==6:
        #        print "top : ",event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_mass[i], event.GenPart_pdgId[i]
         #   if abs(event.GenPart_pdgId[i])<=4:
         #       print "light : ",event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_mass[i], event.GenPart_motherId[i]
         #   if abs(event.GenPart_pdgId[i])==5:
         #       print "b : ",event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_mass[i]
         #   if abs(event.GenPart_pdgId[i])==24:
         #       print "W : ",event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_mass[i], event.GenPart_pdgId[i],event.GenPart_motherId[i]
         #   if abs(event.GenPart_pdgId[i])==11 or abs(event.GenPart_pdgId[i])==13 or abs(event.GenPart_pdgId[i])==15:
         #       print "l : ",event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_mass[i],event.GenPart_motherId[i]

                
        tops = advancedTopBuilder(jets,addW=addW,
                                  nPMax=nPMax)
        
        events.append(tops[:nTMax])
        
            
        #completion
        #if len(jets)<nJMax:
        #    for j in range(len(jets),nJMax):
        #        tmp=[0,0,0,0]
        #        if jetInfos:
        #            tmp.extend([0,0])
        #        jets.append(tmp)
                    
                
        #pairing the jets if needed
        #if wPairing:
        #    pairs=jetPairing(jets,80.4,istop=False, jetInfos=wJetInfos)
        #    eventsPairs.append(pairs)
        #if topPairing:
        #    triplets=jetPairing(jets,172,istop=True, jetInfos=topJetInfos)
        #    eventsTriplets.append(triplets)

            
        n-=1
        sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        sys.stdout.flush()
        if n==0: break

    #print "\nData loaded"
    #print "number of events loaded:", len(events)
    #return events, eventsPairs, eventsTriplets
    return events




def loadEventDataset(fname,n=1000,addTop=False,byBTag=False,
                     addHadTop=False,byBTagHadTop=False,
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
    nJMax=2
    nmax=min(n,t.GetEntries())
    n=min(n,t.GetEntries())
    for event in t:
        lepsZ=[]
        lepW=[]
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
                 event.Jet_phi[j],event.Jet_mass[j],
                 event.Jet_qgl[j],event.Jet_btagCSV[j]*(1+btagVar) ]
            jets.append(jet)

        if len(jets)<nJMax:
            for j in range(len(jets),nJMax):
                jets.append([0,0,0,0,0,0]) 
            
        pairs=jetPairing(jets,80.4,istop=False, jetInfos=True)
           
        #print pairs
        lep1=event.lepZIdx[0]
        lep2=event.lepZIdx[1]
        lepsZ.append([event.LepGood_pt[lep1], event.LepGood_eta[lep1],
                     event.LepGood_phi[lep1], event.LepGood_pdgId[lep1],
                     0,0])
        lepsZ.append([event.LepGood_pt[lep2], event.LepGood_eta[lep2],
                     event.LepGood_phi[lep2], event.LepGood_pdgId[lep2],
                     0,0])
        lw=event.lepWIdx
        lepW.extend([event.LepGood_pt[lw], event.LepGood_eta[lw],
                     event.LepGood_phi[lw], event.LepGood_pdgId[lw],
                     0,0])
        met.extend([ event.met_pt, event.met_phi ])
        
        ZCand=buildZCandidate(lepsZ)
        WCand1=buildWCandidate(lepW,met)
        WCand2=buildWCandidate(lepsZ[0],met)
        WCand3=buildWCandidate(lepsZ[1],met)

        evt=[ZCand,WCand1, WCand2, WCand3] #ZCand,WCand
        evt.extend(pairs)

        # extra hadronic top reconstructino from dijet pair and extra jet, up to 4 top candidates
        if addHadTop:
            triplets=[]
            for w in pairs:
                for j in jets:
                    triplets.append( buildTopFromHadWAndJet(w,j) )
            triplets.sort(key=lambda x: abs(x[3]-172))
            if byBTagHadTop:
                triplets.sort(key=lambda x:1-x[15])         
        
            evt.extend(triplets[:4])

        #extra top reconstruction from main W and jet, up to 4 top candidates
        if addTop:
            tops=[]
            for j in jets:
                tops.append( buildTopFromWAndJet(WCand1, j) )
            tops.sort(key=lambda x: abs(x[3]-172))
            if byBTag:
                tops.sort(key=lambda x:1-x[15])
            #print np.array(tops)
            evt.extend(tops[:4])

            
                
                
        #print evt
        events.append(evt)
        n-=1
        sys.stdout.write("\r%d%% data load remaining" % (n*100/nmax))
        sys.stdout.flush()
        if n==0: break

    return events

"""



def storeOutput(outputName, model, history, test, decTest):

    hWZ=ROOT.TH1F("WZ","WZ",101,0,1.01)
    hTTZ=ROOT.TH1F("TTZ","TTZ",101,0,1.01)
    hAcc=ROOT.TH1F("accurary train","accuracy train",500,0,500)
    hValAcc=ROOT.TH1F("accurary test","accuracy test",500,0,500)
    hLoss=ROOT.TH1F("loss train","loss train",500,0,500)
    hValLoss=ROOT.TH1F("loss test","loss test",500,0,500)

    
    #test
    for i,evt in enumerate(test):
        x=np.reshape(evt, (1, evt.shape[0], evt.shape[1]))
        prediction=model.predict(x, verbose=0)
        idx = np.argmax(prediction)
        if decTest[i][0]==1: #ttZ
            hTTZ.Fill(prediction[0][0],1)
        else:
            hWZ.Fill(prediction[0][0],1)


    hTTZ.SetLineColor(601) #blue
    hWZ.SetLineColor(633) #red
    hTTZ.SetLineWidth(2)
    hWZ.SetLineWidth(2)

    hTTZ.Scale(1./max(1.,hTTZ.Integral()) )
    hWZ.Scale(1./max(1.,hWZ.Integral()) )
    
    c= ROOT.TCanvas("c","c")
    hTTZ.Draw("hist")
    hWZ.Draw("same hist")

    leg=ROOT.TLegend(0.4,0.4,0.7,0.6)
    leg.SetFillColor(0)
    leg.SetLineColor(0)
    leg.SetShadowColor(0)
    
    leg.AddEntry(hTTZ,"ttZ","l")
    leg.AddEntry(hWZ,"WZ","l")
    
    leg.Draw()      

    for i,acc in enumerate( history.history['acc'] ):
        hAcc.Fill(i, acc)
        hValAcc.Fill(i, history.history['val_acc'][i] )
        hLoss.Fill(i, history.history['loss'][i] )
        hValLoss.Fill(i, history.history['val_loss'][i] )

    
    hAcc.SetLineColor(1)
    hValAcc.SetLineColor(2)
    hLoss.SetLineColor(3)
    hValLoss.SetLineColor(4)
    hAcc.SetLineWidth(2)
    hValAcc.SetLineWidth(2)
    hLoss.SetLineWidth(2)
    hValLoss.SetLineWidth(2)
    
    leg2=ROOT.TLegend(0.6,0.4,0.82,0.6)
    leg2.SetFillColor(0)
    leg2.SetLineColor(0)
    leg2.SetShadowColor(0)
    
    leg2.AddEntry(hAcc,"accuracy","l")
    leg2.AddEntry(hValAcc,"valacc","l")
    leg2.AddEntry(hLoss,"loss","l")
    leg2.AddEntry(hValLoss,"valloss","l")
    
    hAcc.GetYaxis().SetRangeUser(0,1.01)
    
    c2= ROOT.TCanvas("c2","c2")
    hAcc.Draw("hist")
    hValAcc.Draw("same hist")
    hLoss.Draw("same hist")
    hValLoss.Draw("same hist")
    
    leg2.Draw()
    
    ofile=ROOT.TFile("rootFiles/"+outputName+".root","recreate")
    ofile.cd()
    hTTZ.Write()
    hWZ.Write()
    c.Write()
    c2.Write()
    hAcc.Write()
    hValAcc.Write()
    hLoss.Write()
    hValLoss.Write()
    ofile.Close()



    

def buildNN(name,train, decTrain, test, decTest, nLayers=2, nNeurons=100, nEpochs=50, batch_size=5000, weights=None):

    outputName=name+"_"+str(nLayers)+"Layers_"+str(nNeurons)+"N_"+str(nEpochs)+"Epochs_batch"+str(batch_size)
    print("==================== Model : "+outputName+" =================================")
    print('Build model...')
    model = Sequential()
    
    for i in range(0,nLayers-1):
        model.add(LSTM(nNeurons, return_sequences=True, input_shape=(train.shape[1],train.shape[2])))
    
    model.add(LSTM(nNeurons, input_shape=(train.shape[1],train.shape[2])))
    model.add(Dense(decTrain.shape[1], activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.01)
    history=model.fit(train, decTrain,
                      batch_size=batch_size,
                      epochs=nEpochs,
                      sample_weight=weights,
                      callbacks=[early_stop,reduce_lr],
                      validation_data=(test, decTest))
    score, acc = model.evaluate(test, decTest,
                                batch_size=batch_size)
    print("\n")
    print('Test score:', score)
    print('Test accuracy:', acc)

    f = open('logs/'+outputName+'.log', 'w')
    f.write('Test score:'+ str(score)+'\n')
    f.write('Test accuracy:'+ str(acc)+'\n')
    f.close()
    
    ### saving the model
    model_json = model.to_json()
    with open("trainings/"+outputName+".json", "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

    ##saving the training
    model.save("trainings/"+outputName+".h5")

    return model, outputName, history

"""
 
def buildZCandidate(leps):
    
    lep1=ROOT.TLorentzVector()
    lep2=ROOT.TLorentzVector()
    lep1.SetPtEtaPhiM(leps[0][0],leps[0][1],leps[0][2],0.0005 if abs(leps[0][3])==11 else 0.105)
    lep2.SetPtEtaPhiM(leps[1][0],leps[1][1],leps[1][2],0.0005 if abs(leps[1][3])==11 else 0.105)

    Z=lep1+lep2

    output=[Z.Pt(),Z.Eta(),Z.Phi(),Z.M(),0,0]
    #output.extend([leps[0][0],leps[0][1],leps[0][2],leps[0][3],0,0])
    #output.extend([leps[1][0],leps[1][1],leps[1][2],leps[1][3],0,0])
    
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

    output=[WPt,0,WPhi,WMT]
    output.extend([lep[0],lep[1],lep[2],lep[3],0,0])
    output.extend([met[0],0,met[1],0,0,0])
    
    return output 

def buildTop(W,jet):
    output=[]
    w4v=ROOT.TLorentzVector() 
    w4v.SetPtEtaPhiM(W[0],W[1],W[2],W[3])
    jet4v=ROOT.TLorentzVector()
    jet4v.SetPtEtaPhiM(jet[0],jet[1],jet[2],jet[3])
    top4v=w4v+jet4v
    output.extend([top4v.Pt(), top4v.Eta(), top4v.Phi(), top4v.M()])
    #print np.array(output)," ==== ",np.array(W)[:4],"  ===== ", np.array(jet)[:4]
    return output

def buildTop(jet1,jet2,jet3):
    output=[]
    j1v=ROOT.TLorentzVector() 
    j1v.SetPtEtaPhiM(jet1[0],jet1[1],jet1[2],jet1[3])
    j2v=ROOT.TLorentzVector()
    j2v.SetPtEtaPhiM(jet2[0],jet2[1],jet2[2],jet2[3])
    j3v=ROOT.TLorentzVector()
    j3v.SetPtEtaPhiM(jet3[0],jet3[1],jet3[2],jet3[3])
    top4v=jet1v+jet2v+jet3v
    output.extend([top4v.Pt(), top4v.Eta(), top4v.Phi(), top4v.M()])
    #print np.array(output)," ==== ",np.array(W)[:4],"  ===== ", np.array(jet)[:4]
    return output

def buildTopFromWAndJet(W,jet):

    output=[]
    if jet[0]==0:
        return [0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
    else:
        w4v=ROOT.TLorentzVector() 
        w4v.SetPtEtaPhiM(W[0],W[1],W[2],W[3])
        jet4v=ROOT.TLorentzVector()
        jet4v.SetPtEtaPhiM(jet[0],jet[1],jet[2],jet[3])
        top4v=w4v+jet4v
        output.extend([top4v.Pt(), top4v.Eta(), top4v.Phi(), top4v.M()])
        output.extend([W[0],W[1],W[2],W[3],0,0])
        output.extend([jet[0],jet[1],jet[2],jet[3],jet[4],jet[5]])

        return output

def buildTopFromHadWAndJet(W,jet):

    output=[]
    if jet[0]==0:
        return [0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
    else:
        w4v=ROOT.TLorentzVector() 
        w4v.SetPtEtaPhiM(W[0],W[1],W[2],W[3])
        jet4v=ROOT.TLorentzVector()
        jet4v.SetPtEtaPhiM(jet[0],jet[1],jet[2],jet[3])
        top4v=w4v+jet4v
        output.extend([top4v.Pt(), top4v.Eta(), top4v.Phi(), top4v.M()])
        output.extend([W[0],W[1],W[2],W[3],W[9],W[15]])
        output.extend([jet[0],jet[1],jet[2],jet[3],jet[4],jet[5]])

        return output

def advancedTopBuilder(jets,jetInfos=True,addW=True,
                       nPMax=4):

    #ws=jetPairing(jets,refMass=80.4,jetInfos=True,istop=False,addW=addW,nPMax=nPMax)
    tops=jetPairing(jets,refMass=172,jetInfos=jetInfos,istop=True,addW=addW,nPMax=nPMax)
    #print np.array(ws)[:,:4]
    #print "========================================="
    #print np.array(tops)
    #print "=========================================",len(tops),nPMax
    if len(tops)<nPMax:
        for p in range(len(tops),nPMax):
            tops.append([0,0,0,0])
            if addW :
                tops[-1].extend([0,0,0,0,0,0])
            if jetInfos:
                tops[-1].extend([0,0,0,0,0,0])
                tops[-1].extend([0,0,0,0,0,0])
                tops[-1].extend([0,0,0,0,0,0])
    
    return tops

def advancedTopBuilder2(jets,addW=True,
                       addWJets=False,addWJetsInfos=False,
                       addBJet=False,addBJetInfos=False,
                       sortByTopMass=True, sortByWMass=True,
                       sortByBTagScore=True,nPMax=4):

    bjets=list(jets)
    bjets.sort(key=lambda j:1.-j[5])
    #print np.array(bjets)
    ws=jetPairing(jets,refMass=80.4,jetInfos=True,nPMax=nPMax)
    print "========================= ws ==================="
    print np.array(ws)[:,:4]
    print "================================================== njet",len(jets), "   nws=",len(ws)
    #print np.array(ws)[:,4]
    idxBJetInfo=-1
    
    tops=[]
    for bj in bjets:
        tmptops=[]
        for w in ws:
            #print w[:4]
            nRec1=len(set(bj[:4]).intersection(w[4:8]))
            nRec2=len(set(bj[:4]).intersection(w[10:14]))
            
            #print bj[:5],"<>",w[4:8],"<>",w[10:14],"==> ",len(set(bj[:4]).intersection(w[4:8])),"   ",len(set(bj[:4]).intersection(w[10:14]))
            if nRec1>=2 or nRec2>=2: continue
            if bj[0]==0 or w[4]==0 or w[10]==0: continue

            top=buildTop(w,bj)
            idx=3
            #W infos
            if addW:
                top.extend(w[:4])
                idx+=4
            if addWJets:
                top.extend(w[4:8]) #jet 1
                top.extend(w[4:8]) #jet 2
                idx+=8
            if addWJetsInfos:
                top.extend(w[8:10]+w[14:16])
                idx+=4
            if addBJet:
                top.extend(bj[0:4])
                idx+=4
            if addBJetInfos:
                top.extend(bj[4:6]+[0,0])
                idx+=2

            if idxBJetInfo==-1:
                #print "=====================>> ",idx
                idxBJetInfo=idx
                
            #print np.array(top)
            #print "===================="
            
            tops.append(top)

    #print idxBJetInfo, len(tops),len(tops[0])
    #print np.array(tops)
    if sortByTopMass and sortByWMass and addWJets and sortByBTagScore and addBJetInfos:
        #print 1
        tops.sort(key=lambda t:( abs(t[3]-172),  1.-t[idxBJetInfo],abs(t[7]-80.4) ) )
    elif sortByTopMass and sortByBTagScore and addBJetInfos:
        #print 2,tops[0][idxBJetInfo]
        #tops.sort(key=lambda t:(1.-t[idxBJetInfo], abs(t[3]-172)))
        tops.sort(key=lambda t:(abs(t[3]-172), 1.-t[idxBJetInfo]))
    elif sortByTopMass and sortByWMass and addWJets:
        #print 3
        tops.sort(key=lambda t:(abs(t[3]-172), abs(t[7]-80.4)))
    elif sortByWMass and addWJets and sortByBTagScore and addBJetInfos:
        #print 4
        tops.sort(key=lambda t:(1.-t[idxBJetInfo], abs(t[7]-80.4) ) )
    elif sortByTopMass:
        #print 5
        tops.sort(key=lambda t:abs(t[3]-172),reverse=True)
    elif sortByBTagScore and addBJetInfos:
        #print 6
        tops.sort(key=lambda t:(1.-t[idxBJetInfo]))
    elif sortByWMass and addWJets:
        #print 7
        tops.sort(key=lambda t:(abs(t[7]-80.4)))

    print "tops ============================== "
    print np.array(tops)[:,:5]
    return tops
"""

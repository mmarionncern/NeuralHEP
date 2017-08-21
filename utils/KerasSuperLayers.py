from keras.layers import Conv1D, Dense, Input, Flatten, AveragePooling1D, Activation, Dropout, Merge, LSTM, Add
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Cropping1D, Cropping2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Permute
from keras.layers.core import Lambda, Reshape
from keras.utils import conv_utils
from math import factorial
import tensorflow as tf
import itertools
import numpy as np
import pandas
import random
import sys
import ROOT
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def cropData(input,ymin,ymax=-1,xmin=-1,xmax=-1):

    nY=input._keras_shape[1]
    nX=input._keras_shape[2]
    if ymax==-1:
        ymax=nY-1
    
    yCropped=Cropping1D((ymin,nY-ymax-1))(input)

    if (xmin==-1 and xmax!=-1) or (xmin!=-1 and xmax==-1):
        print "no cropping along the x axis, check the details"
        return yCropped
    if xmin==-1 and xmax==-1:
        return yCropped

    xCropped=Permute((2,1))(Cropping1D((xmin,nX-xmax-1))(Permute((2,1))(yCropped)))
    return xCropped






def superNormDropLayer(input, name, dropOutRate):

    normLayer=BatchNormalization(name="norm_"+name)(input)

    if dropOutRate<0:
        return normLayer

    dropLayer=Dropout(0.1, name="drop_"+name)(normLayer)

    return dropLayer



def superConvLayer(input, name, nNodes=64, kernelSize=1, strides=1,
                   activation="relu", initializers="glorot_uniform", dropOutRate=0.1):

    #print (input._keras_shape[1],input._keras_shape[2],1)
    convLayer=Conv1D(nNodes,
                     kernel_size=kernelSize,
                     strides=strides,
                     kernel_initializer=initializers,
                     #input_shape=(input._keras_shape[1],input._keras_shape[2],1),
                     activation=activation, name="conv_"+name)(input)

    return superNormDropLayer(convLayer, name, dropOutRate)



def superLSTMLayer(input, name, nNodes=64, activation="tanh",
                   initializers="glorot_uniform", dropOutRate=0.1):

    lstmLayer=LSTM(nNodes,
                   name="lstm_"+name,
                   activation=activation,
                   kernel_initializer=initializers)(input)

    return superNormDropLayer(lstmLayer, name, dropOutRate)


def superDenseLayer(input, name, nNodes=64, activation=None,
                    initializers="glorot_uniform", dropOutRate=0.1):

    denseLayer=Dense(nNodes,
                     name="dense_"+name,
                     activation=activation,
                     kernel_initializer=initializers)(input)

    return superNormDropLayer(denseLayer, name, dropOutRate)



def innerResidualLayer1D(input, nNodes, nInner, bnFactor, kernel_size, strides, activation, initializers, name):
    nInput=input._keras_shape[2]
    #print " -> ",nInput, nNodes, nInner
    def unit(x):
        nbp = int(nNodes / bnFactor)
        
        if nInput==nNodes:
            ident = x

            for i in range(nInner):
                x=BatchNormalization(axis=-1)(x)
                x=Activation(activation)(x)
                x=Conv1D(nbp,
                         kernel_size=kernel_size,
                         strides=strides,
                         kernel_initializer=initializers,
                         padding="valid" if (i!=0 and i!=nInner-1) else "same",
                         name="convResSt_"+name+"_"+str(i) )(x)
            
            out = Add()([ident,x])
        else:
            #x=BatchNormalization(axis=-1)(x)
            #x=Activation(activation)(x)
            #ident = x

            for i in range(nInner):
                x=BatchNormalization(axis=-1)(x)
                x=Activation(activation)(x)
                if i==0: ident=x
                x=Conv1D(nbp,
                         kernel_size=kernel_size,
                         strides=strides,
                         kernel_initializer=initializers,
                         padding="valid" if (i!=0 and i!=nInner-1) else "same",
                         name="convRes_"+name+"_"+str(i) )(x)
            
            ident=Conv1D(nbp,
                         kernel_size=kernel_size,
                         strides=strides,
                         kernel_initializer=initializers,
                         padding="valid",
                         name="convRes_"+name+"_St"+str(i) )(ident)

            out = Add()([ident,x])

        return out
    return unit




def ResidualLayer1D(input, name, nNodes=64, nInner=3, nLayers=1,
                  bnFactor=4,
                  kernel_size=1, strides=1, activation="relu",
                    initializers="glorot_uniform", dropOutRate=0.1) :

    def unit(x):
        nInput=input._keras_shape[2]
        for i in range(nLayers):
            if i==0:
                x = innerResidualLayer1D(input,nNodes,nInner,bnFactor, kernel_size, strides, activation, initializers, name+"L"+str(i))(x)
            else:
                x = innerResidualLayer1D(x,nNodes,nInner,bnFactor, kernel_size, strides, activation, initializers, name+"L"+str(i))(x)
        return x
    return unit





###============================================================
###============================================================
# four vector operations

class BuildCombinationsDim2(Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(BuildCombinationsDim2, self).__init__(**kwargs)
    def build(self, input_shape):
        #self.input_shape=input_shape
        self.ncr = self.k*factorial(input_shape[2])/factorial(self.k)/factorial(input_shape[2]-self.k)
        self.crmatrix = K.ones(shape=(self.ncr,input_shape[2]))
        nm = np.zeros((self.ncr,input_shape[2]))
        for i,val in enumerate([x for x in itertools.chain(*itertools.combinations(xrange(input_shape[2]),self.k))]): #jet loop
            nm[i][val]=1.0
        self.crmatrix = K.constant(nm)
        self.crmatrix = K.transpose(self.crmatrix)
        super(BuildCombinationsDim2, self).build(input_shape)
    def call(self,x):
        #print "--<>--", K.dot(x,self.crmatrix).get_shape(), x.get_shape(), self.crmatrix.get_shape()
        return K.dot(x,self.crmatrix)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.ncr)

    
class BuildCombinationsDim1():
    def __init__(self,input,k):
        #print "-->>", input, k
        self.input = input
        self.k = k
        self.perm1 = Permute((2,1))(self.input)
        self.comb = BuildCombinationsDim2(k)(self.perm1)
        #print "comb ", self.comb.get_shape(), self.comb._keras_shape
        self.perm2 = Permute((2,1))(self.comb)
    def get(self):
        return self.perm2


    
class BuildCombinationsDim2AndSum(Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(BuildCombinationsDim2AndSum, self).__init__(**kwargs)
    def build(self, input_shape):
        self.ncr = self.k*factorial(input_shape[2])/factorial(self.k)/factorial(input_shape[2]-self.k)
        self.crmatrix = K.ones(shape=(self.ncr,input_shape[2]))
        nm = np.zeros((self.ncr,input_shape[2]))
        for i,val in enumerate([x for x in itertools.chain(*itertools.combinations(xrange(input_shape[2]),self.k))]): #jet loop
            nm[i][val]=1.0
        self.crmatrix = K.constant(nm)
        self.crmatrix = K.transpose(self.crmatrix)
        super(BuildCombinationsDim2AndSum, self).build(input_shape)
    def call(self,x):
        p = K.dot(x,self.crmatrix)
        p = K.sum(p, axis=2, keepdims=True) #along columns
        #print p.get_shape()
        return p
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], 1)

    
class BuildCombinations4V():
    def __init__(self,input,k):
        self.input = input
        self.k = k
        self.perm1 = Permute((2,1))(self.input)
        self.comb = BuildCombinationsDim2AndSum(k)(self.perm1)
        self.perm2 = Permute((2,1))(self.comb)
        #print self.perm2.get_shape()
    def get(self):
        return self.perm2


    
class GetPtEtaPhiMFrom4V(Layer): #operates over (None, 1, 4)
    def __init__(self, **kwargs):
        super(GetPtEtaPhiMFrom4V, self).__init__(**kwargs)
    def build(self, input_shape):
        self.pxMask = K.constant([[1],[0],[0],[0]])
        self.pyMask = K.constant([[0],[1],[0],[0]])
        self.pzMask = K.constant([[0],[0],[1],[0]])
        self.EMask  = K.constant([[0],[0],[0],[1]])
        super(GetPtEtaPhiMFrom4V, self).build(input_shape)
    def call(self,x):
        px = K.sum(K.dot(x, self.pxMask), axis=2, keepdims=True)
        py = K.sum(K.dot(x, self.pyMask), axis=2, keepdims=True)
        pz = K.sum(K.dot(x, self.pzMask), axis=2, keepdims=True)
        E  = K.sum(K.dot(x, self.EMask), axis=2, keepdims=True)

        px2=K.square(px)
        py2=K.square(py)
        pz2=K.square(pz)
        E2=K.square(E)
        
        pT=K.sqrt(K.sum(K.concatenate([px2,py2]),axis=2,keepdims=True))
        p2=K.sum(K.concatenate([px2,py2,pz2]),axis=2,keepdims=True)
        p=K.sqrt(p2)
        mp2=K.map_fn(lambda x: -x, p2)
        m=K.sqrt(K.sum(K.concatenate([E2,mp2]),axis=2,keepdims=True))

        r=tf.div( pz, K.maximum(pT,0.00000001) )
        theta=K.map_fn(lambda x: np.pi/2. - tf.atan(x) , r )
        thetaB=tf.where( K.less(pz,0.), K.minimum(theta, 179.999999) , K.maximum(theta, 0.000001) )
        eta= K.map_fn(lambda x: -tf.log(tf.tan(x/2.0)), thetaB)
        
        rp=tf.div(px, p)
        phi=tf.acos(rp) #0->pi
        phi=tf.where(K.less(0.,py), phi, K.map_fn(lambda x: -x, phi) )
        #print m.get_shape(), pT.get_shape(), eta.get_shape(), phi.get_shape()
        #print m.get_shape(), K.concatenate([pT,eta, phi, m]).get_shape()
        
        return K.concatenate([pT, eta, phi, m])
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,4)



class GetPxPyPzEFrom4V(Layer): #operates over (None, 1, 4)
    def __init__(self, **kwargs):
        super(GetPxPyPzEFrom4V, self).__init__(**kwargs)
    def build(self, input_shape):
        self.ptMask = K.constant([[1],[0],[0],[0]])
        self.etaMask = K.constant([[0],[1],[0],[0]])
        self.phiMask = K.constant([[0],[0],[1],[0]])
        self.mMask  = K.constant([[0],[0],[0],[1]])
        super(GetPxPyPzEFrom4V, self).build(input_shape)
    def call(self,x):
        pt = K.sum(K.dot(x, self.ptMask), axis=2, keepdims=True)
        eta = K.sum(K.dot(x, self.etaMask), axis=2, keepdims=True)
        phi = K.sum(K.dot(x, self.phiMask), axis=2, keepdims=True)
        m  = K.sum(K.dot(x, self.mMask), axis=2, keepdims=True)
        
        theta=K.map_fn( lambda h: 2*tf.atan(K.exp(h)), eta)
        p=tf.div(pt, K.sin(theta) )

        px=tf.multiply(pt, K.cos(phi) )
        py=tf.multiply(pt, K.sin(phi) )
        pz=tf.multiply(p, K.cos(theta) )

        p2=K.square(p)
        m2=K.square(m)
        
        E=K.sqrt(K.sum(K.concatenate([p2,m2]),axis=2,keepdims=True))
        
        return K.concatenate([px, py, pz, E])

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,4)



class Sum4V(Layer):
    def __init__(self, **kwargs):
        super(Sum4V, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Sum4V, self).build(input_shape)
    def call(self,x):
        print "sususus ", x.get_shape()
        print x[:,0,:,:]
        return K.sum(x, axis=2, keepdims=True) #along columns
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], 4)


class SumCombinatorial4VBlocks():
    def __init__(self,input,k):
        self.input = input
        nComb=self.input._keras_shape[1]/k
        self.reshape1 = Reshape((nComb,k,4))(input)
        self.sum = Sum4V()(self.reshape1)
        self.reshape2 = Reshape((nComb,4))(self.sum)
        #print self.perm2.get_shape()
    def get(self):
        return self.reshape2
    

class Convert4VBlocks():
    def __init__(self,input, cartToPEPM=True):
        self.input = input
        nObj=input._keras_shape[1]
        self.reshape1 = Reshape((nObj,1,4))(input)
        print "yoyolololo",self.reshape1._keras_shape
        if cartToPEPM:
            self.conv = TimeDistributed(GetPtEtaPhiMFrom4V())(self.reshape1)
        else:
            self.conv = TimeDistributed(GetPxPyPzEFrom4V())(self.reshape1)
        self.reshape2 = Reshape((nObj,4))(self.conv)
        #print self.perm2.get_shape()
    def get(self):
        return self.reshape2
    

class SortTensor(Layer): #input shape (None, nObjs, nVars)
    def __init__(self, nc,  **kwargs): #nc: column index for ordering
        self.nc=nc
        super(SortTensor, self).__init__(**kwargs)
    def build(self, input_shape):
        super(SortTensor, self).build(input_shape)

    def call(self,x):
        shape=x._keras_shape
        
        idxs=tf.nn.top_k(x[:,:,self.nc], k=shape[-2]).indices
        idxs=tf.reshape(idxs, (-1, shape[1]) )
       
        b_idxs=tf.scan(lambda a,x: a+1, idxs, np.array([-1]*shape[1]) )
        b_idxs=tf.to_int32(b_idxs)
        
        idxs=tf.stack([b_idxs,idxs],1)
        idxs=tf.transpose(idxs,perm=[0,2,1])
                
        return tf.gather_nd(x, idxs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],3,4)


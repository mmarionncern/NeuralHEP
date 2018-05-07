from keras.layers import Conv1D, Dense, Input, Flatten, AveragePooling1D, Activation, Dropout, Merge, LSTM, Add
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Cropping1D, Cropping2D,UpSampling1D
from keras.layers.merge import Concatenate, Maximum
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.core import RepeatVector
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Permute
from keras.layers.core import Lambda, Reshape
from keras.utils import conv_utils
from keras.backend import int_shape
from math import factorial
import tensorflow as tf
import itertools
import numpy as np
#import pandas
import random
import sys
#import ROOT
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def cropData(input,ymin,ymax=-1,xmin=-1,xmax=-1):

    nY=input._keras_shape[1]
    nX=input._keras_shape[2]
    if ymax==-1:
        ymax=nY-1
    
    yCropped=Cropping1D((ymin,nY-ymax-1))(input)

    if (xmin==-1 and xmax!=-1) or (xmin!=-1 and xmax==-1):
        print("no cropping along the x axis, check the details")
        return yCropped
    if xmin==-1 and xmax==-1:
        return yCropped

    xCropped=Permute((2,1))(Cropping1D((xmin,nX-xmax-1))(Permute((2,1))(yCropped)))
    return xCropped


def cropData2D(input,ymin,ymax=-1,xmin=-1,xmax=-1):

    nY=input._keras_shape[1]
    nX=input._keras_shape[2]
    if ymax==-1:
        ymax=nY-1
    if xmin==-1:
        xmin=0
    if xmax==-1:
        xmax=nX-1
    cropped=Cropping2D(cropping=((ymin,nY-ymax-1),(xmin,nX-xmax-1)))(input)
    return cropped





def superNormDropLayer(input, name, dropOutRate):

    normLayer=BatchNormalization(name="norm_"+name)(input)

    if dropOutRate<0:
        return normLayer

    dropLayer=Dropout(0.1, name="drop_"+name)(normLayer)

    return dropLayer



def superConvLayer(input, name, nNodes=64, kernelSize=1, strides=1,
                   activation="relu", initializers="glorot_uniform", dropOutRate=0.1):

    convLayer=Conv1D(int(nNodes),
                     kernel_size=kernelSize,
                     strides=strides,
                     kernel_initializer=initializers,
                     activation=activation, name="conv_"+name)(input)

    return superNormDropLayer(convLayer, name, dropOutRate)



def superLSTMLayer(input, name, nNodes=64, activation="tanh",
                   initializers="glorot_uniform",return_sequence=False, dropOutRate=0.1):

    lstmLayer=LSTM(int(nNodes),
                   name="lstm_"+name,
                   activation=activation,
                   return_sequences=return_sequence,
                   kernel_initializer=initializers)(input)

    return superNormDropLayer(lstmLayer, name, dropOutRate)


def superDenseLayer(input, name, nNodes=64, activation=None,
                    initializers="glorot_uniform", dropOutRate=0.1):#, input_dim=None):

    denseLayer=Dense(int(nNodes),
                     name="dense_"+name,
                     activation=activation,
                     #input_dim=input_dim,
                     kernel_initializer=initializers)(input)

    return superNormDropLayer(denseLayer, name, dropOutRate)



def innerResidualLayer1D(input, nNodes, nInner, bnFactor, kernel_size, strides, activation, initializers, name):
    nInput=input._keras_shape[2]
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
        #self.ncr=None
        #self.crmatrix=None
        super(BuildCombinationsDim2, self).__init__(**kwargs)
    def build(self, input_shape):
        self.ncr = self.k*factorial(input_shape[2])/factorial(self.k)/factorial(input_shape[2]-self.k)
        self.crmatrix = K.ones(shape=(self.ncr,input_shape[2]))
        nm = np.zeros((self.ncr,input_shape[2]))
        for i,val in enumerate([x for x in itertools.chain(*itertools.combinations(range(input_shape[2]),self.k))]): #jet loop
            nm[i][val]=1.0
        self.crmatrix = K.constant(nm)
        self.crmatrix = K.transpose(self.crmatrix)
        super(BuildCombinationsDim2, self).build(input_shape)
    def call(self,x):
        return K.dot(x,self.crmatrix)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.ncr)
    def get_config(self):
        config = {
            'k':self.k,
        #    'ncr':self.ncr,
        #    'crmatrix':self.crmatrix
        }
        base_config = super(BuildCombinationsDim2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class BuildCombinationsDim1():
    def __init__(self,input,k):
        self.input = input
        self.k = k
        self.perm1 = Permute((2,1))(self.input)
        self.comb = BuildCombinationsDim2(k)(self.perm1)
        self.perm2 = Permute((2,1))(self.comb)
    def get(self):
        return self.perm2
    def get_config(self):
        config = {
            'k':self.k,
        }
        base_config = super(BuildCombinationsDim1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class BuildSequentialCombinationsDim2(Layer):
    def __init__(self, k, step, **kwargs):
        self.k = k
        self.step= step
        super(BuildSequentialCombinationsDim2, self).__init__(**kwargs)
    def build(self, input_shape):
        self.nCombs=input_shape[2]/self.step
        self.ncr=self.k*factorial(self.step)/factorial(self.k)/factorial(self.step-self.k)
        self.crmatrix = K.ones(shape=(self.ncr,input_shape[2]))
        nm = np.zeros((self.ncr*(input_shape[2]/self.step),input_shape[2]))
        for s in range(int(self.nCombs)):
            for i,val in enumerate([x for x in itertools.chain(*itertools.combinations(range(self.step),self.k))]): #jet loop
                nm[i+(s*self.ncr)][val+s*self.step]=1.0
        self.crmatrix = K.constant(nm)
        self.crmatrix = K.transpose(self.crmatrix)
        super(BuildSequentialCombinationsDim2, self).build(input_shape)
    def call(self,x):
        return K.dot(x,self.crmatrix)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.nCombs*self.ncr)
    def get_config(self):
        config = {
            'k':self.k,
            'step':self.step,
    #        'nCombs':self.nCombs,
    #        'ncr':self.ncr,
    #        'crmatrix':self.crmatrix,
        }
        base_config = super(BuildSequentialCombinationsDim2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class BuildSequentialCombinationsDim1():
    def __init__(self,input,k, step):
        self.input = input
        self.k = k
        self.perm1 = Permute((2,1))(self.input)
        self.comb = BuildSequentialCombinationsDim2(k, step)(self.perm1)
        self.perm2 = Permute((2,1))(self.comb)
    def get(self):
        return self.perm2
    def get_config(self):
        config = {
            'k':self.k,
        }
        base_config = super(BuildSequentialCombinationsDim1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class BuildCombinationsDim2AndSum(Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(BuildCombinationsDim2AndSum, self).__init__(**kwargs)
    def build(self, input_shape):
        self.ncr = self.k*factorial(input_shape[2])/factorial(self.k)/factorial(input_shape[2]-self.k)
        self.crmatrix = K.ones(shape=(self.ncr,input_shape[2]))
        nm = np.zeros((self.ncr,input_shape[2]))
        for i,val in enumerate([x for x in itertools.chain(*itertools.combinations(range(input_shape[2]),self.k))]): #jet loop
            nm[i][val]=1.0
        self.crmatrix = K.constant(nm)
        self.crmatrix = K.transpose(self.crmatrix)
        super(BuildCombinationsDim2AndSum, self).build(input_shape)
    def call(self,x):
        p = K.dot(x,self.crmatrix)
        p = K.sum(p, axis=2, keepdims=True) #along columns
        return p
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], 1)
    def get_config(self):
        config = {
            'k':self.k,
    #        'ncr':self.ncr,
    #        'crmatrix':self.crmatrix,
        }
        base_config = super(BuildCombinationsDim2AndSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class BuildCombinations4V():
    def __init__(self,input,k):
        self.input = input
        self.k = k
        self.perm1 = Permute((2,1))(self.input)
        self.comb = BuildCombinationsDim2AndSum(k)(self.perm1)
        self.perm2 = Permute((2,1))(self.comb)
    def get(self):
        return self.perm2
    def get_config(self):
        config = {
           'k':self.k
        }
        base_config = super(BuildCombinations4V, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
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
        
        rp=tf.div(px, pT)
        phi=tf.acos(rp) #0->pi
        phi=tf.where(K.less(0.,py), phi, K.map_fn(lambda x: -x, phi) )
        
        return K.concatenate([pT, eta, phi, m])
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,4)
    #def get_config(self):
    #    config = {
    #        'pxMask':self.pxMask,
    #        'pyMask':self.pyMask,
    #        'pzMask':self.pzMask,
    #        'EMask':self.EMask,
    #    }
    #    base_config = super(GetPtEtaPhiMFrom4V, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

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
        pz=tf.multiply(-1., tf.multiply(p, K.cos(theta) ))

        p2=K.square(p)
        m2=K.square(m)
        
        E=K.sqrt(K.sum(K.concatenate([p2,m2]),axis=2,keepdims=True))
        
        return K.concatenate([px, py, pz, E])
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,4)
    #def get_config(self):
    #    config = {
    #        'ptMask':self.ptMask,
    #        'etaMask':self.etaMask,
    #        'phiMask':self.phiMask,
    #        'mMask':self.mMask,
    #    }
    #    base_config = super(GetPxPyPzEFrom4V, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


class Sum4V(Layer):
    def __init__(self, **kwargs):
        super(Sum4V, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Sum4V, self).build(input_shape)
    def call(self,x):
        return K.sum(x, axis=2, keepdims=True) #along columns
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], 4)


class SumCombinatorial4VBlocks():
    def __init__(self,input,k):
        self.input = input
        self.nComb=self.input._keras_shape[1]/k
        self.reshape1 = Reshape((int(self.nComb),k,4))(input)
        self.sum = Sum4V()(self.reshape1)
        self.reshape2 = Reshape((int(self.nComb),4))(self.sum)
    def get(self):
        return self.reshape2
    #def get_config(self):
    #    config = {'nCombs':self.nCombs,
    #    }
    #    base_config = super(SumCombinatorial4VBlocks, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

class Convert4VBlocks():
    def __init__(self,input, cartToPEPM=True):
        self.cartToPEPM=cartToPEPM
        self.input = input
        nObj=input._keras_shape[1]
        self.reshape1 = Reshape((int(nObj),1,4))(input)
        if self.cartToPEPM:
            self.conv = TimeDistributed(GetPtEtaPhiMFrom4V())(self.reshape1)
        else:
            self.conv = TimeDistributed(GetPxPyPzEFrom4V())(self.reshape1)
        self.reshape2 = Reshape((int(nObj),4))(self.conv)
        #super(Convert4VBlocks,self).__init__(**kwargs)
    def get(self):
        return self.reshape2
    def get_config(self):
        config = {'cartToPEPM':self.cartToPEPM,
        }
        base_config = super(SumCombinatorial4VBlocks, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DeltaR(Layer):
    #operates over (None, nObjs, 4)
    #perform a dR computation between each pair : (1,2) (3,4) etc.
    def __init__(self, **kwargs):
        super(DeltaR, self).__init__(**kwargs)
    def build(self, input_shape):
        self.etaMask = K.constant([[0],[1],[0],[0]])
        self.phiMask = K.constant([[0],[0],[1],[0]])
        self.v1Mask=K.constant([[1],[0]])
        self.v2Mask=K.constant([[0],[1]])
        self.nCombs=input_shape[1]/2
        super(DeltaR, self).build(input_shape)
    def call(self,x):
        vals=tf.reshape(x,(-1,int(self.nCombs),2,4))
        etas = K.dot(vals, self.etaMask)
        #etas = tf.transpose(etas, perm=[0,2,1])
        etas = tf.transpose(etas, perm=[0,1,3,2])
        phis = K.dot(vals, self.phiMask)
        #phis = tf.transpose(phis, perm=[0,2,1])
        phis = tf.transpose(phis, perm=[0,1,3,2])

        eta1 = K.dot(etas, self.v1Mask)
        phi1 = K.dot(phis, self.v1Mask)
        eta2 = K.dot(etas, self.v2Mask)
        phi2 = K.dot(phis, self.v2Mask)
        
        phi1=tf.where(K.less(phi1,0.), K.map_fn(lambda x: np.pi*2. + x, phi1 ) , phi1 ) #0->2pi
        phi2=tf.where(K.less(phi2,0.), K.map_fn(lambda x: np.pi*2. + x, phi2 ) , phi2 ) #0->2pi
        dphi=tf.subtract(phi1,phi2)
        dphi=tf.where(K.less(dphi,-np.pi), K.map_fn(lambda x: np.pi*2. + x, dphi ) , dphi )
        dphi=tf.where(K.less(np.pi,dphi), K.map_fn(lambda x: x-np.pi*2., dphi ) , dphi )
        
        deta=tf.subtract(eta1,eta2)

        dphi2=K.square(dphi)
        deta2=K.square(deta)

        dR=K.sqrt(K.sum(K.concatenate([dphi2,deta2]),axis=3,keepdims=True))
        dR=tf.reshape(dR,(-1,int(self.nCombs),1))
        
        return dR
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.nCombs,1)
    #def get_config(self):
    #    config = {'etaMask':self.etaMask,
    #              'phiMask':self.phiMask,
    #              'v1Mask':self.v1Mask,
    #              'v2Mask':self.v2Mask,
    #              'nCombs':self.nCombs
    #    }
    #    base_config = super(DeltaR, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

### basic operation functions ===============    
#replaced by GlobalXXXPooling1D

#class MaxElement(Layer): #input shape (None, nObjs, 1)
#    def __init__(self, **kwargs):
#        super(MaxElement, self).__init__(**kwargs)
#    def build(self, input_shape):
#        super(MaxElement, self).build(input_shape)
#
#    def call(self,x):
#        return tf.reduce_max(x, axis=1, keep_dims=True)
#    
#    def compute_output_shape(self, input_shape):
#        return (input_shape[0],1,input_shape[2])

#class MinElement(Layer): #input shape (None, nObjs, 1)
#    def __init__(self, **kwargs):
#        super(MinElement, self).__init__(**kwargs)
#    def build(self, input_shape):
#        super(MinElement, self).build(input_shape)
#
#    def call(self,x):
#        return tf.reduce_min(x, axis=1, keep_dims=True)
#    
#    def compute_output_shape(self, input_shape):
#        return (input_shape[0],1,input_shape[2])
    
#class AverageElement(Layer): #input shape (None, nObjs, 1)
#    def __init__(self, **kwargs):
#        super(AverageElement, self).__init__(**kwargs)
#    def build(self, input_shape):
#        super(AverageElement, self).build(input_shape)
#
#    def call(self,x):
#        return tf.reduce_mean(x, axis=1, keep_dims=True)
#    
#    def compute_output_shape(self, input_shape):
#        return (input_shape[0],1,input_shape[2])


class SequentialReduceOperation(Layer):# input shape (None, nObjs, nVars)
    def __init__(self, operation, k, **kwargs):
        self.operation=operation
        self.op=0 if operation=="+" else ( 1 if operation=="avg" else ( 2 if operation=="max" else ( 3 if operation=="min" else ( 4 if operation=="-" else ( 5 if operation=="-abs" else -1 ) ) ) ) )
        if self.op==-1:
            print("Error, sequential reducive operation not defined!")
            sys.exit(0)
        self.k=k
        super(SequentialReduceOperation, self).__init__(**kwargs)
    def build(self, input_shape):
        self.deep=input_shape[2]
        self.step=input_shape[1]/self.k
        if input_shape[1]%self.k!=0:
            print("Error, sequential reducive operation with mismatching number of pairs")
            sys.exit(0)
        tmpX=np.zeros((self.k,1))
        tmpY=np.full((self.k,1),1)
        tmpX[0][:]=1.
        tmpY[0][:]=0.
        self.xMask=K.constant(tmpX)
        self.yMask=K.constant(tmpY)
        super(SequentialReduceOperation, self).build(input_shape)
    def call(self,x):
        vals=tf.reshape(x,(-1,int(self.step),int(self.k), int(self.deep) ))
        if self.op==0: #addition
            vals=tf.reduce_sum(vals, axis=2)
        elif self.op==1: #average
            vals=tf.reduce_mean(vals, axis=2)#
        elif self.op==2: #maximum
            vals=tf.reduce_max(vals, axis=2)
        elif self.op==3: #minimum
            vals=tf.reduce_min(vals, axis=2)
        elif self.op==4 or self.op==5: #subtraction (first-others)
            vals = tf.transpose(vals, perm=[0,1,3,2])
            xVals= K.dot(vals,self.xMask)
            yVals= K.dot(vals,self.yMask)
            xVals = tf.transpose(xVals, perm=[0,1,3,2])
            yVals = tf.transpose(yVals, perm=[0,1,3,2])
            vals= tf.subtract(xVals,yVals)

            if self.op==5:
                vals=tf.abs(vals)
            
        vals=tf.reshape(vals,(-1,int(self.step),int(self.deep) ))
        return vals
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.step,input_shape[2])
    def get_config(self):
        config = {'k':self.k,
                  'operation':self.operation,
    #              'deep':self.deep,
    #              'step':self.step,
    #              'xMask':self.xMask,
    #              'yMask':self.yMask
        }
        base_config = super(SequentialReduceOperation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class SubtractElement(Layer):  #input shape (None, 2, nVars)
    def __init__(self, **kwargs):
        super(SubtractElement, self).__init__(**kwargs)
    def build(self, input_shape):
        self.xMask=K.constant([[1],[0]])
        self.yMask=K.constant([[0],[1]])
        super(SubtractElement, self).build(input_shape)

    def call(self,x):
        vals = tf.transpose(x, perm=[0,2,1])
        xVals= K.dot(vals,self.xMask)
        yVals= K.dot(vals,self.yMask)
        xVals = tf.transpose(xVals, perm=[0,2,1])
        yVals = tf.transpose(yVals, perm=[0,2,1])
        
        sub= tf.subtract(xVals,yVals)

        return sub
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,input_shape[2])
    #def get_config(self):
    #    config = {'xMask':self.xMask,
    #              'yMask':self.yMask}
    #    base_config = super(SubtractElement, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

class Sort(Layer): #input shape (None, nObjs, nVars)
    def __init__(self, colIdx, reverse=False, sortByClosestValue=False, target=None,  **kwargs): #colIdx: column index used for ordering. if reverse, from smaller to larger value
        self.nc=colIdx
        self.sortByClosestValue=sortByClosestValue
        self.target=target
        self.reverse=reverse
        super(Sort, self).__init__(**kwargs)
    def build(self, input_shape):
        self.mask=None
        if self.sortByClosestValue:
            tmp=np.zeros((input_shape[2],1))
            tmp[self.nc][0]=1
            self.mask=K.constant(tmp)
        super(Sort, self).build(input_shape)

    def call(self,x):    
        shape=x.get_shape().as_list() #int_shape(x) #x._keras_shape
        idxs=None
        if not self.sortByClosestValue:
            idxs=tf.nn.top_k(x[:,:,self.nc], k=shape[-2]).indices
            idxs=tf.reshape(idxs, (-1, shape[1] ) )
        else:
            tmp=K.dot(x,self.mask)
            tmp=K.map_fn(lambda e: -1*abs(e-self.target), tmp)
            idxs=tf.nn.top_k(tmp[:,:,0], k=shape[-2]).indices
            idxs=tf.reshape(idxs, (-1, shape[1] ) )
                    
        b_idxs=tf.scan(lambda a,x: a+1, idxs, np.array([-1]*shape[1]) )
        b_idxs=tf.to_int32(b_idxs)
        
        idxs=tf.stack([b_idxs,idxs],1)
        idxs=tf.transpose(idxs,perm=[0,2,1])
        if self.reverse:
            idxs=tf.reverse(idxs,[1])

        return tf.gather_nd(x, idxs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2])
    def get_config(self):
        config = {'colIdx':self.nc,
                  'sortByClosestValue':self.sortByClosestValue,
                  'target':self.target,
                  'reverse':self.reverse,
    #              'mask':self.mask
        }
        base_config = super(Sort, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SequentialSort(Layer):# input shape (None, nObjs, nVars)
    def __init__(self, k, colIdx, reverse=False, sortByClosestValue=False, target=None, **kwargs):
        self.k=k
        self.nc=colIdx
        self.sortByClosestValue=sortByClosestValue
        self.target=target
        self.reverse=reverse
        super(SequentialSort, self).__init__(**kwargs)
    def build(self, input_shape):
        self.deep=input_shape[2]
        self.step=input_shape[1]/self.k
        if input_shape[1]%self.k!=0:
            print("Error, sequential sort with mismatching number of pairs")
            sys.exit(0)
        tmpOffsets=np.zeros((input_shape[1]))
        for i in range(int(input_shape[1])):
            tmpOffsets[i]=(int(i/self.k))*self.k
        self.offsets=tf.to_int32(K.constant(tmpOffsets))      
        self.mask=None
        if self.sortByClosestValue:
                  tmp=np.zeros((input_shape[2],1))
                  tmp[self.nc][0]=1
                  self.mask=K.constant(tmp)
        super(SequentialSort, self).build(input_shape)
    def call(self,x):
        shape=int_shape(x) #x._keras_shape
        vals=tf.reshape(x,(-1,int(self.step),int(self.k), int(self.deep) ))
        idxs=None
        if not self.sortByClosestValue:
            idxs=tf.nn.top_k(vals[:,:,:,self.nc], k=self.k).indices
            if self.reverse:
                idxs=tf.reverse(idxs,[2])
            idxs=tf.reshape(idxs, (-1, int(shape[1]) ) )
        else:
            tmp=K.dot(vals,self.mask)
            tmp=K.map_fn(lambda e: -1*abs(e-self.target), tmp)
            idxs=tf.nn.top_k(tmp[:,:,:,0], k=self.k).indices
            if self.reverse:
                idxs=tf.reverse(idxs,[2])
            idxs=tf.reshape(idxs, (-1, int(shape[1]) ) )

        idxs=tf.add(idxs,self.offsets)
        b_idxs=tf.scan(lambda a,x: a+1, idxs, np.array([-1]*int(shape[1]) ) )
        b_idxs=tf.to_int32(b_idxs)
        idxs=tf.stack([b_idxs,idxs],1)
        idxs=tf.transpose(idxs,perm=[0,2,1])

        return tf.gather_nd(x, idxs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2])
    def get_config(self):
        config = {'k':self.k,
                  'colIdx':self.nc,
                  'sortByClosestValue':self.sortByClosestValue,
                  'target':self.target,
                  'reverse':self.reverse,
    #              'deep':self.deep,
    #              'step':self.step,
    #              'offsets':self.offsets,
    #              'mask':self.mask
        }
        base_config = super(SequentialSort, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



superLayers={"BuildCombinationsDim2":BuildCombinationsDim2, 
             "BuildCombinationsDim1":BuildCombinationsDim1,
             "BuildSequentialCombinationsDim2":BuildSequentialCombinationsDim2,
             "BuildSequentialCombinationsDim1":BuildSequentialCombinationsDim1,
             "BuildCombinationsDim2AndSum":BuildCombinationsDim2AndSum,
             "BuildCombinations4V":BuildCombinations4V,
             "GetPtEtaPhiMFrom4V":GetPtEtaPhiMFrom4V,
             "GetPxPyPzEFrom4V":GetPxPyPzEFrom4V,
             "Sum4V":Sum4V,
             "SumCombinatorial4VBlocks":SumCombinatorial4VBlocks,
             "Convert4VBlocks":Convert4VBlocks,
             "DeltaR":DeltaR,
             "SequentialReduceOperation":SequentialReduceOperation,
             "SubtractElement":SubtractElement,
             "Sort":Sort,
             "SequentialSort":SequentialSort
}

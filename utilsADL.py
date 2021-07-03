import numpy as np
import pandas as pd
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import scipy
from scipy import io
import sklearn
from sklearn import preprocessing
import pdb
import matplotlib.pyplot as plt

class meanStdCalculator(object):
	# developed and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    # license BSD 3-Clause "New" or "Revised" License
    def __init__(self):
        self.mean     = 0.0
        self.mean_old = 0.0
        self.std      = 0.001
        self.count    = 0.0
        self.minMean  = 100.0
        self.minStd   = 100.0
        self.M_old    = 0.0
        self.M        = 0.0
        self.S        = 0.0
        self.S_old    = 0.0
        
    def updateMeanStd(self, data, cnt = 1):
        self.data     = data
        self.mean_old = self.mean  # copy.deepcopy(self.mean)
        self.M_old    = self.count*self.mean_old
        self.M        = self.M_old + data
        self.S_old    = self.S     # copy.deepcopy(self.S)
        if self.count > 0:
            self.S    = self.S_old + ((self.count*data - self.M_old)**2)/(self.count*(self.count + cnt) + 0.0001)
        
        self.count   += cnt
        self.mean     = self.mean_old + (data-self.mean_old)/((self.count + 0.0001))  # np.divide((data-self.mean_old),self.count + 0.0001)
        self.std      = np.sqrt(self.S/(self.count + 0.0001))
        
        # if (self.std != self.std).any():
        #     print('There is NaN in meanStd')
        #     pdb.set_trace()
    
    def resetMinMeanStd(self):
        self.minMean = self.mean  # copy.deepcopy(self.mean)
        self.minStd  = self.std   # copy.deepcopy(self.std)
        
    def updateMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = self.mean  # copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd  = self.std   # copy.deepcopy(self.std)


class dataLoader(object):
    def __init__(self,fileName,batchSize = 500):
        self.fileName  = fileName
        self.batchSize = batchSize
        self.loadDataFromMatFile()
        
    def loadDataFromMatFile(self):
        data1          = scipy.io.loadmat(self.fileName)  # change your folder
        data           = data1.get('data')
        data           = torch.from_numpy(data)
        data           = data.float()
        self.data      = data[:,0:-1]
        label          = data[:,-1]
        self.label     = label.long()
        self.nData     = data.shape[0]
        self.nBatch    = int(self.nData/self.batchSize)
        self.nInput    = self.data.shape[1]
        self.nOutput   = torch.unique(self.label).shape[0]
        print('Number of input: ', self.nInput)
        print('Number of output: ', self.nOutput)
        print('Number of batch: ', self.nBatch)
    
    def maxMinNormalization(self):
        self.data = torch.from_numpy(preprocessing.minmax_scale(self.data, feature_range=(0.001, 1))).float()
        
    def zScoreNormalization(self):
        self.data = torch.from_numpy(scipy.stats.zscore(self.data, axis=0)).float()

def labeledIdx(nData, nLabeled):
    # torch.manual_seed(0)
    np.random.seed(0)
    idx = torch.tensor(np.random.permutation(nData)[0:int(nLabeled*nData)]).long()
    # idx = torch.randperm(nData)[0:int(nLabeled*nData)]
    # pdb.set_trace()
    
    return idx

def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out

def deleteRowTensor(x,index):
    x = x[torch.arange(x.size(0))!=index] 
    
    return x

def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

def oneHot(label,nClass):
    nData = label.shape[0]
    
    oneHot = torch.zeros(nData,nClass)
    
    for i, lbl in enumerate(label):
        oneHot[i][lbl] = 1
    
    return oneHot

def plotPerformance(Iter,accuracy,loss,hiddenNode,hiddenLayer,winningLayer):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=8)                   # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(5,1,figsize=(8, 12))
#     fig.tight_layout()

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    
    ax1.plot(Iter,accuracy,'k-')
#     ax1.set_title('Testing accuracy')
    ax1.set_ylabel('Áccuracy (%)')
#     ax1.set_xlabel('Number of bathces')
    ax1.yaxis.tick_right()
    ax1.autoscale_view('tight')
    ax1.set_ylim(ymin=0,ymax=100)
    ax1.set_xlim(xmin=0,xmax=len(Iter))

    ax2.plot(Iter,loss,'k-')
#     ax2.set_title('Testing loss')
    ax2.set_ylabel('Testing loss')
#     ax2.set_xlabel('Number of bathces')
    ax2.yaxis.tick_right()
    ax2.autoscale_view('tight')
    ax2.set_ylim(ymin=0)
    ax2.set_xlim(xmin=0,xmax=len(Iter))

    ax3.plot(Iter,hiddenNode,'k-')
#     ax3.set_title('Hidden node evolution')
    ax3.set_ylabel('Hidden node')
#     ax3.set_xlabel('Number of bathces')
    ax3.yaxis.tick_right()
    ax3.autoscale_view('tight')
    ax3.set_ylim(ymin=0)
    ax3.set_xlim(xmin=0,xmax=len(Iter))

    ax4.plot(Iter,hiddenLayer,'k-')
#     ax4.set_title('Hidden layer evolution')
    ax4.set_ylabel('Hidden layer')
#     ax4.set_xlabel('Number of bathces')
    ax4.yaxis.tick_right()
    ax4.autoscale_view('tight')
    ax4.set_ylim(ymin=0)
    ax4.set_xlim(xmin=0,xmax=len(Iter))

    ax5.plot(Iter,winningLayer,'k-')
#     ax5.set_title('Hidden layer evolution')
    ax5.set_ylabel('The winning layer')
    ax5.set_xlabel('Number of bathces')
    ax5.yaxis.tick_right()
    ax5.autoscale_view('tight')
    ax5.set_ylim(ymin=0)
    ax5.set_xlim(xmin=0,xmax=len(Iter))

# def generateWeightXavInit(nInput,nNode,nOut,nNewNode):
#     copyNet         = basicNet(nInput,nNode,nOut)
#     newWeight       = copyNet.linear.weight.data[0:nNewNode]
#     newWeightNext   = copyNet.linear.weight.data[:,0:nNewNode]
#     newOutputWeight = copyNet.linearOutput.weight.data[:,0:nNewNode]
    
#     return newWeight, newOutputWeight, newWeightNext
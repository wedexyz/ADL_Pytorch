import numpy as np
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import pdb
from collections import deque
import random
import warnings
from utilsADL import meanStdCalculator, probitFunc, deleteRowTensor, deleteColTensor, oneHot
from sgdModif import SGD

warnings.filterwarnings("ignore", category=RuntimeWarning)

class basicNet(nn.Module):
    def __init__(self, no_input, no_hidden, classes):
        super(basicNet, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        # self.activation = nn.Sigmoid()
        self.activation = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
    def forward(self, x):
        x  = self.linear(x)
        h  = self.activation(x)
        x  = self.linearOutput(h)
        h2 = (h.detach().clone())**2  # There is no gradient in this graph
        x2 = self.linearOutput(h2)    # There is no gradient in this graph
        
        return x, h, x2

class smallAdl():
    def __init__(self, no_input, no_hidden, classes):
        self.network = basicNet(no_input, no_hidden, classes)
        self.netUpdateProperties()
        
    def getNetProperties(self):
        print(self.network)
        print('No. of inputs :',self.nNetInput)
        print('No. of nodes :',self.nNodes)
        print('No. of parameters :',self.nParameters)
    
    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)
        print('Output weight: \n', self.network.linearOutput.weight)
        print('Output bias: \n', self.network.linearOutput.bias)
        
    # def getTrainableParameters(self):
    #     return list(self.network.parameters())
    
    def netUpdateProperties(self):
        self.nNetInput   = self.network.linear.in_features
        self.nNodes      = self.network.linear.out_features
        self.nOutputs    = self.network.linearOutput.out_features
        self.nParameters = (self.network.linear.in_features*self.network.linear.out_features +
                            len(self.network.linear.bias.data) + 
                            self.network.linearOutput.in_features*self.network.linearOutput.out_features +
                            len(self.network.linearOutput.bias.data))
        
    def nodeGrowing(self,nNewNode = 1):
        nNewNodeCurr = self.nNodes + nNewNode
        
        # grow node
        # newWeight, newOutputWeight,_     = generateWeightXavInit(self.nNetInput,nNewNodeCurr,self.nOutputs,nNewNode)
        newWeight                        = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput))
        newOutputWeight                  = nn.init.xavier_uniform_(torch.empty(self.nOutputs, nNewNode))
        self.network.linear.weight.data  = torch.cat((self.network.linear.weight.data,
                                                          newWeight),0)  # grow input weights
        self.network.linear.bias.data    = torch.cat((self.network.linear.bias.data,
                                                          torch.zeros(nNewNode)),0)  # grow input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        # add new input to the classifier
        self.network.linearOutput.weight.data = torch.cat((self.network.linearOutput.weight.data,
                                                                newOutputWeight),1)
        self.network.linearOutput.in_features = nNewNodeCurr
        del self.network.linearOutput.weight.grad
        del self.network.linearOutput.bias.grad
        
        self.netUpdateProperties()
    
    def nodePruning(self,pruneIdx,nPrunedNode = 1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node
        
        # prune node for current layer, output
        self.network.linear.weight.data  = deleteRowTensor(self.network.linear.weight.data,
                                                           pruneIdx)  # prune input weights
        self.network.linear.bias.data    = deleteRowTensor(self.network.linear.bias.data,
                                                           pruneIdx)  # prune input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad

        # prune input weight of classifier
        self.network.linearOutput.weight.data = deleteColTensor(self.network.linearOutput.weight.data,pruneIdx)
        self.network.linearOutput.in_features = nNewNodeCurr
        del self.network.linearOutput.weight.grad
        del self.network.linearOutput.bias.grad
        
        self.netUpdateProperties()
        
    def inputGrowing(self,nNewInput = 1):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        # _,_,newWeightNext = generateWeightXavInit(nNewInputCurr,self.nNodes,self.nOutputs,nNewInput)
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nNodes, nNewInput))
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data,newWeightNext),1)
        del self.network.linear.weight.grad

        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self,pruneIdx,nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linear.weight.data = deleteColTensor(self.network.linear.weight.data,pruneIdx)
        del self.network.linear.weight.grad

        # update input features
        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()

class ADL():
    def __init__(self,nInput,nOutput,df = 0.001,plt = 0.05,alpha_w = 0.0005,alpha_d = 0.0001,LR = 0.02,trMode = 1):
        # trMode 1 > only take the winning layer parameter
        # trMode 2 > take the winning layer parameter and all previous layer whose voting weight equals 0
        # # random seed control
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)

        # initial network
        self.net                     = [smallAdl(nInput,nOutput,nOutput)]
        self.votingWeight            = [1.0]
        self.dynamicDecreasingFactor = [1.0]
        self.winLayerIdx             = 0

        # network significance
        self.averageBias  = [meanStdCalculator()]
        self.averageVar   = [meanStdCalculator()]
        self.averageInput = meanStdCalculator()

        # hyper parameters
        self.decreasingFactor    = df
        self.pruneLayerThreshold = plt
        self.lr                  = LR
        self.trainingMode        = trMode
        self.criterion           = nn.CrossEntropyLoss()

        # drift detection parameters
        self.alphaWarning   = alpha_w
        self.alphaDrift     = alpha_d
        self.driftStatusOld = 0
        self.driftStatus    = 0
        self.driftHistory   = []
        
        # Evolving
        self.growNode   = False
        self.pruneNode  = False
        self.growLayer  = False
        self.pruneLayer = False

        # data
        self.bufferData     = torch.Tensor().float()
        self.bufferLabel    = torch.Tensor().long()
        self.accFmatrix     = deque([])

        # properties
        self.nHiddenLayer = 1
        self.nHiddenNode  = nOutput
    
    def updateNetProperties(self):
        self.nHiddenLayer = len(self.net)
        nHiddenNode = 0
        for _,nett in enumerate(self.net):
            nHiddenNode  += nett.nNodes
        self.nHiddenNode  = nHiddenNode

    def getNetProperties(self):
        for _,nett in enumerate(self.net):
            nett.getNetProperties()
        print('Voting weight: ',self.votingWeight)
            
    # ============================= Evolving mechanism =============================
    def layerGrowing(self):
        if self.driftStatus == 2:
            self.net                     = self.net + [smallAdl(self.net[-1].nNodes,self.net[0].nOutputs,self.net[0].nOutputs)]
            self.votingWeight            = self.votingWeight + [1.0]
            self.dynamicDecreasingFactor = self.dynamicDecreasingFactor + [1.0]
            self.votingWeight            = (self.votingWeight/np.sum(self.votingWeight)).tolist()
            self.averageBias             = self.averageBias + [meanStdCalculator()]
            self.averageVar              = self.averageVar  + [meanStdCalculator()]
            self.winLayerIdx             = len(self.net) - 1
            self.updateNetProperties()
            # print('*** ADD a new LAYER ***')

    def layerPruning(self):
        if np.count_nonzero(self.votingWeight) > 1 and self.driftStatus == 0 and self.driftStatusOld != 2:
            prunedLayerList = []
            nLayer = np.count_nonzero(self.votingWeight)

            # layer pruning identification
            for i in range(0,len(self.yListNoSmax)):
                if self.votingWeight[i] == 0:
                    continue

                for j in range(i+1,len(self.yListNoSmax)):
                    if self.votingWeight[j] == 0:
                        continue

                    A = torch.FloatTensor(self.yListNoSmax[i]).transpose(0,1)
                    B = torch.FloatTensor(self.yListNoSmax[j]).transpose(0,1)
                    nOutput = A.shape[0]
                    MICI = []
                    for k in range(0,nOutput):
                        varianceA = np.var(A[k].tolist())
                        varianceB = np.var(B[k].tolist())
                        corrAB = np.corrcoef(A[k].tolist(),B[k].tolist())[0][1]

                        if (corrAB != corrAB).any():
    #                         print('There is NaN in LAYER pruning')
                            corrAB = 0.0

                        mici = (varianceA + varianceB - np.sqrt((varianceA + varianceB)**2 - 
                                                                4*varianceA*varianceB*(1-corrAB**2)))/2

    #                     print('mici of ',i,'-th layer and ',j,'-th layer and ',k,'-th output is: ',mici)
                        MICI.append(mici)

                    # layer pruning
                    if np.max(np.abs(MICI)) < self.pruneLayerThreshold:
                        # print('layer ',i+1, 'and layer ',j+1, 'are highly correlated with MICI ', np.max(np.abs(MICI)))
                        if self.votingWeight[i] < self.votingWeight[j]:
                            prunedLayerList.append(i)
                            self.votingWeight[i] = 0
                            # print('\\\ hidden LAYER ',i+1, 'is PRUNED ///')
                        else:
                            prunedLayerList.append(j)
                            self.votingWeight[j] = 0
                            # print('\\\ hidden LAYER ',j+1, 'is PRUNED ///')

                        nLayer -= 1
                        if nLayer <= 1:
                            break

            self.removeLastLayer()
            self.updateNetProperties()
            self.winLayerIdentifier()
            self.votingWeight = (self.votingWeight/np.sum(self.votingWeight)).tolist()
    
    def removeLastLayer(self):
        while self.votingWeight[-1] == 0:
            del self.net[-1]
            del self.votingWeight[-1]
            del self.dynamicDecreasingFactor[-1]
            del self.averageBias[-1]
            del self.averageVar[-1]
            # print('### A LAST hidden LAYER is REMOVED ###')
        
    def hiddenNodeGrowing(self):
        if self.winLayerIdx <= (len(self.net)-1):
            copyNet = copy.deepcopy(self.net[self.winLayerIdx])
            copyNet.nodeGrowing()
            self.net[self.winLayerIdx] = copy.deepcopy(copyNet)
            if self.winLayerIdx != (len(self.net)-1):
                copyNextNet = copy.deepcopy(self.net[self.winLayerIdx+1])
                copyNextNet.inputGrowing()
                self.net[self.winLayerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError
        
    def hiddenNodePruning(self):
        if self.winLayerIdx <= (len(self.net)-1):
            copyNet = copy.deepcopy(self.net[self.winLayerIdx])
            copyNet.nodePruning(self.leastSignificantNode)
            self.net[self.winLayerIdx] = copy.deepcopy(copyNet)
            if self.winLayerIdx != (len(self.net)-1):
                copyNextNet = copy.deepcopy(self.net[self.winLayerIdx+1])
                copyNextNet.inputPruning(self.leastSignificantNode)
                self.net[self.winLayerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('--- Hidden NODE No: ',self.leastSignificantNode,' is PRUNED ---')
            self.updateNetProperties()
        else:
            raise IndexError
            
    # ============================= forward pass =============================
    def feedforwardTest(self,x,device = torch.device('cpu')):
        # feedforward to all layers
        with torch.no_grad():
            nData       = x.shape[0]
            y           = torch.zeros(nData,self.net[0].nOutputs)
            yList       = []
            yListNoSmax = []
            hList       = []

            minibatch_data = x.to(device)
            minibatch_data = minibatch_data.type(torch.float)
            tempVar        = minibatch_data

            for iLayer in range(len(self.net)):
                currnet          = self.net[iLayer].network
                obj              = currnet.eval()
                obj              = obj.to(device)
                tempY, tempVar,_ = obj(tempVar)
                hList            = hList + [tempVar.tolist()]
                y                = y + tempY*self.votingWeight[iLayer]
                if self.votingWeight[iLayer] == 0:
                    yList        = yList + [[]]
                else:
                    yList        = yList + [F.softmax(tempY,dim=1).tolist()]
                    yListNoSmax  = yListNoSmax + [tempY.tolist()]
        
            self.scoresTest            = y
            self.yList                 = yList  # output of all layers
            self.yListNoSmax           = yListNoSmax
            self.multiClassProbability = F.softmax(y.data,dim=1)
            self.predictedLabelProbability, self.predictedLabel = torch.max(self.multiClassProbability, 1)
        
    def feedforwardTrain(self,x,device = torch.device('cpu')):
        # feedforward to the winning layer
        
        minibatch_data = x.to(device)
        minibatch_data = minibatch_data.type(torch.float)
        tempVar        = minibatch_data

        for iLayer in range(len(self.net)):
            currnet          = self.net[iLayer].network
            obj              = currnet.train()
            obj              = obj.to(device)
            tempY, tempVar,_ = obj(tempVar)
            if iLayer == self.winLayerIdx:
                self.scoresTrain = tempY
                break
                
    def feedforwardBiasVar(self,x,label_oneHot,device = torch.device('cpu')):
        # feedforward from the input to the winning layer
        # y in one hot vector form, float, already put in device
        with torch.no_grad():
            minibatch_data  = x.to(device)
            minibatch_data  = minibatch_data.type(torch.float)
            minibatch_label = label_oneHot

            tempVar = minibatch_data
            for iLayer in range(len(self.net)):
                currnet               = self.net[iLayer].network
                obj                   = currnet.eval()
                obj                   = obj.to(device)
                tempY, tempVar,tempY2 = obj(tempVar)

                if iLayer == 0:
                    tempVar2          = (tempVar.detach().clone())**2
                else:
                    tempY2,tempVar2,_ = obj(tempVar2)

                if iLayer == self.winLayerIdx:
                    break
            
#             tempVar  = torch.mean(tempVar,dim=0)
#             tempY    = F.softmax(tempY,dim=1)
#             tempY2   = F.softmax(tempY2,dim=1)
#             bias     = torch.mean((tempY - minibatch_label)**2,dim=0)
#             variance = torch.mean(tempY2 - tempY**2,dim=0)
#             bias     = torch.norm(bias)
#             variance = torch.norm(variance)
            tempY    = F.softmax(tempY,dim=1)
            tempY2   = F.softmax(tempY2,dim=1)
            bias     = torch.norm((tempY - minibatch_label)**2)
            variance = torch.norm(tempY2 - tempY**2)

            self.bias = bias.item()
            self.variance = variance.item()
            self.hiddenNodeSignificance = tempVar 
        
    # ============================= Network Evaluation =============================
    def calculateAccuracyMatrices(self,trueClassLabel, labeledDataIdx, labeled = True):
        # accuracy matrix for the whole network
        if labeled:
            self.F_matrix = (self.predictedLabel != trueClassLabel).int().tolist()  # 1: wrong, 0: correct
        else:
            self.F_matrix = (self.predictedLabel[labeledDataIdx] != trueClassLabel[labeledDataIdx]).int().tolist()
        
        # accuracy matrix for each local output
        F_matrixList = []
        if len(self.net) > 1:
            for iLayer in range(len(self.net)):
                if self.votingWeight[iLayer] == 0:
                    F_matrixList = F_matrixList + [[]]
                else:
                    _, predictedLabel = torch.max(torch.FloatTensor(self.yList[iLayer]).data, 1)
                    
                    # 1: wrong, 0: correct
                    F_matrixList = F_matrixList + [(predictedLabel != trueClassLabel).int().tolist()] 
            self.F_matrixList = F_matrixList
    
    def winLayerIdentifier(self):
        self.winLayerIdx = 0
        # idx = np.argmax(np.asarray(votWeight)/(np.asarray(allLoss) + 0.001))
        self.winLayerIdx = np.argmax(np.asarray(self.votingWeight))
            
    def driftDetection(self):
        # need to be modified
        self.driftStatusOld = self.driftStatus
        driftStatus = 0  # 0: no drift, 1: warning, 2: drift

        if np.max(self.F_matrix) != 0:
            
            # Prepare accuracy matrix.
            # combine buffer data, when previous batch is warning
            # F_matrix is the accuracy matrix of the current batch
            if self.driftStatusOld == 1:
                self.F_matrix = self.bufferF_matrix + self.F_matrix

            # combine current and previous feature matrix
            combinedAccMatrix = self.F_matrix

            # prepare statistical coefficient to confirm a cut point
            nData             = len(combinedAccMatrix)
            cutPointCandidate = [int(nData/4),int(nData/2),int(nData*3/4)]
            cutPoint          = 0
            errorBoundF       = np.sqrt((1/(2*nData))*np.log(1/self.alphaDrift))
            miu_F             = np.mean(self.F_matrix)   
            
            # confirm the cut point
            for iCut in cutPointCandidate:
                miu_E       = np.mean(combinedAccMatrix[0:iCut])
                nE          = len(combinedAccMatrix[0:iCut])
                errorBoundE = np.sqrt((1/(2*nE))*np.log(1/self.alphaDrift))
                if (miu_F + errorBoundF) <= (miu_E + errorBoundE):
                    cutPoint = iCut
                    # print('A cut point is detected cut: ', cutPoint)
                    break

            if cutPoint > 0:
                # prepare statistical coefficient to confirm a drift
                errorBoundDrift = ((np.max(combinedAccMatrix) - np.min(combinedAccMatrix))*
                                        np.sqrt(((nData - nE)/(2*nE*nData))*np.log(1/self.alphaDrift)))

                # if np.abs(miu_F - miu_E) >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
                if miu_E - miu_F >= errorBoundDrift:   # This formula is only able to detect drift when the performance decreses
                    # print('H0 is rejected with size: ', errorBoundDrift)
                    # print('Status: DRIFT')
                    driftStatus         = 2
                    self.accFmatrix     = deque([])
                    self.bufferF_matrix = []
                else:
                    # prepare statistical coefficient to confirm a warning
                    errorBoundWarning = ((np.max(combinedAccMatrix) - np.min(combinedAccMatrix))*
                                        np.sqrt(((nData - nE)/(2*nE*nData))*np.log(1/self.alphaWarning)))

                    # if np.abs(miu_F - miu_E) >= errorBoundWarning and self.driftStatusOld != 1:
                    if miu_E - miu_F >= errorBoundWarning and self.driftStatusOld != 1:
                        # print('H0 is rejected with size: ', errorBoundWarning)
                        # print('Status: WARNING')
                        driftStatus = 1
                        self.bufferF_matrix = self.F_matrix

                    else:
                        # print('H0 is NOT rejected')
                        # print('Status: STABLE')
                        driftStatus = 0
            else:
                # confirm stable
                # print('H0 is NOT rejected')
                # print('Status: STABLE')
                driftStatus = 0

        self.driftStatus = driftStatus
        self.driftHistory.append(driftStatus)
        
    def growNodeIdentification(self):
        dynamicKsigmaGrow = 1.3*np.exp(-self.bias) + 0.7
        growCondition1    = (self.averageBias[self.winLayerIdx].minMean + 
                             dynamicKsigmaGrow*self.averageBias[self.winLayerIdx].minStd)
        growCondition2    = self.averageBias[self.winLayerIdx].mean + self.averageBias[self.winLayerIdx].std

        if growCondition2 > growCondition1 and self.averageBias[self.winLayerIdx].count >= 1:
            self.growNode = True
        else:
            self.growNode = False
    
    def pruneNodeIdentification(self):
        dynamicKsigmaPrune = 1.3*np.exp(-self.variance) + 0.7
        pruneCondition1    = (self.averageVar[self.winLayerIdx].minMean + 
                              2*dynamicKsigmaPrune*self.averageVar[self.winLayerIdx].minStd)
        pruneCondition2    = self.averageVar[self.winLayerIdx].mean + self.averageVar[self.winLayerIdx].std
        
        if (pruneCondition2 > pruneCondition1 and not self.growNode and 
            self.averageVar[self.winLayerIdx].count >= 20 and
            self.net[self.winLayerIdx].nNodes > self.net[self.winLayerIdx].nOutputs):
            self.pruneNode = True
            self.findLeastSignificantNode()
        else:
            self.pruneNode = False

    def findLeastSignificantNode(self):
        # find the least significant node in the winning layer
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.leastSignificantNode = torch.argmin(torch.abs(self.hiddenNodeSignificance)).tolist()
    
    def updateBiasVariance(self):
        # calculate mean of bias
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBias[self.winLayerIdx].updateMeanStd(self.bias)
        if self.averageBias[self.winLayerIdx].count < 1 or self.growNode:
            self.averageBias[self.winLayerIdx].resetMinMeanStd()
        else:
            self.averageBias[self.winLayerIdx].updateMeanStdMin()
        
        # calculate mean of variance
        self.averageVar[self.winLayerIdx].updateMeanStd(self.variance)
        if self.averageVar[self.winLayerIdx].count < 20 or self.pruneNode:
            self.averageVar[self.winLayerIdx].resetMinMeanStd()
        else:
            self.averageVar[self.winLayerIdx].updateMeanStdMin()
        
    # ============================= Training ============================= 
    def updateVotingWeight(self):
        if np.count_nonzero(self.votingWeight) > 1:
            for idx in range(0,len(self.votingWeight)):
                currFmat = self.F_matrixList[idx]
                for iData in range(0,len(currFmat)):
                    if currFmat[iData] == 1:  # detect wrong prediction
                        # penalty
                        self.dynamicDecreasingFactor[idx] = np.maximum(self.dynamicDecreasingFactor[idx] - 
                                                                  self.decreasingFactor, self.decreasingFactor)
                        self.votingWeight[idx]            = np.maximum(self.votingWeight[idx]*
                                                                       self.dynamicDecreasingFactor[idx],
                                                                       self.decreasingFactor)
                    elif currFmat[iData] == 0:  # detect correct prediction
                        # reward
                        self.dynamicDecreasingFactor[idx] = np.minimum(self.dynamicDecreasingFactor[idx] + 
                                                                       self.decreasingFactor, 1)
                        self.votingWeight[idx]            = np.minimum(self.votingWeight[idx]*
                                                                       (1 + self.dynamicDecreasingFactor[idx]), 1)

            self.winLayerIdentifier()
            self.votingWeight = (self.votingWeight/np.sum(self.votingWeight)).tolist()
            
    def getTrainableParameters(self,mode=1):
        # mode 1 > only take the winning layer parameter
        # mode 2 > take the winning layer parameter and all previous layer whose voting weight equals 0
        netOptim  = []
        netOptim  = netOptim + list(self.net[self.winLayerIdx].network.parameters())
        if mode == 2:
            for iLayer in range((self.winLayerIdx) - 1, -1, -1):
                if self.votingWeight[iLayer] != 0:
                    break
                netOptim = netOptim + list(self.net[iLayer].network.parameters())
        self.netOptim = netOptim

    def surrogate_loss(self,alpha = 0.00005):
        loss = []
        
        for p in self.netOptim:

            loss.append((alpha/2 * (p)**2).sum())

        return sum(loss)
        
    def training(self,device = torch.device('cpu'),batchSize = 1,epoch = 1):
    
        # shuffle the data
        nData = self.batchData.shape[0]
        
        # label for bias var calculation
        y_biasVar = F.one_hot(self.batchLabel, num_classes = self.net[0].nOutputs).float()
        
        for iEpoch in range(0,epoch):

            shuffled_indices = torch.randperm(nData)

            for iData in range(0,nData,batchSize):
                # load data
                indices                  = shuffled_indices[iData:iData+batchSize]

                minibatch_xTrain         = self.batchData[indices]
                minibatch_xTrain         = minibatch_xTrain.to(device)
                minibatch_xTrain_biasVar = minibatch_xTrain

                minibatch_labelTrain     = self.batchLabel[indices]
                minibatch_labelTrain     = minibatch_labelTrain.to(device)
                minibatch_labelTrain     = minibatch_labelTrain.long()

                if iEpoch == 0:
                    minibatch_label_biasVar = y_biasVar[indices]
                    minibatch_label_biasVar = minibatch_label_biasVar.to(device)
                    
                    if batchSize > 1:
                        minibatch_xTrain_biasVar = torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0)
                        minibatch_label_biasVar  = torch.mean(minibatch_label_biasVar,dim=0).unsqueeze(dim=0)

                    # calculate mean of input
                    # self.averageInput.updateMeanStd(torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0))
                    self.averageInput.updateMeanStd(minibatch_xTrain_biasVar)

                    # get bias and variance
                    outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)   # for Sigmoid activation function
                    self.feedforwardBiasVar(outProbit,minibatch_label_biasVar)             # for Sigmoid activation function
                    # self.feedforwardBiasVar(self.averageInput.mean,minibatch_label_biasVar)  # for ReLU activation function

                    # update bias variance
                    self.updateBiasVariance()

                    # growing
                    self.growNodeIdentification()
                    if self.growNode:
                        self.hiddenNodeGrowing()

                    # pruning
                    if not self.growNode:
                        self.pruneNodeIdentification()
                        if self.pruneNode:
                            self.hiddenNodePruning()

                # declare parameters to be trained
                self.getTrainableParameters(mode = self.trainingMode)
                optimizer = torch.optim.SGD(self.netOptim, lr = self.lr, momentum = 0.95) #, weight_decay = 0.00005)

                # optimizer = torch.optim.SGD(self.netOptim, lr = self.lr, momentum = 0.95)
                
                # if len(self.netOptim) > 4:
                #     optimizer = SGD(self.netOptim, lr = self.lr, momentum = 0.95, weight_decay = 0.00005)
                # else:
                #     optimizer = SGD(self.netOptim, lr = self.lr, momentum = 0.95)

                # forward pass
                self.feedforwardTrain(minibatch_xTrain)
                loss = self.criterion(self.scoresTrain,minibatch_labelTrain)
                
                # loss = self.criterion(self.scoresTrain,minibatch_labelTrain) + self.surrogate_loss()
                
                # if len(self.netOptim) > 4:
                #     loss = self.surrogate_loss()
                # else:
                #     loss = self.criterion(self.scoresTrain,minibatch_labelTrain) + self.surrogate_loss()

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # apply gradient
                optimizer.step()
        
    def trainingDataPreparation(self, batchData, batchLabel, activeLearning = False,
        advSamplesGenrator = False, minorityClassList = None):

        if activeLearning:
            # sample selection
            # MCP: multiclass probability
            sortedMCP,_          = torch.sort(self.multiClassProbability, descending=True)
            sortedMCP            = torch.transpose(sortedMCP, 1, 0)
            sampleConfidence     = sortedMCP[0]/torch.sum(sortedMCP[0:2], dim=0)
            indexSelectedSamples = sampleConfidence <= 0.75
            indexSelectedSamples = (indexSelectedSamples != 0).nonzero().squeeze()

            # selected samples
            batchData  = batchData[indexSelectedSamples]
            batchLabel = batchLabel[indexSelectedSamples]
            # print('selected sample size',batchData.shape[0])

        # training data preparation
        if self.driftStatus == 0 or self.driftStatus == 2:  # STABLE or DRIFT
            # check buffer
            if self.bufferData.shape[0] != 0:
                # add buffer to the current data batch
                self.batchData  = torch.cat((self.bufferData,batchData),0)
                self.batchLabel = torch.cat((self.bufferLabel,batchLabel),0)

                # clear buffer
                self.bufferData  = torch.Tensor().float()
                self.bufferLabel = torch.Tensor().long()
            else:
                # there is no buffer data
                self.batchData  = batchData
                self.batchLabel = batchLabel

        if self.driftStatus == 1:  # WARNING
            # store data to buffer
            # print('Store data to buffer')
            self.bufferData  = batchData
            self.bufferLabel = batchLabel

        # generate adversarial samples for minority class
        if advSamplesGenrator and (self.driftStatus == 0 or self.driftStatus == 2):
            # prepare data
            
            if minorityClassList is not None and len(minorityClassList) != 0:
                
                nIdealData = int(self.batchData.shape[0]/self.net[0].nOutputs)

                # select the minority class data
                # adversarialBatchData  = self.batchData [self.batchLabel == minorityClass]
                # adversarialBatchLabel = self.batchLabel[self.batchLabel == minorityClass]

                # nMinorityClass = adversarialBatchData.shape[0]
                # nMajorityClass = self.batchData.shape[0] - nMinorityClass

                for iClass in minorityClassList:
                    
                    if self.batchData [self.batchLabel == iClass].shape[0] == 0:
                        continue

                    # select the minority class data
                    adversarialBatchData  = self.batchData [self.batchLabel == iClass]
                    adversarialBatchLabel = self.batchLabel[self.batchLabel == iClass]

                    # forward pass
                    adversarialBatchData.requires_grad_()
                    self.feedforwardTrain(adversarialBatchData)
                    lossAdversarial = self.criterion(self.scoresTrain,adversarialBatchLabel)

                    # backward pass
                    lossAdversarial.backward()

                    nMinorityClass  = adversarialBatchData.shape[0]
                    nTimes          = int(nIdealData/nMinorityClass)
                    randConstSize   = adversarialBatchData.detach().clone().repeat(nTimes,1).shape[0]

                    adversarialData = (adversarialBatchData.detach().clone().repeat(nTimes,1) + 
                        0.01*torch.rand(randConstSize,1)*torch.sign(adversarialBatchData.grad).repeat(nTimes,1))
                    adversarialLabel = adversarialBatchLabel.repeat(nTimes)

                    # pdb.set_trace()
                    self.batchData  = torch.cat((self.batchData,adversarialData),0)
                    self.batchLabel = torch.cat((self.batchLabel,adversarialLabel),0)
            else:
                # select all data
                adversarialBatchData  = self.batchData.detach().clone()
                adversarialBatchLabel = self.batchLabel.detach().clone()

                # forward pass
                adversarialBatchData.requires_grad_()
                self.feedforwardTrain(adversarialBatchData)
                lossAdversarial = self.criterion(self.scoresTrain,adversarialBatchLabel)

                # backward pass
                lossAdversarial.backward()

                # get adversarial samples
                randConst = np.round_(0.007*np.abs(np.random.normal()).item(), decimals=4, out=None)
                adversarialBatchData = adversarialBatchData.detach().clone() + 0.007*torch.sign(adversarialBatchData.grad)

                self.batchData  = torch.cat((self.batchData,adversarialBatchData),0)
                self.batchLabel = torch.cat((self.batchLabel,adversarialBatchLabel),0)
                # print('Total sample size',self.batchData.shape[0])
        
    # ============================= Testing ============================= 
    def testing(self,x,label,device = torch.device('cpu')):
        # load data
        x     = x.to(device)
        label = label.to(device)
        label = label.long()
        
        # testing
        start_test          = time.time()
        self.feedforwardTest(x)
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        
        loss                = self.criterion(self.scoresTest,label)
        self.testingLoss    = loss.detach().item()
        correct             = (self.predictedLabel == label).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label
        
    def dispPerformance(self):
        print('Testing Accuracy: {}'.format(self.accuracy))
        print('Testing Loss: {}'.format(self.testingLoss))
        print('Testing Time: {}'.format(self.testingTime))
import numpy as np
import torch
import random
import time
import pdb
from utilsADL import meanStdCalculator, plotPerformance, labeledIdx
from ADLbasic import ADL
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import progressbar
import pdb

def ADLmain(ADLnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1):
    # random seed control
    # np.random.seed(0)
    # torch.manual_seed(0)
    # random.seed(0)
    
    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testing_Loss = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory     = []
    lossHistory         = []
    hiddenNodeHistory   = []
    hiddenLayerHistory  = []
    winningLayerHistory = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode  = []
    nHiddenLayer = []

    # batch loop
    bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    # bar = progressbar.ProgressBar()
    for iBatch in range(0,dataStreams.nBatch):
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        # testing
        ADLnet.testing(batchData,batchLabel)
        if iBatch > 0:
            Y_pred = Y_pred + ADLnet.predictedLabel.tolist()
            Y_true = Y_true + ADLnet.trueClassLabel.tolist()

            Accuracy.append(ADLnet.accuracy)
            testing_Loss.append(ADLnet.testingLoss)

        # if iBatch == 1 or iBatch%50 == 0:
            # print('\n')
            # print(iBatch,'- th batch of:', dataStreams.nBatch)
            # ADLnet.dispPerformance()
            
        # update voting weight
        start_train = time.time()

        lblIdx = labeledIdx(nBatchData, nLabeled)
        ADLnet.calculateAccuracyMatrices(batchLabel, lblIdx, labeled = labeled)
        ADLnet.updateVotingWeight()

        if iBatch > 0:
            # drift detection
            ADLnet.driftDetection()

            # grow layer
            ADLnet.layerGrowing()

            # prune layer identification and pruning
            ADLnet.layerPruning()

        # training data preparation
        if nLabeled < 1:
            ADLnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])
        elif nLabeled == 1:
            ADLnet.trainingDataPreparation(batchData,batchLabel)

        # training
        if ADLnet.driftStatus == 0 or ADLnet.driftStatus == 2:  # only train if it is stable or drift
            ADLnet.training(batchSize = trainingBatchSize, epoch = noOfEpoch)

        end_train = time.time()
        training_time = end_train - start_train

        if iBatch > 0:
            # calculate performance
            testingTime.append(ADLnet.testingTime)
            trainingTime.append(training_time)

            accuracyHistory.append(ADLnet.accuracy)
            lossHistory.append(ADLnet.testingLoss)
            
            # calculate network evolution
            nHiddenLayer.append(ADLnet.nHiddenLayer)
            nHiddenNode.append(ADLnet.nHiddenNode)

            hiddenNodeHistory.append(ADLnet.nHiddenNode)
            hiddenLayerHistory.append(ADLnet.nHiddenLayer)
            winningLayerHistory.append(ADLnet.winLayerIdx+1)

            Iter.append(iBatch)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Testing Loss: ',np.mean(testing_Loss),'(+/-)',np.std(testing_Loss))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))

    print('\n')
    print('=== Final network structure ===')
    ADLnet.getNetProperties()


    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        ADLnet.nHiddenLayer,ADLnet.nHiddenNode]

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenLayerHistory,winningLayerHistory]

    return ADLnet, performanceHistory, allPerformance


def ADLmainId(ADLnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1, nInitLabel = 1000):
    # random seed control
    # np.random.seed(0)
    # torch.manual_seed(0)
    # random.seed(0)
    
    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testing_Loss = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory     = []
    lossHistory         = []
    hiddenNodeHistory   = []
    hiddenLayerHistory  = []
    winningLayerHistory = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode  = []
    nHiddenLayer = []

    nInit = 0
    
    # batch loop
    bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    # bar = progressbar.ProgressBar()
    for iBatch in range(0,dataStreams.nBatch):
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        nInit     += nBatchData

        # testing
        ADLnet.testing(batchData,batchLabel)
        if nInit > nInitLabel:
            Y_pred = Y_pred + ADLnet.predictedLabel.tolist()
            Y_true = Y_true + ADLnet.trueClassLabel.tolist()

            Accuracy.append(ADLnet.accuracy)
            testing_Loss.append(ADLnet.testingLoss)

        # if iBatch == 1 or iBatch%50 == 0:
            # print('\n')
            # print(iBatch,'- th batch of:', dataStreams.nBatch)
            # ADLnet.dispPerformance()
            
        # update voting weight

        start_train = time.time()

        if nInit <= nInitLabel:

            lblIdx = labeledIdx(nBatchData, nLabeled)
            ADLnet.calculateAccuracyMatrices(batchLabel, lblIdx, labeled = labeled)
            ADLnet.updateVotingWeight()

            if iBatch > 0:
                # drift detection
                ADLnet.driftDetection()

                # grow layer
                ADLnet.layerGrowing()

                # prune layer identification and pruning
                ADLnet.layerPruning()

            # training data preparation
            if nLabeled < 1:
                ADLnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])
            elif nLabeled == 1:
                ADLnet.trainingDataPreparation(batchData,batchLabel)

            # training
            if ADLnet.driftStatus == 0 or ADLnet.driftStatus == 2:  # only train if it is stable or drift
                ADLnet.training(batchSize = trainingBatchSize, epoch = noOfEpoch)

        end_train = time.time()
        training_time = end_train - start_train

        if nInit > nInitLabel:
            # calculate performance
            testingTime.append(ADLnet.testingTime)
            trainingTime.append(training_time)

            accuracyHistory.append(ADLnet.accuracy)
            lossHistory.append(ADLnet.testingLoss)
            
            # calculate network evolution
            nHiddenLayer.append(ADLnet.nHiddenLayer)
            nHiddenNode.append(ADLnet.nHiddenNode)

            hiddenNodeHistory.append(ADLnet.nHiddenNode)
            hiddenLayerHistory.append(ADLnet.nHiddenLayer)
            winningLayerHistory.append(ADLnet.winLayerIdx+1)

            Iter.append(iBatch)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Testing Loss: ',np.mean(testing_Loss),'(+/-)',np.std(testing_Loss))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))

    print('\n')
    print('=== Final network structure ===')
    ADLnet.getNetProperties()
    
    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        ADLnet.nHiddenLayer,ADLnet.nHiddenNode]

    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenLayerHistory,winningLayerHistory]

    return ADLnet, performanceHistory, allPerformance
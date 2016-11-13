import numpy as np
import matplotlib.pyplot as plt
import math

class KD_Node:
    def __init__(self, parent, samples):
        self.parent = parent
        self.children = ()
        self.sample = samples
        self.bbMin = np.amin(samples[:,:-1], axis=0)
        self.bbMax = np.amax(samples[:,:-1], axis=0)

def buildTree(samples, dimIdx=0, parent=None):
    l = len(samples)
    if l == 1:
        return KD_Node(parent, samples)
    elif l>0:
        mu = np.mean(samples[:, dimIdx])
        samplesL = samples[samples[:,dimIdx] < mu]
        samplesR = samples[samples[:,dimIdx] >= mu]
        if len(samplesL) == 0 or len(samplesR) == 0:
            samplesL = samples[0:l/2,:]
            samplesR = samples[l/2:,:]
        dimIdx = dimIdx + 1
        if dimIdx >= samples.shape[1] - 1:
            dimIdx = 0
        ret = KD_Node(parent, samples)
        ret.children = (buildTree(samplesL, dimIdx, ret), buildTree(samplesR, dimIdx, ret))
        return ret
    
def knn_search(node, testSample, k, outVals):
    worstDistToBB = 0
    maxDistFromSample = 0
    idxWithGreatestDist = 0
    
    for i in range(len(outVals)):
        outVal = outVals[i][0,:-1]
        pointOnBBBorder = np.maximum(np.minimum(outVal, node.bbMax), node.bbMin)
        worstDistToBB = max(worstDistToBB, np.linalg.norm(pointOnBBBorder-testSample))
        distToTestSample = np.linalg.norm(outVal-testSample)
        if maxDistFromSample < distToTestSample:
            idxWithGreatestDist = i
            maxDistFromSample = distToTestSample
        
    if len(outVals) < k or worstDistToBB < maxDistFromSample:
        if len(node.children) == 0:
            if len(outVals) < k:
                outVals.append(node.sample)
            else:
                outVals[idxWithGreatestDist] = node.sample
        else:
            knn_search(node.children[0], testSample, k, outVals)
            knn_search(node.children[1], testSample, k, outVals)
        
def aufgabe1(training, testing):
    root = buildTree(training)
    for k in [1,3,5]:
        correctClassified = 0
        for sample in testing:
            nn = []
            knn_search(root, np.array([sample[:-1]]), k, nn)
            estimatedClass = 0
            for n in nn:
                estimatedClass += n[0,-1]
            estimatedClass = round(float(estimatedClass) / float(len(nn)))
            if estimatedClass == sample[-1]:
                correctClassified += 1
        print("correctly classified: {} total: {} quality (k={}): {}".format(correctClassified, len(testing), k, float(correctClassified) / float(len(testing))))
        
def aufgabe2(training):
    xx = np.linspace(-10, 500, 200)
    densities = np.zeros((len(training), xx.shape[0]))
    for weight in range(6):
        samples = training[training[:,2] == weight]
        mu = np.mean(samples[:,1])
        v_helper = np.mat(samples)[:,1] - mu
        var1 = v_helper.transpose() * v_helper * 1. / (samples.shape[0]-1) # Stichprobenvarianz
        var2 = v_helper.transpose() * v_helper * 1. / samples.shape[0]     # Varianz, wie in der VL
        apriori = float(len(samples)) / float(len(training))
        print("weight: {} apriori: {} mean: {} var1: {} var2: {}".format(weight, apriori, mu, var1[0,0], var2[0,0]))
        densities[weight] = math.sqrt(2*math.pi*var1) * np.exp(-.5*(xx-mu)*(xx-mu) / var1) * apriori
    
    # gaussverteilungen
    handles = []
    for weight in range(6):
        h, = plt.plot(xx, densities[weight,:], label=str(weight))
        handles.append(h)
    plt.legend(handles)
    plt.show()
    
    # wahrscheinlichkeiten P(Weight|X)
    handles = []
    for weight in range(6):
        h, = plt.plot(xx, densities[weight,:] / np.sum(densities, axis=0), label=str(weight))
        handles.append(h)
    plt.legend(handles)
    plt.show()
    
    
    
if __name__ == "__main__":
    training = np.genfromtxt('chickwts_training.csv', delimiter=',')
    testing = np.genfromtxt('chickwts_testing.csv', delimiter=',')
    
    aufgabe2(training)
    
    ## mit ID
    aufgabe1(training, testing)
    permutation = np.array([0,2,1])
    aufgabe1(training[:,permutation], testing[:,permutation])
    
    ## ohne id
    training = np.genfromtxt('chickwts_training.csv', delimiter=',')[:,1:]
    testing = np.genfromtxt('chickwts_testing.csv', delimiter=',')[:,1:]
    
    aufgabe1(training, testing)
    permutation = np.array([1,0])
    aufgabe1(training[:,permutation], testing[:,permutation])
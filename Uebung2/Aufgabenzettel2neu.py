# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:16:47 2016

@author: Lui
"""


import numpy as np
import csv

test_Matrix_Digit3 = [];
test_Matrix_Digit5 = [];
test_Matrix_Digit7 = [];
test_Matrix_Digit8 = [];

f = open( 'zip.test' , "r");
csv_f = csv.reader(f,delimiter=' ')
for row in csv_f:
    if int(row[0])==3:
        splitRow = [ float(row[i]) for i in range(1,len(row)) ]
        test_Matrix_Digit3.append( splitRow )

f = open( 'zip.test' , "r");
csv_f = csv.reader(f,delimiter=' ')
for row in csv_f:
    if int(row[0])==5:
        splitRow = [ float(row[i]) for i in range(1,len(row)) ]
        test_Matrix_Digit5.append( splitRow )


f = open( 'zip.test' , "r");
csv_f = csv.reader(f,delimiter=' ')
for row in csv_f:
    if int(row[0])==7:
        splitRow = [ float(row[i]) for i in range(1,len(row)) ]
        test_Matrix_Digit7.append( splitRow )

f = open( 'zip.test' , "r");
csv_f = csv.reader(f,delimiter=' ')
for row in csv_f:
    if int(row[0])==8:
        splitRow = [ float(row[i]) for i in range(1,len(row)) ]
        test_Matrix_Digit8.append( splitRow )
f.close()
test_Matrix_Digit3= np.matrix(test_Matrix_Digit3)
test_Matrix_Digit5= np.matrix(test_Matrix_Digit5)
test_Matrix_Digit7= np.matrix(test_Matrix_Digit7)
test_Matrix_Digit8= np.matrix(test_Matrix_Digit8)

   
def readTrainingSet( fileName ):
    f = open( fileName , "r");
    csv_f = csv.reader(f)
    dataMatrix = [];
    for row in csv_f:
        splitRow = [ float(row[i]) for i in range(len(row)) ]
        dataMatrix.append( splitRow )
    f.close()
    return np.matrix(dataMatrix)
    

def composeMatrices((A,B)):
    
    return np.concatenate((A,B))


def solveLinearRegression(A,B):
    number_of_rowsA = len(A)
    number_of_rowsB = len(B)
    
    y_A = np.ones(number_of_rowsA)
    y_B = (-1)*np.ones(number_of_rowsB)
    y   = np.matrix(composeMatrices((y_A,y_B)))
    
    C = composeMatrices((A,B))
    return (C.T*C).I*C.T*y.T   
    
def classifyObjects(train_Set1, train_Set2, test_Set1, test_Set2):
    
    train_Matrix1 = readTrainingSet(train_Set1)
    train_Matrix2 = readTrainingSet(train_Set1)
 
    classifier = solveLinearRegression(train_Matrix1, train_Matrix2).T
    
    test_Matrix = composeMatrices((test_Set1, test_Set2)).T
    
    test_objects1 = len(test_Set1)
    test_objects2 = len(test_Set2)
    
    classVector1 = np.matrix(np.ones(test_objects1)).T
    classVector2 = np.matrix((-1)*np.ones(test_objects2)).T
    
    classVector = np.matrix(composeMatrices((classVector1,classVector2))).T
    
    classified_Objects = np.sign(classifier*test_Matrix)
    
    cc_proportion = ((classVector*classified_Objects.T).item()+len(classVector.T))/(2*len(classVector.T))
    
    return cc_proportion
    
print(classifyObjects('train3.3','train7.7',test_Matrix_Digit3,test_Matrix_Digit7))



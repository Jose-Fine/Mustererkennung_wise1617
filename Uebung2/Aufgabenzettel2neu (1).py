# Abgabe Uebung 2 Mustererkennung
# Leo & Josephine Mertens

import numpy as np
import csv

def read_test_data():
    # prepare data
    test_Matrix_Digit3 = [];
    test_Matrix_Digit5 = [];
    test_Matrix_Digit7 = [];
    test_Matrix_Digit8 = [];

    # read data from file
    f = open( 'zip.test' , "r");
    csv_f = csv.reader(f,delimiter=' ')
    for row in csv_f:
        if int(row[0])==3:
            splitRow = [ float(row[i]) for i in range(1,len(row)) ]
            test_Matrix_Digit3.append( splitRow )
        elif int(row[0])==5:
            splitRow = [ float(row[i]) for i in range(1,len(row)) ]
            test_Matrix_Digit5.append( splitRow )
        elif int(row[0])==7:
            splitRow = [ float(row[i]) for i in range(1,len(row)) ]
            test_Matrix_Digit7.append( splitRow )
        elif int(row[0])==8:
            splitRow = [ float(row[i]) for i in range(1,len(row)) ]
            test_Matrix_Digit8.append( splitRow )
    f.close()

    test_Mat3 = np.matrix(test_Matrix_Digit3)     
    test_Mat5 = np.matrix(test_Matrix_Digit5)
    test_Mat7 = np.matrix(test_Matrix_Digit7)
    test_Mat8 = np.matrix(test_Matrix_Digit8)

    arr = [test_Mat3, test_Mat5, test_Mat7, test_Mat8]

    print(test_Matrix_Digit3)
    return arr

   
def readTrainingSet( fileName ):
    f = open( fileName , "r");
    csv_f = csv.reader(f)
    dataMatrix = [];
    for row in csv_f:
        splitRow = [ float(row[i]) for i in range(len(row)) ]
        dataMatrix.append( splitRow )
    f.close()
    return np.matrix(dataMatrix)
    

def composeMatrices(A,B):
    
    return np.concatenate((A,B))


def solveLinearRegression(A,B):
    number_of_rowsA = len(A)
    number_of_rowsB = len(B)
    
    y_A = np.ones(number_of_rowsA)
    y_B = (-1)*np.ones(number_of_rowsB)
    y   = np.matrix(composeMatrices((y_A,y_B)))
    
    C = composeMatrices((A,B))
    if np.linalg.det(C.T*C)!=0:
        return (C.T*C).I*C.T*y.T 
    else:
        n = len(C.T*C)
        eps = 0.0001
        i = 0
        while np.linalg.det(C.T*C +eps*2**i*np.identity(n))==0:
            i=i+1
        return (C.T*C+eps*2**i*np.identity(n)).I*C.T*y.T
            

# 	
def classifyObjects(train_Set1, train_Set2, test_Set1, test_Set2):    
	
    train_Matrix1 = readTrainingSet(train_Set1)
    train_Matrix2 = readTrainingSet(train_Set2)
 
    # calculate binary classificator
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

def main():
    # get the input data
    Mat_list = read_test_data()
	
    # classifie training data and given testdata
    print(classifyObjects('train.3','train.5',Mat_list[0],Mat_list[1]))
    print(classifyObjects('train.3','train.7',Mat_list[0],Mat_list[2]))
    print(classifyObjects('train.3','train.8',Mat_list[0],Mat_list[3]))
    print(classifyObjects('train.5','train.7',Mat_list[1],Mat_list[2]))
    print(classifyObjects('train.5','train.8',Mat_list[1],Mat_list[3]))
    print(classifyObjects('train.7','train.8',Mat_list[2],Mat_list[3]))
	
main()
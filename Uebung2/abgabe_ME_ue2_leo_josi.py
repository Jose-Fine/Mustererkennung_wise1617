# Abgabe Uebung 2 Mustererkennung
# Leonardo Balestrieri & Josephine Mertens

import numpy as np
import csv

# Einlesen der Testdaten. Speichere jeden Datensatz(anhand der Label unterscheidbar) als n x m Matrix
# Return: Liste der Matrizen
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

    return arr

# Einlesen des Testdatensatzes und speichere diesen als Mat und gebe ihn zurück   
def readTrainingSet( fileName ):
    f = open( fileName , "r");
    csv_f = csv.reader(f)
    dataMatrix = [];
    for row in csv_f:
        splitRow = [ float(row[i]) for i in range(len(row)) ]
        dataMatrix.append( splitRow )
    f.close()
    return np.matrix(dataMatrix)
    
# Konkatenation
def composeMatrices(A,B):
    
    return np.concatenate((A,B))


def solveMultivariatNormalDistribution(A,B):
    number_of_rowsA = len(A)
    number_of_rowsB = len(B)
    
	
    y_A = np.ones(number_of_rowsA)
    y_B = (-1)*np.ones(number_of_rowsB)
    y   = np.matrix(composeMatrices(y_A,y_B))
    
	# Fall das Determinante == 0 ist. Füge dann Rauschen hinzu
    C = composeMatrices(A,B)
	
	
	# Berechne lineare regression, wenn Determinate ungleich 0
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
	# prepare data
    train_Matrix1 = readTrainingSet(train_Set1)
    train_Matrix2 = readTrainingSet(train_Set2)
	
    # Erzeuge alle k-means und speichere diese in Liste (256 Stück für train_Set1)
    kMeans_train_Set1 = kMeansAll(train_Set2)
    kMeans_train_Set2 = kMeansAll(train_Set2)
 
    # Berechne Kovarianzmatrix der Trainingsdatensätze
    covariance_Matrix1 = calcCovarianceMatrix(train_Set1, kMeans_train_Set1)
    covariance_Matrix2 = calcCovarianceMatrix(train_Set2, kMeans_train_Set2)

    # calculate binary classificator
	# ToDo: Eingabeparameter anpassen.kmeans und cavariancematrix müssen übergeben werden
    classifier = solveMultivariatNormalDistribution(train_Matrix1, train_Matrix2).T

    test_Matrix = composeMatrices(test_Set1, test_Set2).T
    
    test_objects1 = len(test_Set1)
    test_objects2 = len(test_Set2)
    
    classVector1 = np.matrix(np.ones(test_objects1)).T
    classVector2 = np.matrix((-1)*np.ones(test_objects2)).T
    
    classVector = np.matrix(composeMatrices(classVector1,classVector2)).T
    
    classified_Objects = np.sign(classifier*test_Matrix)
    
    cc_proportion = ((classVector*classified_Objects.T).item()+len(classVector.T))/(2*len(classVector.T))
    
    return cc_proportion 

def main():
    # get the input data
    Mat_list = read_test_data()
	
    # classifie training data and given testdata
    #print('Klassifikationsgüte für Train3 und Train5:  ' + repr(classifyObjects('train.3','train.5',Mat_list[0],Mat_list[1])))
    #print('Klassifikationsgüte für Train3 und Train7:  ' + repr(classifyObjects('train.3','train.7',Mat_list[0],Mat_list[2])))
    #print('Klassifikationsgüte für Train3 und Train8:  ' + repr(classifyObjects('train.3','train.8',Mat_list[0],Mat_list[3])))
    #print('Klassifikationsgüte für Train5 und Train7:  ' + repr(classifyObjects('train.5','train.7',Mat_list[1],Mat_list[2])))
    #print('Klassifikationsgüte für Train5 und Train8:  ' + repr(classifyObjects('train.5','train.8',Mat_list[1],Mat_list[3])))
    #print('Klassifikationsgüte für Train7 und Train8:  ' + repr(classifyObjects('train.7','train.8',Mat_list[2],Mat_list[3])))
	


# Helperfunction
# Determinante 0.022 für Rauschen

#Berechne alle k-means der Datenmatrix und gebe sie als Liste/Matrix zurück
# works fine
def kMeansAll(matrix):
    # addiere alle Werte in der Spalte
    means = []
    [cols,rows] = matrix.shape
    # iteriere über alle Spalten
    for i in range(rows):
        sigma = 0
        # berechne mean der Spalte mittels Addition der Zeilenwerte
        for j in range(cols):
            sigma += matrix[j,i]
        means.append(sigma/cols)
    return means

# Berechne Kovarianzmatrix der Trainingsdatensätze
# works fine
def calcCovarianceMatrix(matrix_trainSet, kMeans_trainSet):
    # generiere quadratische Matrix 
    [cols,rows] = matrix_trainSet.shape
    cov_mat = np.zeros(shape=(rows,rows))

    # fülle covariance mat mit Werten
    for i in range(rows):
        cov_tmp = 0
        vector1 = matrix_trainSet[:,i:i+1]
        for j in range(rows):   
            # symetric matrix, only fill one side
            if (j >= i):
                vector2 = matrix_trainSet[:,j:j+1]
                cov_value = cov(vector1,vector2, kMeans_trainSet[i], kMeans_trainSet[j])

                cov_mat[j,i] = 1/cols * cov_value
    return(cov_mat)

# Helper zur Berechnung der inneren Summe
# works fine
def cov(vector1,vector2, kMean1, kMean2):
    res = 0
    for i in range(len(vector1)):
        tmp1 = vector1[i] - kMean1
        tmp2 = vector2[i] - kMean2
        res = res + tmp1*tmp2
    return res

#main()

#testmain
def testMain():
    # prepare data
    train_Matrix1 = readTrainingSet('C:/Users/Fine/Documents/mustererkennung_ws1617/u02/data/train.3')
    train_Matrix2 = readTrainingSet('C:/Users/Fine/Documents/mustererkennung_ws1617/u02/data/train.5')

    # Erzeuge alle k-means und speichere diese in Liste
    kMeans_train_Set1 = kMeansAll(train_Matrix1)
    kMeans_train_Set2 = kMeansAll(train_Matrix2)

    # Berechne Kovarianzmatrix der Trainingsdatensätze
    covariance_Matrix1 = np.cov(train_Matrix1)

    # Something went wrong. Result is wrong so we use numpy.cov
    #covariance_Matrix1 = np.matrix(calcCovarianceMatrix(train_Matrix1, kMeans_train_Set1))
    covariance_Matrix2 = np.cov(train_Matrix2)

    # next step: calc determinante of each covariance matrix which is greater than 0
    optimized_cova_mat1 = proofDetOfCovarianceMatrix(covariance_Matrix1)
    det1 = np.linalg.det(optimized_cova_mat1)
    print('determinante 1 : ' + repr(det1))
    optimized_cova_mat2 = proofDetOfCovarianceMatrix(covariance_Matrix2)
    det2 = np.linalg.det(optimized_cova_mat2)
    print('determinante 2 : ' + repr(det2))
	
	# now we can calculate multivariate normal distribution of each class
    #solveMultivariatNormalDistribution()
	
def proofDetOfCovarianceMatrix(matrix):
    # Prüfe ob determinante null ist, falls ja füge rauschen auf Diagonale hinzu
    if np.linalg.det(matrix)!=0:
        return matrix
    else:
        n = len(matrix)
        eps = 0.0001
        i = 0
        while np.linalg.det(matrix +eps*2**i*np.identity(n))==0:
            i=i+1
        return (matrix+eps*2**i*np.identity(n))

	

testMain()
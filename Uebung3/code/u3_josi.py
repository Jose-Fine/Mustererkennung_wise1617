# Abgabe Uebung 3 Mustererkennung
# Gauss Klassifikator
# Josephine Mertens

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

# caclulate mutivariat normal distribution and return probability value
def solveMultivariatNormalDistribution(test_data_mat, cova_mat, kMeans):
    # calculate determinante
    det= np.linalg.det(cova_mat)

    # subset of algorithm 
    a = 1/(np.sqrt(((2*np.pi)**len(test_data_mat))*det))
    b = test_data_mat - np.matrix(kMeans)
    c = np.e**(-0.5 * b.T * np.linalg.inv(cova_mat) * b)
   
    return c

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
# Fehler in Berechnung. Unterschiedliches Ergbenis und sehr lahmarschig..
def calcCovarianceMatrix(matrix_trainSet, kMeans_trainSet):
    # generiere quadratische Matrix 
    [cols,rows] = matrix_trainSet.shape
    cov_mat = np.zeros(shape=(rows,rows))

    # fülle covariance mat mit Werten
    for i in range(rows):
        cov_tmp = 0
        vector1 = matrix_trainSet[:,i:i+1]
        for j in range(rows):   
            # symmetric matrix, only fill one side
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
def classifyObjects(train_Set1, train_Set2, test_Set1, test_Set2):
    # prepare data
    train_Matrix1 = readTrainingSet(train_Set1)
    train_Matrix2 = readTrainingSet(train_Set2)

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
    optimized_cova_mat2 = proofDetOfCovarianceMatrix(covariance_Matrix2)
	
	# now we can calculate multivariate normal distribution of each class
    prob_x_class1 = solveMultivariatNormalDistribution(test_Set1, optimized_cova_mat1, kMeans_train_Set1)
    print('Wahrscheinlichkeit erster Datensatz: ' + repr(prob_x_class1))
    prob_x_class2 = solveMultivariatNormalDistribution(test_Set2, optimized_cova_mat2, kMeans_train_Set2)
    print('Wahrscheinlichkeit zweiter Datensatz: ' + repr(prob_x_class2))


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

def main():
    # get the input data
    Mat_list = read_test_data()
	
    # classifie training data and given testdata
    print('Klassifikationsgüte für Train3 und Train5:  ' + repr(classifyObjects('train.3','train.5',Mat_list[0],Mat_list[1])))
    #print('Klassifikationsgüte für Train3 und Train7:  ' + repr(classifyObjects('train.3','train.7',Mat_list[0],Mat_list[2])))
    #print('Klassifikationsgüte für Train3 und Train8:  ' + repr(classifyObjects('train.3','train.8',Mat_list[0],Mat_list[3])))
    #print('Klassifikationsgüte für Train5 und Train7:  ' + repr(classifyObjects('train.5','train.7',Mat_list[1],Mat_list[2])))
    #print('Klassifikationsgüte für Train5 und Train8:  ' + repr(classifyObjects('train.5','train.8',Mat_list[1],Mat_list[3])))
    #print('Klassifikationsgüte für Train7 und Train8:  ' + repr(classifyObjects('train.7','train.8',Mat_list[2],Mat_list[3])))
	
	

main()
# Uebung 1 Mustererkennung
# Josephine Mertens

# Aufgabe 1
# 1.1 calculate kd_tree with chickwts_training.csv

import csv
import operator
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib.mlab as mlab
from tkinter.filedialog import askopenfilename
from tkinter import Tk

typeOfFeed = ['horsebean','linseed','soybean','sunflower','meatmeal','casein']

# helper function to save csv data as list of tuples
def getTreeData():
    #get filepath with tkinter module
    Tk().withdraw()
    filepath = askopenfilename()
    data_list = []

    with open(filepath,'r') as infile:
        for line in infile:
            data_list.append(tuple(line.strip().split(',')))
        return data_list
    
# generate kd tree Quelle:https://gist.github.com/tompaton/863301
class KDTree(object):
    def __init__(self, tupel_list, _depth=0):
        if tupel_list:
            self.axis = _depth % len(tupel_list[0])

            # sort tupel list
            tupel_list = sorted(tupel_list, key= lambda point: point[self.axis])
            median = len(tupel_list) // 2

            self.location = tupel_list[median]
            self.child_left = KDTree(tupel_list[:median], _depth +1)
            self.child_right = KDTree(tupel_list[median +1:], _depth +1)
        else:
            self.axis = 0
            self.location = None
            self.child_left = None
            self.child_right = None

    def __repr__(self):
        if self.location:
            return "(%d, %s, %s, %s)" % (self.axis, repr(self.location), repr(self.child_left), repr(self.child_right))
        else:
            return "None"

    def closest_point(self, point, _best=None):
        if self.location is None:
            return _best
 
        if _best is None:
            _best = self.location
 
        # consider the current node
        if distance(self.location, point) < distance(_best, point):
            _best = self.location
 
        # search the near branch
        _best = self._child_near(point).closest_point(point, _best)
 
        # search the away branch - maybe
        if self._distance_axis(point) < distance(_best, point):
            _best = self._child_away(point).closest_point(point, _best)
 
        return _best
    
    def _distance_axis(self, point):
        axis_point = list(point)
        axis_point[self.axis] = self.location[self.axis]
        return distance(tuple(axis_point), point)

    def _child_near(self, point):
        """
        Either left or right child, whichever is closest to the point
        """
        if point[self.axis] < self.location[self.axis]:
            return self.child_left
        else:
            return self.child_right

    def _child_away(self, point):
        """
        Either left or right child, whichever is furthest from the point
        """
        if self._child_near(point) is self.child_left:
            return self.child_right
        else:
            return self.child_left 

def distance(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

## Ende Quelle



# 1.2 k -NN algorithm with chickwts_testing.csv
# split the input data in two classes. One for feedtype and the other for weight

def split_data(data, list_feedtype=[], list_weight=[]):    
    for it in range(len(data)):
        tup = []
        tup.append(data[it][0])
        tup.append(data[it][2])
        list_feedtype.append(tuple(tup))
        list_weight.append(data[it][0:2])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)



def main():
    # prepare data for k-NN algorithm
    data = getTreeData()
    tree = KDTree(data)

    print('KDTree of chickwts_training data. First value represents axis: 0 = x, 1 = y, 2 = z .\n')
    print(tree)

    feed_list = []
    weight_list = []
    split_data(data, feed_list, weight_list)
    ## todo ##
    ## k-NN ##

    #task 2.1
    print('Normal distribution with a posteriori probability')
    normal_distribution_main(data)

    #task 2.2
    print('Normal distribution with a posteriori probability \n')
    a_posteriori(data)


# Aufgabe 2
# 2.1 calculate normal distribution

def normal_distribution_main(data):
    #prepare data
    class_horsebean = []
    class_linseed = []
    class_soybean = []
    class_sunflower = []
    class_meatmeal = []
    class_casein = []   
    n = len(data)

    for i in range(len(data)):
        if(data[i][2] == '0'):
            class_horsebean.append(data[i][1])
        elif(data[i][2] == '1'):
            class_linseed.append(data[i][1])
        elif(data[i][2] == '2'):
            class_soybean.append(data[i][1])
        elif(data[i][2] == '3'):
            class_sunflower.append(data[i][1])
        elif(data[i][2] == '4'):
            class_meatmeal.append(data[i][1])
        else:
            class_casein.append(data[i][1])

    x = np.linspace(-50,200,100)
	
	posterior = lambda n, h, q: (n+1)*st.binom(n,q).pmf(h)
    f, (ax1,ax2) = plt.subplots(2)
	
    ax1.plot(x, normal_distribution(x,class_horsebean, mu(n, class_horsebean)), 'r')
    ax1.plot(x, normal_distribution(x,class_linseed, mu(n, class_linseed)), 'g')
    ax1.plot(x, normal_distribution(x,class_sunflower, mu(n, class_sunflower)), 'b')
    ax1.plot(x, normal_distribution(x,class_meatmeal, mu(n, class_meatmeal)), 'y')
    ax1.plot(x, normal_distribution(x,class_casein, mu(n, class_casein)), 'c')
    ax1.plot(x, normal_distribution(x,class_soybean, mu(n, class_soybean)), 'm')

    ax1.set_title('Normal Distribution with A Priori')
    
    ax2.plot(x, normal_distribution(x,class_horsebean, mu(n, class_horsebean)), 'r')
    ax2.plot(x, normal_distribution(x,class_linseed, mu(n, class_linseed)), 'g')
    ax2.plot(x, normal_distribution(x,class_sunflower, mu(n, class_sunflower)), 'b')
    ax2.plot(x, normal_distribution(x,class_meatmeal, mu(n, class_meatmeal)), 'y')
    ax2.plot(x, normal_distribution(x,class_casein, mu(n, class_casein)), 'c')
    ax2.plot(x, normal_distribution(x,class_soybean, mu(n, class_soybean)), 'm')
	
    ax2.set_title('Normal Distribution with A Posteriori')
	
    plt.show()
    

def normal_distribution(x,data_set, mue):
    sigma = np.sqrt(variance(data_set))
    
    tmp = sigma * np.sqrt(2*np.pi)
    tmp1 = (x - mue)/sigma
    tmp2 = (-0.5)*(tmp1**2)
    tmp3 = 1/tmp
    res = np.exp(tmp2)

    return tmp3 * res   


# Helper functions

# Erwartungswert
def mu(N, sub_data):
    mu_n = 0
    for x in range(len(sub_data)):
        tmp = int(sub_data[x])
        num = 0
        for y in range(len(sub_data)):
            if(tmp == int(sub_data[y])):
                num = num +1
        p_tmp = (num/N)
        mu_n += p_tmp * tmp        
    return mu_n

# Varianz delta**2
def variance(var_list):
    suma = 0
    av = average(var_list)
    for s in range(len(var_list)-1):
        suma += (int(var_list[s])- av)**2
    return float(suma/av)    
    
# Durchschnitt
def average(var_list):
    suma = 0
    for s in range(len(var_list)):
        suma += int(var_list[s])
    return float(suma/len(var_list))

# 2.2 A-Priori Probability
# Laplace rule, every event has the same probability
def a_priori(data):
    # probability for every event is 1/n , n: num of group member
    n = len(data)
    # P(c0)
    p_c0= p_c1= p_c2= p_c3= p_c4= p_c5 = 0
    for x in range(len(data)):
        if (data[x][2]=='0'):
            p_c0 += int(data[x][1])
        elif(data[x][2]=='1'):
            p_c1 += int(data[x][1])
        elif(data[x][2]=='2'):
            p_c2 += int(data[x][1])
        elif(data[x][2]=='3'):
            p_c3 += int(data[x][1])
        elif(data[x][2]=='4'):
            p_c4 += int(data[x][1])
        elif(data[x][2]=='5'):
            p_c5 += int(data[x][1])
            
    print("\nP(Foodclass 0) is : " + repr(1/n *p_c0))
    print("P(Foodclass 1) is : " + repr(1/n *p_c1))
    print("P(Foodclass 2) is : " + repr(1/n *p_c2))
    print("P(Foodclass 3) is : " + repr(1/n *p_c3))
    print("P(Foodclass 4) is : " + repr(1/n *p_c4))
    print("P(Foodclass 5) is : " + repr(1/n *p_c5))

# 2.3 A-Posteriori Probability
def a_posteriori(data):
    #real probability of class
    problist =[]
    for i in range(len(data)):
        num = 0
        tmp = data[i][1]
        for it in range(len(data)):
            if (tmp == data[it][1]):
                num += 1
        prob = num/len(data)
        problist.append(tuple([prob,data[i][2]]))
        
        
        
            

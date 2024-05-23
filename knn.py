import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import operator
import math
import os
def Lp(data1,data2,p):
    Lp=0
    for i in range(len(data1)):
        Lp+=pow(abs(data1[i]-data2[i]),p)
    Lp=pow(Lp,1/p)
    return Lp
def I(gui_data,phi_data):
    sum = 0
    for i in range(len(gui_data)):
        if(gui_data[i] == phi_data[i]):
            sum  += 1
    return sum

class KNN:
    def  __init__(self,k):
        self.k=k

    def predict(self,dataset,k,p,testdata):
        dist = []

        xdata = []
        ydata =[]
        phi_x = testdata[:len(testdata)-1]
        phi_y = testdata[len(testdata)-1]
        dist_k = {}
        train_dataset = np.array(dataset)
        for data in train_dataset:
            xdata.append(data[:len(data)-1] )
            ydata.append(data[len(data)-1])

        for i in range(len(xdata)):
            dist.append((Lp(xdata[i],phi_x,p),ydata[i]))
        dist = np.array(dist)
        print(dist)
        for i in range(k):
            index =np.argmin(dist[:,0])
            label = dist[index][1]
            print(dist[index])
            if label in dist_k:
                dist_k[label] += 1
            else:
                dist_k[label] = 1
            dist = np.delete(dist, index, axis=0)
        print(dist_k)
        ans = max(dist_k.items(), key=operator.itemgetter(1))[0]
        return ans


import pandas as pd
filename ='/Users/wangzhizhou/Code Directory/统计数学方法/KNN/DataSet/iris_all.csv'

import numpy as np
mydata = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0)
obj = KNN(4)
print(obj.predict(mydata,4,2,[5.4,3,4.5,1.5,0]))

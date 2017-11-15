import sys
import numpy as np
filename = 'data_singlevar.txt'
X=[]
y=[]
with open(filename,'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)
#eigendogs

import numpy as np
import scipy as sp
from scipy import misc
import os
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import numpy.linalg as ln

def display(dogs,m,n):
    for d in dogs:
        pylab.figure()
        pylab.imshow(d.reshape(m,n), cmap=plt.cm.gray)
        
    pylab.show()

facedir = "./dogs"

facefiles = os.listdir(facedir)
nfaces = len(os.listdir(facedir))

A = []
AA = []

m,n = np.array(Image.open(facedir + "/" + facefiles[1]).convert("L")).shape


img = ""
for f in facefiles:
    if f[0] != '.':
        img = np.array(Image.open(facedir + "/" + f).convert("L"))
        A.append(img.flatten())
        AA.append(img.flatten()/nfaces)

#find average face
meanA = sum(AA)

#check out the avg face
plt.imshow(meanA.reshape(m,n), cmap=plt.cm.gray)
plt.show()
#subtract average face from each face

A -= meanA

#find the eigenvectors of the nfaces x nfaces matrix (hermitian?)
#apply them to the dictionary matrix A
L = np.dot(A,A.T)
w,v = np.linalg.eigh(L)
eigendogs = np.dot(A.T,v).T

display(eigendogs[0:3],m,n)



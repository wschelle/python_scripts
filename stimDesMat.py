# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:58:57 2022

@author: WauWter
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nipy.modalities.fmri import hrf, utils
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from Python.python_scripts.wmcsv_convolve import wmcsv_convolve


ntrial=32
delay=0
fps=60
ncons=4
nconp=4
ncont=ncons+nconp
TR=1.
hrflen=25
timestep=0.1

inputdur=2.5
outputdur=1.5

rdir='c:/Users/wousch/Documents/PsychoPy/Exp/wmgating3f_fMRI/data/'
#rdir='c:/Users/woute/Documents/tasks/WMtask/wmgating3f/data/'
fname1='_WMgate_2022_Dec_08_1438.csv'
fname2='_WMgate_2022_Dec_08_1415.csv'
fname3='_WMgate_2022_Dec_08_1539.csv'

sub='fMRI_test'

dm1,dmci1=wmcsv_convolve(rdir+fname1)
dm2,dmci2=wmcsv_convolve(rdir+fname2)
dm3,dmci3=wmcsv_convolve(rdir+fname3)
# single file read
# datfile=rdir+fname
# df = pd.read_csv(datfile)

dmci=np.concatenate((dmci1,dmci2,dmci3))
dm=np.concatenate((dm1,dm2,dm3))
reg = LinearRegression()
ncondslist=np.arange(0,ncont)
mrc=np.ones([ncont])
for i in range(ncont):
    reg.fit(dmci[:,ncondslist[ncondslist != i]], dmci[:,i])
    bla = reg.predict(dmci[:,ncondslist[ncondslist != i]])
    mrc[i] = r2_score(dmci[:,i],bla)

print(mrc)

realtime=np.arange(0,len(dmci),0.1)
scantime=np.arange(0,len(dmci),TR)
fig, axes = plt.subplots(8, 1, figsize=(8, 15))
for i in range(ncont):
    axes[i].plot(realtime, dm[:,i],"r")
    axes[i].plot(scantime, dmci[:,i],"b--")
plt.show()

plt.plot(dmci)
plt.show()









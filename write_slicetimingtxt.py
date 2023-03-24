#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:05:48 2023

@author: WauWter
"""

import json

rdir='/home/control/wousch/Pilot/'
sub='Pilot002'
tasks=['wm1','wm2','wm3','loc']
for i in range(len(tasks)):
    jf=open(rdir+sub+'/func/'+tasks[i]+'/'+tasks[i]+'.json','r')
    values = json.load(jf)
    st=values['SliceTiming']
    with open(rdir+sub+'/func/'+tasks[i]+'/'+tasks[i]+'-slicetiming.1D', 'w') as stf:
        for item in st:
            stf.write("%s " % item)
    jf.close()

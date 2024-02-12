#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:05:48 2023

@author: WauWter
"""

import json

rdir='/home/control/wousch/project/bids/'
sub='sub-001'
tasks=['task-wmg_run-1','task-wmg_run-2','task-wmg_run-3','task-loc_run-1']
for i in range(len(tasks)):
    jf=open(rdir+sub+'/func/'+sub+'_'+tasks[i]+'_bold.json','r')
    values = json.load(jf)
    st=values['SliceTiming']
    with open(rdir+sub+'/derivatives/pipe/func/'+tasks[i]+'_slicetiming.1D', 'w') as stf:
        for item in st:
            stf.write("%s " % item)
    jf.close()

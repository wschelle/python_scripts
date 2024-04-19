#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:49:07 2024

@author: wousch
"""

import pydicom
import pandas as pd
from os import listdir
from os.path import isfile, join


def bids_subtsv_dcm(dicomfolder,bidsfolder,subid):
    
    tsv=pd.read_csv(bidsfolder+'participants.tsv',sep='\t')
    dcmfiles = [f for f in listdir(dicomfolder) if isfile(join(dicomfolder, f))]
    dcm = pydicom.read_file(dicomfolder+dcmfiles[0])
    newrow = [subid,dcm['PatientAge'][1:3],dcm['PatientSex'][0],dcm['PatientSize']._value,dcm['PatientWeight']._value,'n/a']
    cursub=len(tsv)
    tsv.loc[cursub] = newrow
    tsv.to_csv(bidsfolder+'participants.tsv', index=False,sep='\t')
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:29:46 2023

@author: WauWter
"""

import numpy as np
import struct
import gzip
import copy
    
def readnii(nifti_file,header_only=False,scaling=True):
    if nifti_file[-2:]=='gz':
        with gzip.open(nifti_file, 'rb') as file:
            data = file.read()
    else:
        with open(nifti_file, 'rb') as file:
            data = file.read()
    print('Loading: '+nifti_file+'...')
    header={
        "sizeof_hdr":struct.unpack("i",data[0:4])[0],
        "data_type":struct.unpack("b"*10,data[4:14]),
        "db_name":struct.unpack("b"*18,data[14:32]),
        "extents":struct.unpack("i",data[32:36])[0],
        "session_error":struct.unpack("h",data[36:38])[0],
        "regular":struct.unpack("b",data[38:39])[0],
        "dim_info":struct.unpack("b",data[39:40])[0],
        "dim":struct.unpack("h"*8,data[40:56]),
        "intent_p1":struct.unpack("f",data[56:60])[0],
        "intent_p2":struct.unpack("f",data[60:64])[0],
        "intent_p3":struct.unpack("f",data[64:68])[0],
        "intent_code":struct.unpack("h",data[68:70])[0],
        "datatype":struct.unpack("h",data[70:72])[0],
        "bitpix":struct.unpack("h",data[72:74])[0],
        "slice_start":struct.unpack("h",data[74:76])[0],
        "pixdim":struct.unpack("f"*8,data[76:108]),
        "vox_offset":struct.unpack("f",data[108:112])[0],
        "scl_slope":struct.unpack("f",data[112:116])[0],
        "scl_inter":struct.unpack("f",data[116:120])[0],
        "slice_end":struct.unpack("h",data[120:122])[0],
        "slice_code":struct.unpack("b",data[122:123])[0],
        "xyzt_units":struct.unpack("b",data[123:124])[0],
        "cal_max":struct.unpack("f",data[124:128])[0],
        "cal_min":struct.unpack("f",data[128:132])[0],
        "slice_duration":struct.unpack("f",data[132:136])[0],
        "toffset":struct.unpack("f",data[136:140])[0],
        "glmax":struct.unpack("i",data[140:144])[0],
        "glmin":struct.unpack("i",data[144:148])[0],
        "descrip":struct.unpack("b"*80,data[148:228]),
        "aux_file":struct.unpack("b"*24,data[228:252]),
        "qform_code":struct.unpack("h",data[252:254])[0],
        "sform_code":struct.unpack("h",data[254:256])[0],
        "quatern_b":struct.unpack("f",data[256:260])[0],
        "quatern_c":struct.unpack("f",data[260:264])[0],
        "quatern_d":struct.unpack("f",data[264:268])[0],
        "qoffset_x":struct.unpack("f",data[268:272])[0],
        "qoffset_y":struct.unpack("f",data[272:276])[0],
        "qoffset_z":struct.unpack("f",data[276:280])[0],
        "srow_x":struct.unpack("f"*4,data[280:296]),
        "srow_y":struct.unpack("f"*4,data[296:312]),
        "srow_z":struct.unpack("f"*4,data[312:328]),
        "intent_name":struct.unpack("b"*16,data[328:344]),
        "magic":struct.unpack("b"*4,data[344:348])
        }
    if header_only:
        return(header)
    else:
        data_start=int(header['vox_offset'])
        totalsize=1
        for i in header['dim'][1:]:
            totalsize*=i
        data_dims=header['dim'][1:]
        if header['datatype']==0:
            print('Yeah... We from data support are just as clueless as you. Going home.')
            return(header)
        elif header['datatype']==1:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=bool)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==2:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.ubyte)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==4:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.int16)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==8:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.int32)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==16:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.float32)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==32:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.csingle)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==64:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.double)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==256:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.byte)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==512:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.ushort)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==768:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.uintc)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==1024:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.longlong)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==1280:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.ulonglong)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        elif header['datatype']==1536:
            nii=np.frombuffer(data, offset=data_start,count=totalsize, dtype=np.longdouble)
            nii=np.squeeze(np.reshape(nii,data_dims[::-1]))
            nii=nii.T
        
        nifti=copy.copy(nii)
        if (scaling==True) & (header['scl_slope']!=0):
            nifti = (nifti * header['scl_slope'])
            nifti = (nifti + header['scl_inter'])
            
        return(nifti, header)

def savenii(data,header,filename):
    print('Writing: '+filename+'...')
    header['vox_offset']=352
    data=data.T
    if filename[-2:]=='gz':
        with gzip.open(filename, 'wb') as f:
            f.write(struct.pack('i', header['sizeof_hdr']))
            f.write(struct.pack('10b', *header['data_type']))
            f.write(struct.pack('18b', *header['db_name']))
            f.write(struct.pack('i', header['extents']))
            f.write(struct.pack('h', header['session_error']))
            f.write(struct.pack('b', header['regular']))
            f.write(struct.pack('b', header['dim_info']))
            f.write(struct.pack('8h', *header['dim']))
            f.write(struct.pack('f', header['intent_p1']))
            f.write(struct.pack('f', header['intent_p2']))
            f.write(struct.pack('f', header['intent_p3']))
            f.write(struct.pack('h', header['intent_code']))
            f.write(struct.pack('h', header['datatype']))
            f.write(struct.pack('h', header['bitpix']))
            f.write(struct.pack('h', header['slice_start']))
            f.write(struct.pack('8f', *header['pixdim']))
            f.write(struct.pack('f', header['vox_offset']))
            f.write(struct.pack('f', header['scl_slope']))
            f.write(struct.pack('f', header['scl_inter']))
            f.write(struct.pack('h', header['slice_end']))
            f.write(struct.pack('b', header['slice_code']))
            f.write(struct.pack('b', header['xyzt_units']))
            f.write(struct.pack('f', header['cal_max']))
            f.write(struct.pack('f', header['cal_min']))
            f.write(struct.pack('f', header['slice_duration']))
            f.write(struct.pack('f', header['toffset']))
            f.write(struct.pack('i', header['glmax']))
            f.write(struct.pack('i', header['glmin']))
            f.write(struct.pack('80b', *header['descrip']))
            f.write(struct.pack('24b', *header['aux_file']))
            f.write(struct.pack('h', header['qform_code']))
            f.write(struct.pack('h', header['sform_code']))
            f.write(struct.pack('f', header['quatern_b']))
            f.write(struct.pack('f', header['quatern_c']))
            f.write(struct.pack('f', header['quatern_d']))
            f.write(struct.pack('f', header['qoffset_x']))
            f.write(struct.pack('f', header['qoffset_y']))
            f.write(struct.pack('f', header['qoffset_z']))
            f.write(struct.pack('4f', *header['srow_x']))
            f.write(struct.pack('4f', *header['srow_y']))
            f.write(struct.pack('4f', *header['srow_z']))
            f.write(struct.pack('16b', *header['intent_name']))
            f.write(struct.pack('4b', *header['magic']))
            f.write(bytearray(4))
            if data.dtype=='int16':
                f.write(struct.pack('h'*data.size,*np.reshape(data,data.size)))
            elif data.dtype=='int32':
                f.write(struct.pack('i'*data.size,*np.reshape(data,data.size)))
            elif data.dtype=='float32':
                f.write(struct.pack('f'*data.size,*np.reshape(data,data.size)))
            elif data.dtype=='float64':
                f.write(struct.pack('d'*data.size,*np.reshape(data,data.size)))
    else:
        with open(filename, 'wb') as f:
            f.write(struct.pack('i', header['sizeof_hdr']))
            f.write(struct.pack('10b', *header['data_type']))
            f.write(struct.pack('18b', *header['db_name']))
            f.write(struct.pack('i', header['extents']))
            f.write(struct.pack('h', header['session_error']))
            f.write(struct.pack('b', header['regular']))
            f.write(struct.pack('b', header['dim_info']))
            f.write(struct.pack('8h', *header['dim']))
            f.write(struct.pack('f', header['intent_p1']))
            f.write(struct.pack('f', header['intent_p2']))
            f.write(struct.pack('f', header['intent_p3']))
            f.write(struct.pack('h', header['intent_code']))
            f.write(struct.pack('h', header['datatype']))
            f.write(struct.pack('h', header['bitpix']))
            f.write(struct.pack('h', header['slice_start']))
            f.write(struct.pack('8f', *header['pixdim']))
            f.write(struct.pack('f', header['vox_offset']))
            f.write(struct.pack('f', header['scl_slope']))
            f.write(struct.pack('f', header['scl_inter']))
            f.write(struct.pack('h', header['slice_end']))
            f.write(struct.pack('b', header['slice_code']))
            f.write(struct.pack('b', header['xyzt_units']))
            f.write(struct.pack('f', header['cal_max']))
            f.write(struct.pack('f', header['cal_min']))
            f.write(struct.pack('f', header['slice_duration']))
            f.write(struct.pack('f', header['toffset']))
            f.write(struct.pack('i', header['glmax']))
            f.write(struct.pack('i', header['glmin']))
            f.write(struct.pack('80b', *header['descrip']))
            f.write(struct.pack('24b', *header['aux_file']))
            f.write(struct.pack('h', header['qform_code']))
            f.write(struct.pack('h', header['sform_code']))
            f.write(struct.pack('f', header['quatern_b']))
            f.write(struct.pack('f', header['quatern_c']))
            f.write(struct.pack('f', header['quatern_d']))
            f.write(struct.pack('f', header['qoffset_x']))
            f.write(struct.pack('f', header['qoffset_y']))
            f.write(struct.pack('f', header['qoffset_z']))
            f.write(struct.pack('4f', *header['srow_x']))
            f.write(struct.pack('4f', *header['srow_y']))
            f.write(struct.pack('4f', *header['srow_z']))
            f.write(struct.pack('16b', *header['intent_name']))
            f.write(struct.pack('4b', *header['magic']))
            f.write(bytearray(4))
            if data.dtype=='int16':
                f.write(struct.pack('h'*data.size,*np.reshape(data,data.size)))
            if data.dtype=='int32':
                f.write(struct.pack('i'*data.size,*np.reshape(data,data.size)))
            if data.dtype=='float32':
                f.write(struct.pack('f'*data.size,*np.reshape(data,data.size)))
            if data.dtype=='float64':
                f.write(struct.pack('d'*data.size,*np.reshape(data,data.size)))
    
        
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:34:37 2023

@author: wousch
"""
import numpy as np
import matplotlib.pyplot as plt
from Python.python_scripts.wauwterfmri import gloverhrf,gammahrf,doublegammahrf,inverselog_hrf,lmfit_ilhrf
import lmfit
from copy import deepcopy
from tqdm import tqdm
from scipy.special import erf

def param_convolve(hrf,par0):
    onset=par0[0]*par0[4]
    durat=par0[1]*par0[4]
    xax=np.arange(len(hrf)*3)
    act=(erf(xax-(onset+len(hrf)))+1)/2
    deact=(1-erf(xax-(onset+durat+len(hrf))))/2
    desmat=act*deact
    condes=np.convolve(desmat,hrf)
    condes=condes[len(hrf):2*len(hrf)]
    condes/=np.max(condes)
    condes*=par0[3]
    condes+=par0[2]
    return condes
    
def deconvolve(params,hrf,ydata):
    par0=np.zeros(5,dtype=np.float32)
    par0[0]=params['onset'].value
    par0[1]=params['duration'].value
    par0[2]=params['const'].value
    par0[3]=params['amplitude'].value
    par0[4]=params['Hz'].value
    ymodel=param_convolve(hrf,par0)
    return (ymodel-ydata)

def lmfit_doublegamma(params,hrflen,ydata,timestep):
    p0=np.zeros(7,dtype=np.float32)
    p0[0]=params['tpeak'].value
    p0[1]=params['tunder'].value
    p0[2]=params['dpeak'].value
    p0[3]=params['dunder'].value
    p0[4]=params['pu_ratio'].value
    p0[5]=params['const'].value
    p0[6]=params['amplitude'].value
    ymodel=doublegammahrf(p0,hrflen,timestep)
    return (ymodel-ydata)    

# p0=np.array([0.,3,0,1.,10],dtype=np.float32)
# params = lmfit.Parameters()
# params.add('onset',p0[0],min=-len(hrf)/10,max=len(hrf)/10)
# params.add('duration',p0[1],min=0.5,max=len(hrf)/10)
# params.add('const',p0[2])
# params.add('amplitude',p0[3])
# params.add('Hz',p0[4],vary=False)


hrf=gammahrf(19,0.1)

p0=np.array([5,10,1.5,0.8,3,0,1])
hrf=doublegammahrf(p0,19,0.1)
plt.plot(np.arange(190)/10,hrf)
plt.show()




# p0=np.array([1.7, 2.7, 1.2, -1, 8.5, 1., 0])
# params = lmfit.Parameters()
# params.add('A1', p0[0],min=-15,max=15)
# params.add('T1', p0[1],min=-1,max=15)
# params.add('D1', p0[2],min=0.1,max=15)
# params.add('A2', p0[3],min=-15,max=15)
# params.add('T2', p0[4],min=4,max=30)
# params.add('D2', p0[5],min=0.1,max=15)
# params.add('C', p0[6],min=-10,max=10)

p0=np.array([2.18, 3.26, 0.98, -2.35, 6.23, 2.27, 0.17, 18.26, 2.57, 0])
params = lmfit.Parameters()
params.add('A1', p0[0],min=-15,max=15)
params.add('T1', p0[1],min=-2,max=10)
params.add('D1', p0[2],min=0.1,max=2)
params.add('A2', p0[3],min=-15,max=15)
params.add('T2', p0[4],min=1.5,max=15)
params.add('D2', p0[5],min=0.1,max=5)
params.add('A3', p0[6],min=-15,max=15)
params.add('T3', p0[7],min=8,max=30)
params.add('D3', p0[8],min=0.1,max=10)
params.add('C', p0[9],min=-5,max=5)

avgsfir=np.nanmean(sfirio[:,-1,:,3,:],axis=1)
# avgsfir=np.nanmean(avgsfir,axis=1)
# avgsfir=np.nanmean(avgsfir,axis=1)
avgmfir=np.zeros([nsub,len(p0)],dtype=np.float32)
avgmfirfit=np.zeros([nsub,nfir],dtype=np.float32)

t=np.arange(0,19,0.1)
for i in tqdm(range(nsub)):
    zfit = lmfit.minimize(lmfit_ilhrf, params, args=(t, avgsfir[i,:]))
    avgmfir[i,0]=zfit.params['A1'].value
    avgmfir[i,1]=zfit.params['T1'].value
    avgmfir[i,2]=zfit.params['D1'].value
    avgmfir[i,3]=zfit.params['A2'].value
    avgmfir[i,4]=zfit.params['T2'].value
    avgmfir[i,5]=zfit.params['D2'].value
    avgmfir[i,6]=zfit.params['A3'].value
    avgmfir[i,7]=zfit.params['T3'].value
    avgmfir[i,8]=zfit.params['D3'].value
    avgmfir[i,9]=zfit.params['C'].value
    avgmfirfit[i,:]=inverselog_hrf(t,avgmfir[i,:])
    avgmfirfit[i,:]/=np.max(avgmfirfit[i,:])



avgsfir=np.nanmean(sfir,axis=1)
avgsfir=np.nanmean(avgsfir,axis=1)
avgsfir=np.nanmean(avgsfir,axis=1)

# ,epsfcn=0.1
decon=np.zeros([nsub,ncat2,nroi,ncon,len(p0)+1])
condur=np.array([3,3,6],dtype=np.float32)

for i in tqdm(range(nsub)):
    for j in range(ncat2):
        for k in range(nroi):
            for l in range(ncon):
                if np.sum(np.isnan(sfir[i,j,k,l,:]))==0:
                    tmp1=np.mean(sfir[i,j,k,l,0:10])
                    tmp2=np.mean(sfir[i,j,k,l,40:70])
                    p0=np.array([0.,condur[l],tmp1,tmp2-tmp1,10],dtype=np.float32)
                    params = lmfit.Parameters()
                    params.add('onset',p0[0],min=-8.,max=len(hrf)/20)
                    params.add('duration',p0[1],min=1.,max=len(hrf)/15)
                    params.add('const',p0[2])
                    params.add('amplitude',p0[3])
                    params.add('Hz',p0[4],vary=False)
                    # zfit = lmfit.minimize(deconvolve, params, args=(hrf, sfir[i,j,k,l,:]))
                    zfit = lmfit.minimize(deconvolve, params, args=(avgmfirfit[i,:], sfir[i,j,k,l,:]))
                    decon[i,j,k,l,0]=zfit.params['onset'].value
                    decon[i,j,k,l,1]=zfit.params['duration'].value
                    decon[i,j,k,l,2]=zfit.params['const'].value
                    decon[i,j,k,l,3]=zfit.params['amplitude'].value
                    decon[i,j,k,l,4]=zfit.params['Hz'].value
                    decon[i,j,k,l,5]=zfit.chisqr
                    #print(decon[i,j,k,l,:])


# csub=10
# ccat=3
# croi=8
# ccon=0
# yfit=param_convolve(hrf,tmp)
# plt.plot(sfir[csub,ccat,croi,ccon,:])
# plt.plot(hrf)
# plt.plot(yfit)
# plt.show()


mfirfit=np.nanmean(decon,axis=0)
efirfit=np.nanstd(decon,axis=0)/np.sqrt(nsub)

ratio_in_ign=decon[:,:,:,0,:]-decon[:,:,:,1,:]
ratio_in_out=decon[:,:,:,0,:]-decon[:,:,:,2,:]

eratio_in_ign=np.nanstd(ratio_in_ign,axis=0)/np.sqrt(nsub)
eratio_in_out=np.nanstd(ratio_in_out,axis=0)/np.sqrt(nsub)
ratio_in_ign=np.nanmean(ratio_in_ign,axis=0)
ratio_in_out=np.nanmean(ratio_in_out,axis=0)

curpar=0#onset latency
curcat=0#faces

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Onset latency Faces')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_onsetlatency_face_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio onset latency Faces')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_onsetlatency_face_vs_ignore_17ROIs.png',dpi=300)



curcat=1#scenes

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Onset latency Scenes')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_onsetlatency_scene_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio onset latency Scenes')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_onsetlatency_scene_vs_ignore_17ROIs.png',dpi=300)





curcat=2#tools

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Onset latency Tools')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_onsetlatency_tool_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio onset latency Tools')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_onsetlatency_tool_vs_ignore_17ROIs.png',dpi=300)




curcat=3#allin

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Onset latency All')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_onsetlatency_all_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio onset latency All')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_onsetlatency_all_vs_ignore_17ROIs.png',dpi=300)






curpar=1#temporal dispersion
curcat=0#faces

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Temporal Dispersion Faces')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_tempdispersion_face_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio Temporal Dispersion Faces')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_tempdispersion_face_vs_ignore_17ROIs.png',dpi=300)



curcat=1#scenes

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Temporal Dispersion Scenes')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_tempdispersion_scene_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio Temporal Dispersion Scenes')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_tempdispersion_scene_vs_ignore_17ROIs.png',dpi=300)





curcat=2#tools

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Temporal Dispersion Tools')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_tempdispersion_tool_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio Temporal Dispersion Tools')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_tempdispersion_tool_vs_ignore_17ROIs.png',dpi=300)




curcat=3#allin

tmp2a=mfirfit[curcat,:,0,curpar]
tmp3a=mfirfit[curcat,:,1,curpar]
tmp4a=mfirfit[curcat,:,2,curpar]

tmp2e=efirfit[curcat,:,0,curpar]
tmp3e=efirfit[curcat,:,1,curpar]
tmp4e=efirfit[curcat,:,2,curpar]

tmp2b=mfirfit[curcat,:,0,curpar].argsort()
tmp3b=mfirfit[curcat,:,1,curpar].argsort()
tmp4b=mfirfit[curcat,:,2,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]
tmp4a=tmp4a[tmp4b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
tmp4e=tmp4e[tmp4b]

tmpin = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpign = np.array(list(roilab.values()))[tmp3b.astype(int)]
tmpout = np.array(list(roilab.values()))[tmp4b.astype(int)]

colors = plt.cm.viridis(np.linspace(0.1,0.9,nroi))
fig, axes = plt.subplots(3,1, figsize=(15,10))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[2].bar(range(nroi),tmp4a,color=colors)
axes[2].errorbar(range(nroi),tmp4a, yerr=tmp4e, color="black",fmt='none')
axes[0].set_title('Temporal Dispersion All')
axes[0].set_ylabel('input open')
axes[1].set_ylabel('input closed')
axes[2].set_ylabel('output open')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpin, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpign, rotation = 45, va="center", position=(0,-0.05))
axes[2].set_xticks(range(nroi))
axes[2].set_xticklabels(tmpout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'/figures/FIR_tempdispersion_all_vs_ignore_17ROIs.png',dpi=300)


tmp2a=ratio_in_ign[curcat,:,curpar]
tmp3a=ratio_in_out[curcat,:,curpar]
tmp2a[np.isinf(tmp2a)]=0
tmp3a[np.isinf(tmp3a)]=0

tmp2e=eratio_in_ign[curcat,:,curpar]
tmp3e=eratio_in_out[curcat,:,curpar]
tmp2e[np.isinf(tmp2e)]=0
tmp3e[np.isinf(tmp3e)]=0

tmp2b=ratio_in_ign[curcat,:,curpar].argsort()
tmp3b=ratio_in_out[curcat,:,curpar].argsort()

tmp2a=tmp2a[tmp2b]
tmp3a=tmp3a[tmp3b]

tmp2e=tmp2e[tmp2b]
tmp3e=tmp3e[tmp3b]
  
tmpinign = np.array(list(roilab.values()))[tmp2b.astype(int)]
tmpinout = np.array(list(roilab.values()))[tmp3b.astype(int)]

fig, axes = plt.subplots(2,1, figsize=(12,7))
axes[0].bar(range(nroi),tmp2a,color=colors)
axes[0].errorbar(range(nroi),tmp2a, yerr=tmp2e, color="black",fmt='none')
axes[1].bar(range(nroi),tmp3a,color=colors)
axes[1].errorbar(range(nroi),tmp3a, yerr=tmp3e, color="black",fmt='none')
axes[0].set_title('Ratio Temporal Dispersion All')
axes[0].set_ylabel('Gate open vs closed')
axes[1].set_ylabel('Gate in vs out')
axes[0].set_xticks(range(nroi))
axes[0].set_xticklabels(tmpinign, rotation = 45, va="center", position=(0,-0.05))
axes[1].set_xticks(range(nroi))
axes[1].set_xticklabels(tmpinout, rotation = 45, va="center", position=(0,-0.05))
fig.tight_layout(pad=1)
plt.show()
# fig.savefig(gdir+'figures/FIR_delta_tempdispersion_all_vs_ignore_17ROIs.png',dpi=300)



    
import argparse
from typing import final
import timeit 
import numpy as np
from numpy import linalg
from numpy.core.shape_base import block
#from numpy.linalg.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import sys
import math
from scipy.io import wavfile
from scipy import signal
from scipy.signal.filter_design import normalize
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from scipy.io import wavfile
from scipy import signal
from scipy.signal.filter_design import normalize
import Dictionary
import PanelFiles
import AtomClass

def DANNO(residue,dicSize,chosenNet,step,L,approxRatio,net1,net2,net3):
    """
    Function 
    """
    chosenDic = 0
    res=np.zeros((1,int(dicSize/8)))

    res[0] = [residue[8*x] for x in np.arange(math.ceil(dicSize/8))]
    res[0]=np.array(res[0]/np.linalg.norm(res[0]))
    #print(res)
    print(f"Diff ApproxRatio  {approxRatio[(step%L)-1] - approxRatio[(step%L)-2]}")

    if (step==0 or approxRatio[(step%L)-1] - approxRatio[(step%L)-2] <= -1e-1):
        chosenNet = 1
    elif ( approxRatio[(step%L)-1]-approxRatio[(step%L)-2] <= -5e-2 and approxRatio[(step%L)-1]-approxRatio[(step%L)-2] > -1e-1):
        chosenNet = 2
    elif ( approxRatio[(step%L)-1]-approxRatio[(step%L)-2] <= 0 and approxRatio[(step%L)-1]-approxRatio[(step%L)-2] > -5e-2 ):
        chosenNet = 3
    else:print("Invalid Net")

    if (chosenNet == 1 or step == 0):          
        # Load Neural Network 1
        #net1 = load_model('Net1.h5')
        predDic = net1.predict(res).round()
    elif (chosenNet == 2):
        #Load Neural Network 2
        #net2 = load_model('Net2.h5')
        predDic = net2.predict(res).round()

    elif (chosenNet == 3):
        #Load Neural Network 3
        #net3 = load_model('Net3.h5')
        predDic = net3.predict(res).round()

    else:print("Invalid Net2")
    

    if predDic.argmax()==0: chosenDic=6
    elif predDic.argmax()==1: chosenDic=5
    elif predDic.argmax()==2: chosenDic=4
    else: print("Invalid Net1")
    print(f"NetDic:    {chosenDic}\n")
    return chosenDic ,chosenNet
    #Comentário do Nicholas: Adicionar os 2 dicionário a DANNO depois

def matchingPursuit(residue, chosenParm ,dicSize,dicData,decompData,step,chosenDic,dicAtoms,a0,b0,flagOMP,Fs):
   
   #FALTA ALTERAR (NICHOLAS)
 #   blockRange=PanelFiles.FileDecompBlockRange()
  #  blockRange.loadData('panelBlockRange.dat')

   # decompData=PanelFiles.FileDecomp()
    #decompData.loadData('panelDecomp.dat')

    #dicData=PanelFiles.FileDictionary()
    #dicData.loadData('panelDictionary.dat')
#Apagar linhas acima depois
################################################
    chosenTau=0
    chosenXi=0.0
    chosenOptPhase=0.0
    maxInnerProd=0.0
    fileName = "MPTradicional.out"

    realAtom=np.zeros(dicSize)
    realAtomDec=np.zeros(dicSize)
    realAtomInc=np.zeros(dicSize)
    realAtomWin=np.zeros(2*dicSize)

    complexAtom=np.zeros(dicSize,dtype='complex')
    complexAtomDec=np.zeros(dicSize,dtype='complex')
    complexAtomInc=np.zeros(dicSize,dtype='complex')
    complexAtomXi=np.zeros(dicSize,dtype='complex')
    complexAtom2Xi=np.zeros(dicSize,dtype='complex')

    conv_zxi0_w2 = np.zeros(2*dicSize)
    conv_zxi0_w2_expinc = np.zeros(2*dicSize)

    gabDic = Dictionary.GabDic()
    gabDic.setSignalSize(dicSize)
    triangDic = Dictionary.TriangDic()
    triangDic.setSignalSize(dicSize)
    bateDic = Dictionary.BateDic()
    bateDic.setSignalSize(dicSize)
    sigmoDic = Dictionary.SigmoDic()
    sigmoDic.setSignalSize(dicSize)
    chiDic = Dictionary.ChiDic()
    chiDic.setSignalSize(dicSize)

    parm=AtomClass.Atom()

    #chosenParm.__init__()
    #chosenParm.innerProd=0.0
    #chosenParm.rho=0.0
    #chosenParm.xi=0.0
    #chosenParm.phase=0.0
    #chosenParm.a=0.0
    #chosenParm.b=0.0
    #chosenParm.eta=0.0
    chosenParm.setAtom(innerProd=0.0,rho=0.0,xi=0.0,phase=0.0,a=0,b=0,eta=0.0)


    MPType=decompData.getMPType()
    #print(MPType)

    for j in np.arange(dicData.getNumDicBlock()):
        
        s=dicData.getDicData()[j][0]
        deltaTau=dicData.getDicData()[j][1]
        fdiscrType=dicData.getDicData()[j][2]
        freqi=dicData.getDicData()[j][3]
        freqf=dicData.getDicData()[j][4]
        dicType=dicData.getDicData()[j][5]
        decay=dicData.getDicData()[j][6]
        rise=dicData.getDicData()[j][7]
        
        N=dicSize
        #print("PA")
        if (chosenDic==dicType or chosenDic==0):
           # print("PO")
           # print(f"MPDicCC:    {chosenDic}\n")
            #print(f"MPDic:    {dicType}\n")
            if((s> dicSize)and(s!=88888)):
                print("Scale greater than blockSize!")
            
            #if (decompData.getSigType()==1):
            #Fs=getSamplingRate

            if ((freqf==9999999999) and (freqi==9999999999)):
                freqi=27.5
                freqf=Fs
            elif ((freqf==9999999999) and (freqi==0000000000)):
                freqf=Fs
                freqi= freqf/2
            elif (freqf==9999999999):
                freqf=Fs
            elif (freqi==9999999999):
                freqi=freqf/s
            elif (freqi==0000000000):
                freqi = freqf/2
            
            if (freqf>Fs): print("Final frquency Greater than Fs")
            
            if (fdiscrType==1):#Linear
                Nfreq=int(freqf/(2*freqi))
                xi_vec=np.array([(2*math.pi/Fs)*(freqi*i) for i in range(Nfreq)])

            elif (fdiscrType==2): #geometric with quarter-tone discretization
                Nfreq = int(math.ceil(math.log10(freqf/freqi)/math.log10(2.0))+1)
                xi_vec=np.array([(2*math.pi/Fs)*(freqi*math.pow(2.0,(i-1)/24)) for i in range(1,Nfreq)])
                xi_vec=np.insert(xi_vec,0,0.0)
            
            if (dicType == 3): #Dicionáriuo de Impulsos
                for k in np.arange(N):

                    if (np.abs(residue[k]) > np.abs(maxInnerProd)):
                        maxInnerProd = residue[k]
                        parm.setAtom(u=k,dicType=3)
                        setParameters(chosenParm,dicType,maxInnerProd,0,0,chosenOptPhase,k,k,k,0)

            if (dicType == 4): #Dicionário de Gabor

                if (MPType == 3):

                    parm.setAtom(rho=1.0/s,s=s,b=N-1,dicType=4)
                    gabDic.setRealAtom(parm,N)
                    for k in np.arange(N):
                        realAtomWin[k]=gabDic.getRealAtom()[k]
                        realAtomWin[(2*N-1-k)]=gabDic.getRealAtom()[k]
                    #if j==0:
                    # plt.plot(gabDic.getRealAtom())
                    # plt.show()
                    
                    # Complex Atom

                    for k in np.arange(Nfreq):

                        xi=xi_vec[k]
                       

                        if ((xi==0)or(xi>=((2*math.pi)/s))):
                            # Xi
                            parm.setAtom(rho=0.0,s=s,xi=xi,b=N-1,dicType=4)
                            gabDic.setComplexAtom(parm,N)
                            complexAtomXi=gabDic.getComplexAtom()
                            #if j==0:
                                #plt.plot(residue)
                                #plt.show()
                            #2Xi
                            parm.setAtom(rho=0.0,s=s,xi=2*xi,b=N-1,dicType=4)
                            gabDic.setComplexAtom(parm,N)
                            complexAtom2Xi=gabDic.getComplexAtom()

                    maxInnerProd,chosenTau,chosenOptPhase = fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,dicSize,1,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,(1.0/decay))
                        
                    if (math.fabs(maxInnerProd)>math.fabs(chosenParm.innerProd)):
                        #print(maxInnerProd)
                        setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),xi,chosenOptPhase,chosenTau,chosenTau,N-1,rise)


            if (dicType == 5): #Dicionário Triangular

                if (MPType == 3):
                    
                    for a in np.arange(1,math.ceil((s/5.0))):
                        if (a >= s/8.0):
                    
                            parm.setAtom(rho=1.0/s,s=s,a=a,u=a, b=s-a-1,dicType=5)
                            triangDic.setRealAtom(parm,2*N)
                            for k in np.arange(2*N):
                                realAtomWin[k]=triangDic.getRealAtom()[k]
                                #realAtomWin[(2*N-1-k)]=triangDic.getRealAtom()[k]
                            #if j==0:
                            #plt.plot(realAtomWin)
                            #plt.show()
                            
                            # Complex Atom

                            for k in np.arange(Nfreq):

                                xi=xi_vec[k]
                            

                                if ((xi==0)or(xi>=((2*math.pi)/s))):
                                    # Xi
                                    parm.setAtom(rho=0.0,s=0.0,xi=xi,b=N-1,dicType=5)
                                    triangDic.setComplexAtomExp(parm,N)
                                    complexAtomXi=triangDic.getComplexAtomExp()
                                    #if j==0:
                                        #plt.plot(residue)
                                        #plt.show()
                                    #2Xi
                                    parm.setAtom(rho=0.0,s=0.0,xi=2*xi,b=N-1,dicType=5)
                                    triangDic.setComplexAtomExp(parm,N)
                                    complexAtom2Xi=triangDic.getComplexAtomExp()

                            maxInnerProd,chosenTau,chosenOptPhase = fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,dicSize,1,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,(1.0/decay))
                                
                            if (math.fabs(maxInnerProd)>math.fabs(chosenParm.innerProd)):
                                #print(maxInnerProd)
                                setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),xi,chosenOptPhase,chosenTau,chosenTau,N-1,rise)


            if (dicType == 6): #Dicionário de Bateman
                if (MPType == 1):
                    print("Entrou MP1")
                    
                    for u in np.arange(N):
                        parm.setAtom(rho=decay,eta=rise,u=u, b=N-1,dicType=6)
                        
                        bateDic.setComplexAtom(parm,N)
                        complexAtom=bateDic.getComplexAtom()

                        maxInnerProd,chosenOptPhase=computeOptimumPhase(residue,chosenOptPhase,maxInnerProd,N,0,complexAtom,fileName,s,u,N)
                        # print("maxInnerProd:{mip}".format(mip=maxInnerProd))
                        
                        if(np.abs(maxInnerProd)>np.abs(parm.innerProd)):
                            
                            setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),0,chosenOptPhase,u,u,N-1,rise)
                            # print("parmInner:{ip}".format(ip=str(chosenParm)))

                if (MPType == 3):
                    #print("MP Function")


                    #beta=rise
                    #rho=decay
                    
                    # Real Atom
                    parm.setAtom(rho=decay,eta=rise,b=N-1,dicType=6)
                    
                    #bateDic.setRealAtom(parm,N)
                    #realAtom=bateDic.getRealAtom()
                    bateDic.setRealAtom(parm,2*N)
                    realAtomWin=bateDic.getRealAtom()
                    
                    # Complex Atom
                   # print(np.dot(realAtom,residue))

                    for k in np.arange(Nfreq):

                        xi=xi_vec[k]
                       

                        if ((xi==0)or(xi>=((2*math.pi)/s))):
                            # Xi
                            parm.setAtom(rho=decay,eta=rise,xi=xi,b=N-1,dicType=6)
                            bateDic.setComplexAtom(parm,N)
                            complexAtomXi=bateDic.getComplexAtom()
                           
                            #2Xi
                            parm.setAtom(rho=decay,eta=rise,xi=2*xi,b=N-1,dicType=6)
                            bateDic.setComplexAtom(parm,N)
                            complexAtom2Xi=bateDic.getComplexAtom()
                            

                    maxInnerProd,chosenTau,chosenOptPhase = fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,dicSize,1,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,(1.0/decay))
                        
                    if (math.fabs(maxInnerProd)>math.fabs(chosenParm.innerProd)):
                        #print(maxInnerProd)
                        setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),xi,chosenOptPhase,chosenTau,chosenTau,N-1,rise)
                        
                        
            if (dicType == 7): #Dicionário de Sigmóides Exponenciais
                            
                            if (MPType == 3):
                                #print("MP Function")
                        
                                # Real Atom
                                parm.setAtom(rho=decay,eta=rise,b=N-1,s=s,dicType=7)
                                sigmoDic.setRealAtom(parm,2*N)
                                realAtomWin=sigmoDic.getRealAtom()
                                
                                #if j==0:
                                #plt.plot(realAtomWin)
                                #plt.show()
                                
                                # Complex Atom

                                for k in np.arange(Nfreq):

                                    xi=xi_vec[k]
                                

                                    if ((xi==0)or(xi>=((2*math.pi)/s))):
                                        # Xi
                                        parm.setAtom(rho=decay,eta=rise,xi=xi,b=N-1,s=s,dicType=7)
                                        sigmoDic.setComplexAtom(parm,N)
                                        complexAtomXi=sigmoDic.getComplexAtom()
                                        #if j==0:
                                            #plt.plot(residue)
                                            #plt.show()
                                        #2Xi
                                        parm.setAtom(rho=decay,eta=rise,xi=2*xi,b=N-1,s=s,dicType=7)
                                        sigmoDic.setComplexAtom(parm,N)
                                        complexAtom2Xi=sigmoDic.getComplexAtom()

                                maxInnerProd,chosenTau,chosenOptPhase = fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,dicSize,1,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,(1.0/decay))
                                    
                                if (math.fabs(maxInnerProd)>math.fabs(chosenParm.innerProd)):
                                    #print(maxInnerProd)
                                    setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),xi,chosenOptPhase,chosenTau,chosenTau,N-1,rise)


            if (dicType == 8): #Dicionário CHI2
                            
                            if (MPType == 3):
                                #print("MP Function")


                                #beta=rise
                                #rho=decay
                                
                                # Real Atom
                                parm.setAtom(rho=decay,eta=rise,s=s,b=N-1,dicType=8)
                                chiDic.setRealAtom(parm,2*N)
                                realAtomWin=chiDic.getRealAtom()
                                #if j==0:
                                #plt.plot(realAtomWin)
                                #plt.show()
                                
                                # Complex Atom

                                for k in np.arange(Nfreq):

                                    xi=xi_vec[k]
                                

                                    if ((xi==0)or(xi>=((2*math.pi)/s))):
                                        # Xi
                                        parm.setAtom(rho=decay,eta=rise,xi=xi,b=N-1,dicType=8)
                                        chiDic.setComplexAtom(parm,N)
                                        complexAtomXi=chiDic.getComplexAtom()
                                        #if j==0:
                                            #plt.plot(residue)
                                            #plt.show()
                                        #2Xi
                                        parm.setAtom(rho=decay,eta=rise,xi=2*xi,b=N-1,dicType=8)
                                        chiDic.setComplexAtom(parm,N)
                                        complexAtom2Xi=chiDic.getComplexAtom()

                                maxInnerProd,chosenTau,chosenOptPhase = fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,dicSize,1,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,s)
                                    
                                if (math.fabs(maxInnerProd)>math.fabs(chosenParm.innerProd)):
                                    #print(maxInnerProd)
                                    setParameters(chosenParm,dicType,maxInnerProd,(1.0/decay),xi,chosenOptPhase,chosenTau,0,N-1,rise) #Dúvida


def fastMPKolasaModified(residue,maxInnerProd,chosenOptPhase,chosenTau,N,decincAsymmFlag,deltaTau,xi,realAtomWin,complexAtomXi,complexAtom2Xi,conv_zxi0_w2,fileName,s):
    
    #f2=open('LogKolasaModified.dat','a')

    #f=open('MPLog.out','a')

    w1=np.zeros(2*N,dtype='complex') #w1=np.zeros(2*N)
    w2=np.zeros(2*N,dtype='complex')
    z1=np.zeros(2*N,dtype='complex')
    z2=np.zeros(2*N,dtype='complex')

    if (len(realAtomWin)==len(complexAtomXi)):
        for i in np.arange(len(w1)):
            w1[i]=complex(realAtomWin[N-1-i],0)
            w2[i]=complex(realAtomWin[N-1-i]*realAtomWin[N-1-i],0)
            z1[i]=complex(residue[i],complexAtomXi[i].imag*residue[i])
            z2[i]=complex(complexAtom2Xi[i].real,complexAtom2Xi[i].imag)
    else:
        for i in np.arange(len(realAtomWin)):
            w1[i]=complex(realAtomWin[2*N-1-i],0)
            w2[i]=complex(realAtomWin[2*N-1-i]*realAtomWin[2*N-1-i])
        for i in np.arange(len(complexAtomXi)):
            z1[i]=complex(residue[i],complexAtomXi[i].imag*residue[i])
            z2[i]=complex(complexAtom2Xi[i].real,complexAtom2Xi[i].imag)

    #plt.plot(w1.real)
    #plt.show()
    
    conv_zw1=np.fft.ifft( np.fft.fft(w1)*np.fft.fft(z1) )
    conv_zw2=np.fft.ifft( np.fft.fft(w2)*np.fft.fft(z2) ) 

    #plt.plot(conv_zw1.real)
    #plt.show()

    #conv_zw1=
    
    
    
   
    #conv_zw2=signal.convolve(w2,z2)
    conv_zxi0_w2=np.zeros(len(conv_zw2))
    if (xi==0):
        for i in np.arange(len(conv_zw2)):
            conv_zxi0_w2[i]=conv_zw2[i].real
    

    for tau in np.arange(0,N,int(deltaTau)):

        innerProd_xp=conv_zw1[tau-1].real #/ (2*N)
        innerProd_xq=conv_zw1[tau-1].imag #/ (2*N)
        innerProd_pp=0.5*(conv_zxi0_w2[tau-1] + conv_zw2[tau-1].real)# / (2*N)
        innerProd_qq=0.5*(conv_zxi0_w2[tau-1] - conv_zw2[tau-1].real) #/ (2*N)
        innerProd_pq=0.5*(conv_zw2[tau-1].imag ) #/ (2*N)
        
        a1 = innerProd_xp * innerProd_qq - innerProd_xq * innerProd_pq
        b1 = innerProd_xq * innerProd_pp - innerProd_xp * innerProd_pq
        
        if ((xi==0)or((xi*10000)==math.pi*10000)):
            optPhase=0
            if (math.fabs(innerProd_pp)>1e-10):
                innerProd = innerProd_xp /math.sqrt(innerProd_pp)
                
            else:
                innerProd=0.0
        elif (a1==0):
            optPhase=math.pi/2
            if (math.fabs(innerProd_qq>1e-10)):
                innerProd = -innerProd_xq / math.sqrt(innerProd_qq)
            else:
                innerProd=0.0
        else:
            optPhase = math.atan(-(b1/a1))
            if (math.fabs(a1*a1*innerProd_pp + b1*b1*innerProd_qq + 2*a1*b1*innerProd_pq)>1e-10):
                innerProd = (((a1 / math.fabs(a1)) * (innerProd_xp * a1 + innerProd_xq*b1)) /
                            (math.sqrt(a1*a1*innerProd_pp + b1*b1*innerProd_qq + 2*a1*b1*innerProd_pq)))

            else:
                innerProd=0.0
         #print(innerProd)
        if (math.fabs(innerProd)>math.fabs(maxInnerProd)):
            maxInnerProd = innerProd
            chosenTau = tau
            chosenOptPhase = optPhase
            #print('innerProd - {ip}, Tau - {tau}, optPhase - {phase}'.format(ip=maxInnerProd,tau=chosenTau,phase=chosenOptPhase))
        
        #f2.write("innerProd - {ip},   Tau : {tau},   optPhase : {phase},   xp:{xp} ,   xq:{xq}  pp:{pp} , qq:{qq} ,  pq:{pq} \n "
        #.format(ip=maxInnerProd,tau=chosenTau,phase=chosenOptPhase,xp=innerProd_xp,xq=innerProd_xq,pp=innerProd_pp,qq=innerProd_qq,pq=innerProd_pq))

        #f2.close
    #     f.write(f"IP : {innerProd:15.8f}  rho : {1/s:15.8f} xi : {xi:15.8f} optph - {chosenOptPhase:15.8f} ")
    #     f.write(f"chosenTau : {chosenTau:15.8f}  tau : {tau:15.8f} xp : {innerProd_xp:15.8f} xq - {innerProd_xq:15.8f} ")
    #     f.write(f"pp : {innerProd_pp:15.8f}  qq : {innerProd_qq:15.8f} pq : {innerProd_pq:15.8f} maxIp - {maxInnerProd:15.8f}\n ")
    # f.close()

     
        
    return maxInnerProd,chosenTau,chosenOptPhase     

def computeOptimumPhase(residue,opt_phase,innerProd,signalSize,xi,complexAtom,fileName,s,tau,N):

    innerProd = 0
    innerProdReal = 0.0
    innerProdImag = 0.0
    innerProdRealImag = 0.0

    complexDic=np.zeros(N,dtype = 'complex')

    for i in np.arange(signalSize):
        complexDic[i] = complexAtom[i]
        

        innerProdReal += residue[i]*complexAtom[i].real
        innerProdImag += residue[i]*complexAtom[i].imag
        innerProdRealImag += complexAtom[i].real * complexAtom[i].imag
    
    p = np.linalg.norm(complexDic[i].real)
    q = np.linalg.norm(complexDic[i].imag)

    a1 = innerProdReal*(q*q) - innerProdImag * innerProdRealImag
    b1 = innerProdImag*(p*p) - innerProdReal * innerProdRealImag

    if(xi == 0) or int(10000*xi) == (10000*np.pi):
        opt_phase = 0
        innerProd = innerProdReal/p
    elif(a1==0):
        opt_phase = np.pi/2
        innerProd = -innerProdImag/q
    elif(a1!=0 and xi!=0):
        opt_phase = np.arctan(-(b1/a1))
        innerProd = (a1/np.abs(a1))*(innerProdReal*a1+innerProdImag*b1)/np.sqrt(a1*a1*p*p+b1*b1*q*q+2*a1*b1*innerProdRealImag)

    


    return innerProd,opt_phase




def setParameters(parm,dicType,innerProd,s,xi,optPhase,tau,a,b,eta):
    if (dicType == 3):
        parm.setAtom(rho=0,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
    elif (dicType == 4):
        parm.setAtom(rho=1/s,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
    elif (dicType == 5):
        parm.setAtom(rho=1/s,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
    elif (dicType==6):
        parm.setAtom(rho=1/s,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
        
    elif (dicType==7):
        parm.setAtom(rho=1/s,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
        
    elif (dicType==8):
        parm.setAtom(rho=1/s,xi=xi,phase=optPhase,u=tau,a=a,b=b,eta=eta,innerProd=innerProd,s=s,dicType=dicType)
        
    else:
        print("DicType Inválido")


def adjustParameters(residue,parm):
    
    if (parm.dicType == 4):
        dic= Dictionary.GabDic()
        dic.setSignalSize(len(residue))
        dic.adjustParameters(residue,parm)
    elif (parm.dicType == 5):
        dic= Dictionary.TriangDic()
        dic.setSignalSize(len(residue))
        dic.adjustParameters(residue,parm)
    elif (parm.dicType == 6):
        dic = Dictionary.BateDic()
        dic.setSignalSize(len(residue))
        dic.adjustParameters(residue,parm)
    elif (parm.dicType == 7):
        dic = Dictionary.SigmoDic()
        dic.setSignalSize(len(residue))
        dic.adjustParameters(residue,parm)
    elif (parm.dicType == 8):
        dic = Dictionary.ChiDic()
        dic.setSignalSize(len(residue))
        dic.adjustParameters(residue,parm)
    else:
        print("Invalid Dictionary Type")

def updateResidue(residue,dicSize,parm):
    if (parm.dicType == 3):
        realAtom = np.zeros(dicSize)
        realAtom[parm.u] = 1
    if (parm.dicType == 4):
        dic = Dictionary.GabDic()
        dic.setSignalSize(dicSize)
        dic.setRealAtom(parm,dicSize)
        realAtom = dic.getRealAtom()

    elif (parm.dicType == 5):
        dic = Dictionary.TriangDic()
        dic.setSignalSize(dicSize)
        dic.setRealAtom(parm,dicSize)
        realAtom = dic.getRealAtom()

    elif (parm.dicType == 6):
        dic = Dictionary.BateDic()
        dic.setSignalSize(dicSize)
        dic.setRealAtom(parm,dicSize)
        realAtom = dic.getRealAtom()
    
    elif (parm.dicType == 7):
        dic = Dictionary.SigmoDic()
        dic.setSignalSize(dicSize)
        dic.setRealAtom(parm,dicSize)
        realAtom = dic.getRealAtom()
     
    elif (parm.dicType == 8):
        dic = Dictionary.ChiDic()
        dic.setSignalSize(dicSize)
        dic.setRealAtom(parm,dicSize)
        realAtom = dic.getRealAtom()

    else:
        print("Invalid Dictionary Type1")
        realAtom=np.zeros(dicSize)
    residue = residue - realAtom * parm.innerProd
    return residue


    

# inputFile = 'edaCvx1.wav'
# print(sys.argv)
#inputFile= 'Sample_Bate.wav'
def decompEDA(inputFile):

    
    
    origSignal,Fs = sf.read(inputFile)
    signalSize=len(origSignal)
    blockRange=PanelFiles.FileDecompBlockRange()
    blockRange.loadData('panelBlockRange.dat')
    

    decompData=PanelFiles.FileDecomp()
    decompData.loadData('panelDecomp.dat')
    if ( decompData.getFlagML()==1):
        net1 = load_model('Net1.h5')
        net2 = load_model('Net2.h5')
        net3 = load_model('Net3.h5')

    dicData=PanelFiles.FileDictionary()
    dicData.loadData('panelDictionary.dat')

    chosenDic=decompData.getDicType()
    chosenNet = 0
    numBlock=math.ceil(len(origSignal)/blockRange.getBlockHop())

    initBlock= blockRange.getInitBlock()
    finalBlock= blockRange.getFinalBlock()

    if (finalBlock==9999):
        finalBlock=numBlock
    
    #Creating a sba File
    f=open(str(inputFile).split(".")[0]+'_b'+str(initBlock)+'-'+str(finalBlock)+'.sba','w')

    f.write(f"Sign. Type :          {decompData.getSigType():5d}\n")               
    f.write(f"No. Signals:          {1:5d}\n") #Change if the signal has 2 or more channels
    f.write(f"Signal Size:       {len(origSignal):8d}\n")
    f.write(f"Block Hop:            {blockRange.getBlockHop():5d}\n")
    f.write(f"Block Size:           {blockRange.getBlockSize():5d}\n")
    f.write(f"Samp. Freq :     {Fs:10.2f}\n")
    f.write(f"Init. Block:          {initBlock:5d}\n")
    f.write(f"Final Block:          {finalBlock:5d}\n")
    f.write(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
    f.write(f"Signal:               {1:5d}\n")
    f.write(f"Norm:            {np.linalg.norm(origSignal):10.5f}\n")

    nMaxStep=decompData.getMaxIter()

    nbits=math.ceil(math.log10(blockRange.getBlockSize())/math.log10(2))
    dicSize=int(math.pow(2.0,nbits))

    residue=np.zeros(dicSize)
    signal=np.zeros(dicSize)
  
    #step=0
    #norm=0

    chosenParm=AtomClass.Atom()

    #chosenParm={'innerProd':0.0 , 's':0.0 , 'rho':0.0 , 'xi':0.0 ,
    #    'phase':0.0, 'u':0 , 'a':0 , 'b':0 , 'nextAtom':0 ,
    #     'prevAtom':0 , 'origAtomIndex':0 , 'eta':0.0 , 'dicType':0 }

    L=math.ceil(math.log10(dicSize)/math.log10(2))
    approxRatio=np.zeros(L)
    tolAppRatio=decompData.getApproxRatioTarget()
    snrTarget=decompData.getSNRTarget()

    #dicAtoms=[x for x in range(dicData.getNumDicBlock()*dicSize)]
    


    #for i in range(dicData.getNumDicBlock()*dicSize):
    #    dicAtoms[i]=AtomClass.Atom()
    dicAtoms=0
    #for i in range(numSignal): # Uncomment line if 2 or more signals were used

    if (np.linalg.norm(origSignal)!=0):
        
        for j in np.arange(initBlock-1,finalBlock):#finalBlock):
            residue=np.zeros(dicSize)
            a0=-1
            b0=dicSize
            befSupInnerP=0.0
            aftSupInnerP=0.0
            endFlag=0

            if (j * blockRange.getBlockHop()+dicSize <= signalSize):
                for i in np.arange(dicSize):
                    residue[i]=origSignal[j*blockRange.getBlockHop()+i]#:j*blockRange.getBlockHop()+dicSize+i]
                
            else:
                for i in np.arange(numBlock*dicSize-len(origSignal)):
                    residue[i]=origSignal[j*blockRange.getBlockHop()+i]
                

            initBlockNorm = np.linalg.norm(residue)
            residue = residue / np.linalg.norm(origSignal)
            #plt.plot(residue)
            #plt.title("{norm}".format(norm=np.linalg.norm(residue)))
            #plt.show()
           

            signal=residue
            

            f.write("--------------------------------------------------------------\n")
            f.write("Block:                    {b}\n".format(b=j+1))
            f.write("Norm:                     {blocknorm}\n".format(blocknorm=initBlockNorm))
            f.write("No.        Coef.           Decaying        Freq            Phase        Tau    Ti    Tf      Rising      dicType PrevAtom  AppRatio   meanAppRat befSup     aftSup     normRatio  SNR(dB)   chosenNet\n")
            

            #if (step>=nMaxStep):break
            for step in np.arange(nMaxStep):

                if ( decompData.getFlagML()==1):
                    chosenDic,chosenNet =  DANNO(residue,dicSize,chosenNet,step,L,approxRatio,net1,net2,net3)
                
                matchingPursuit(residue, chosenParm,dicSize,dicData,decompData,step,chosenDic,dicAtoms,a0,b0,decompData.getFlagOMP(),Fs)
                #print(chosenParm)
                approxRatio[step % L] = math.fabs(chosenParm.innerProd/np.linalg.norm(residue))
                meanApproxRatio=0.0
                for k in range(L):
                    meanApproxRatio+=approxRatio[k]/L
                
                adjustParameters(residue,chosenParm)
                
                if (decompData.getFlagOMP()==0):
                    
                    if (step==0):residue_log=residue

                    residue=updateResidue(residue,dicSize,chosenParm)
                    residue_log=np.vstack((residue_log,residue))
                    SNR=20*math.log10(initBlockNorm/np.linalg.norm(residue))
                    snr=20*(math.log10(1.0/(np.linalg.norm(residue)/initBlockNorm))/math.log10(10.0))
                elif (decompData.getFlagOMP()==1):
                    pass
                # updateResidue(residue,signal,b,v,Ai,a,initBlockNorm,step,prevAtoms,chosenParm)
                else:
                    print("FlagOMP - Valor inválido")

                f.write(f"{step+1:5d} {chosenParm.innerProd:15.8f} {chosenParm.rho:15.8f}") 
                f.write(f" {chosenParm.xi:15.8f} {chosenParm.phase:15.8f} {chosenParm.u:5d}")
                f.write(f" {chosenParm.a:5d} {chosenParm.b:5d} {chosenParm.eta:15.8f}")
                f.write(f" {int(chosenParm.dicType):5d}    {int(chosenParm.prevAtom):5d}    {approxRatio[step%L]:15.7f} {meanApproxRatio:10.7f}")
                f.write(f" {befSupInnerP:15.7f} {aftSupInnerP:15.7f} {np.linalg.norm(origSignal)/initBlockNorm:15.7f} {SNR:10.7f} {chosenNet:5d}\n")
                                

                if ((meanApproxRatio <= tolAppRatio) 
                    or (step >= nMaxStep-1) 
                    or (20*math.log10(initBlockNorm/np.linalg.norm(residue)) >= snrTarget) 
                    or (math.fabs(chosenParm.innerProd)<=1e-12) ): 
                    endFlag=1
                if ( endFlag==1):
                    f.write(f"99999\n")
                    if ( j == finalBlock):
                        f.write(f"88888\n")
                    break

                print("Bloco:  {j} ----- step: {step}".format(j=j,step=step))
    f.close()

parser = argparse.ArgumentParser( 
    description="Read file"
)
parser.add_argument("filename",help="Path to a file to process")
args = parser.parse_args()
print(args.filename)
inputFile = args.filename
start = timeit.default_timer()

decompEDA(inputFile)

stop = timeit.default_timer()

print('Time: ', stop - start)

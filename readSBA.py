import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
from scipy.io import wavfile

def readsba(fileName):
    
    Fs, origSignal = wavfile.read(fileName+'.wav')

    gabDic = Dictionary.GabDic()
    gabDic.setSignalSize(dicSize)
    triangDic = Dictionary.TriangDic()
    triangDic.setSignalSize(dicSize)
    bateDic = Dictionary.BateDic()
    bateDic.setSignalSize(dicSize)

    f=open(fileName+'_b1-36.sba','r')

    #Sign.Type
    string=f.readline()
    sigtype=int(string[14:])

    #Sign. No Signals
    string=f.readline()
    nSignal=int(string[14:])

    #Signal Size
    string=f.readline()
    signalSize=int(string[14:])

    #Block Hop
    string=f.readline()
    blockHop=int(string[14:])

    #Block Size
    string=f.readline()
    blockSize=int(string[14:])

    #Samp Freq
    string=f.readline()
    Fs=float(string[14:])

    #Init Block
    string=f.readline()
    initBlock=int(string[14:])

    #Final Block
    string=f.readline()
    finalBlock=int(string[14:])

    ############################

    i = 0
    ix = 0
    iSignal = 0
    iBlock = 0
    nMaxSignal = 1
    x = np.zeros((blockSize,1))

    energyExp = 0
    energyGab = 0
    energyImp = 0
    energySin = 0
    enExp = 0
    enGab = 0
    enImp = 0
    enSin = 0
    nBlock = 0

    if len(origSignal) < (initBlock-1)*blockHop+(blockSize):
        for j in  np.arange(0,(len(origSignal) - (initBlock-1)*blockHop)):
            x[j] = x[j] + origSignal[(j+(initBlock-1)*blockHop)]
    else:
        x = origSignal[((initBlock-1)*blockHop):(initBlock-1)*blockHop+(blockSize)]



    probDicType = np.zeros((blockSize,1))

    rcell={} 
    acell={}
    scell={}
    dcell={}
    ncell={}
    ecell={}


    while string[:5] !='88888':

        if ix==10000:
            break

        string=f.readline()


        if string[:4]=='XXXX':

            iSignal+=1
            string=f.readline()
            nSignal=int(string[8:])
            string=f.readline()
            normSignal=float(string[8:])

        elif string[:4]=='----':

            energyExp = 0
            energyGab = 0
            energyImp = 0
            energySin = 0
            enExp = 0
            enGab = 0
            enImp = 0
            enSin = 0
            i = 0
            ix = 0
            rmtx = []
            amtx = []
            dmtx = []
            smtx = []
            nmtx=[]
            emtx = np.zeros((3,512))
            #emtx = zeros(4,512);
            x = np.zeros((blockSize,1))
            iBlock +=1
            string = f.readline()
            nBlock = int(string[7:])
            string = f.readline()
            normBlock = float(string[7:])
            string = f.readline()
            if len(origSignal) < (nBlock-1)*blockHop+(blockSize):
                for j in np.arange((len(origSignal) - (nBlock-1)*blockHop)):
                    x[j] = x[j] + origSignal[(j+(nBlock-1)*blockHop)];
            else:
                x = origSignal[((nBlock-1)*blockHop):(nBlock-1)*blockHop+(blockSize)]
                x1 = x

        elif string[:4]=='####':
            print('Bloco com amostras nulas')
        elif string[:5]=='99999':
            continue
        elif string[:5]=='88888':
            continue

        else:
            #string=f.readline()

            sb=np.zeros(len(list(string.split())))
            for idx,si in enumerate(list(string.split())):
                sb[idx]=float(si)
            #print(sb)
            if ( (sb[0]!=99999) and (sb[0]!=88888) and (sb[0]!=77777) 
                and (nBlock>=initBlock) and (nBlock<=finalBlock)):

                if sb[9]==4:
                    realAtom,_=gengabor(sb[2:8],blockSize)
                    dmtx.append(sb[9])

                elif sb[9]==5:
                    realAtom,_=gentriang(sb[2:8],blockSize)
                    dmtx.append(sb[9])

                elif sb[9]==6:
                    realAtom,_=genbateman(sb[2:9],blockSize)
                    dmtx.append(sb[9])
                else:
                    print("Dicionário não encontrado")


                if sb[9]!=3:

                    rmtx.append(x)
                    amtx.append(sb[11])
                    smtx.append(sb[16])
                    nmtx.append(sb[17])
                    #print(sb)

                x = x-(normSignal*sb[1]*realAtom)

            if((ix == 512 or sb[16]>=13 ) and nBlock>= initBlock and
               nBlock<=finalBlock):

                rcell[iBlock] = rmtx;
                acell[iBlock] = amtx;
                scell[iBlock] = smtx;
                dcell[iBlock] = dmtx;
                ncell[iBlock] = nmtx;
                ecell[iBlock] = emtx;

            i+=1
            ix+=1

    
    f.close()
    return rcell,dcell,acell
#def readsba(fileName,origSignal,signal,initBlock,finalBlock):
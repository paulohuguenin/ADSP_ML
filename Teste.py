from typing import final
import numpy as np
from numpy import float64, linalg
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import sys
import math
from scipy.io import wavfile
from scipy import signal
from scipy.signal.signaltools import residue
import Dictionary
import PanelFiles
import AtomClass
import soundfile as sf


def setAtom1(parm):
    parm.setAtom(rho=0.2,eta=0.5)

def setAtom2():
    chosenParm=AtomClass.Atom()
    setAtom1(chosenParm)
    print(chosenParm)

inputFile='eda1'
initBlock=1
finalBlock=36

signalType=2
noSignal=1
sigSize=18343
blockHop=512
blockSize=512
Fs=8.00
initBlockNorm=1
resi=2.345435345
'''
f=open(str(inputFile)+'_b'+str(initBlock)+'-'+str(finalBlock)+'.sba','w')

f.write("Sign. Type :              {sigType}\n".format(sigType=signalType))  
f.write("No. Signals:              {noSig}\n".format(noSig=noSignal))
f.write("Block Hop:                {bhop}\n".format(bhop=blockHop))
f.write("Block Size:               {bsize}\n".format(bsize=blockSize))
f.write("Samp. Freq:               {Fs}\n".format(Fs=Fs))
f.write("Init. Block:              {iB}\n".format(iB=initBlock))
f.write("Final Block:              {fB}\n".format(fB=finalBlock))
f.write("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
f.write("Signal:                   {Sign}\n".format(Sign=noSignal))
f.write("Norm:                     {norm}\n".format(norm=initBlockNorm))

f.write("--------------------------------------------------------------\n")
f.write("Block:                    {b}\n".format(b=1+1))
f.write("Norm:                     {norm}\n".format(norm=resi))
f.write("No.        Coef.           Decaying        Freq            Phase        Tau    Ti    Tf      Rising      dicType PrevAtom  AppRatio   meanAppRat befSup     aftSup     normRatio  SNR(dB)   chosenNet\n")
coef=2.323423423423
indice=1
f.write(f"{indice:3d}  {resi:.8f} {coef:.8f} \n")
f.write(f"{indice*10:3d}   {resi:.8f} {coef:.8f} \n")
f.close()
'''
Fs, origSignal = wavfile.read('eda1.wav')
print(np.linalg.norm(origSignal[0:512]))

m_norm=float64(0.0)
for i in np.arange(len(origSignal)):
    m_norm += float64(origSignal[i])*float64(origSignal[i])
m_norm=np.sqrt(m_norm)
print(m_norm)
print(type(m_norm))

data, fs = sf.read('eda1.wav')
print(np.linalg.norm(data))

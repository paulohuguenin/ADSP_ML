#from _typeshed import Self
import numpy as np
import soundfile as sf
import Dictionary
import AtomClass


class FileDecomp():

    def __init__(self):

        self.sigType = 0
        self.dicType =0
        self.MPType =0
        self.flagOMP =0
        self.flagML =0
        self.maxIter = 0
        self.flagHeur=0
        self.coefBlockSub = 0
        self.atomCont = 0
        self.flagSupTemp = 0
        self.flagOptDecay = 0
        self.approxRatioTarget = 0.0
        self.SNRTarget = 0.0
        self.flagPrintDecompStage = 0
        self.loadRDFile = 0
        self.blockHop = 0
        self.blockSize = 0
        self.initBlock = 0
        self.endBlock = 0
        self.coefSupTemp = 0
        self.flagSeqBlock = 0
        self.coefSeqBlock = 0
       
        

    def loadData(self,fileName):

        self.fileName=fileName

        f=open(self.fileName)

        #Empty Line
        string=f.readline()

        #Sign.Type
        string=f.readline()
        self.sigType=int(string[39:50])
        
        #MP. Type
        string=f.readline()
        self.MPType=int(string[39:50])

        #Dic Type
        string=f.readline()
        self.dicType=int(string[39:50])

        #Flag OMP
        string=f.readline()
        self.flagOMP=int(string[39:50])

        #Flag ML/ANN
        string=f.readline()
        self.flagML=int(string[39:50])

        #Número máx de iterações
        string=f.readline()
        self.maxIter=int(string[39:50])

        #Coef. for temporal support
        string=f.readline()
        self.coefSupTemp=float(string[39:50])

        # Flag Heuristic
        string=f.readline()
        self.flagHeur=int(string[39:50])

        #coef sequence blocks heuristic
        string=f.readline()
        self.coefBlockSub=float(string[39:50])
                
        # Flag for evaluating atom continuity
        string=f.readline()
        self.atomCont=int(string[39:50])
        
        # Flag Temporal Support
        string=f.readline()
        self.flagSupTemp=int(string[39:50])

        #Flag Optimum Decaying
        string=f.readline()
        self.flagOptDecay=int(string[39:50])

        # Approximation Ratio Target
        string=f.readline()
        self.approxRatioTarget=float(string[39:50])
        
        # SNR Target
        string=f.readline()
        self.SNRTarget=float(string[39:50])
        
        # Flag print Decomp
        string=f.readline()
        self.flagPrintDecompStage=int(string[39:50])

        # Flag Load RD File
        string=f.readline()
        self.loadRDFile=int(string[39:50])

        f.close()
    
    def getSigType(self):
        return self.sigType

    def getMPType(self):
        return self.MPType
    
    def getDicType(self):
        return self.dicType

    def getFlagOMP(self):
        return self.flagOMP
    
    def getFlagML(self):
        return self.flagML
        
    def getMaxIter(self):
        return self.maxIter
    
    def getCoefSupTemp(self):
        return self.coefSupTemp
        
    def getFlagHeur(self):
        return self.flagHeur
        
    def getCoefBlockSub(self):
        return self.coefBlockSub
    
    def getAtomCont(self):
        return self.atomCont

    def getFlagSupTemp(self):
        return self.flagSupTemp
    
    def getFlagOptDecay(self):
        return self.flagOptDecay

    def getApproxRatioTarget(self):
        return self.approxRatioTarget
    
    def getSNRTarget(self):
        return self.SNRTarget

    def getFlagPrintDecompStage(self):
        return self.flagPrintDecompStage
    
    def getLoadRDFile(self):
        return self.loadRDFile
    
class FileDecompBlockRange():

    def __init__(self):
        
        self.blockHop=512
        self.blockSize=512
        self.initBlock=1
        self.finalBlock=9999
    
    def loadData(self,fileName):

        self.fileName=fileName

        f=open(self.fileName)

        #Empty Line
        string=f.readline()

        #Block Hop
        string=f.readline()
        self.blockHop=int(string[39:50])

        #Block Size
        string=f.readline()
        self.blockSize=int(string[39:50])

        #Init Block
        string=f.readline()
        self.initBlock=int(string[39:50])

        #Final Block
        string=f.readline()
        self.finalBlock=int(string[39:50])

        f.close()
    
    def getBlockHop(self):
        return self.blockHop

    def getBlockSize(self):
        return self.blockSize
    
    def getInitBlock(self):
        return self.initBlock
    
    def getFinalBlock(self):
        return self.finalBlock

class FileDictionary():

    def __init__(self):
        self.panelDic=[]

    def loadData(self,fileName):

        self.numDic=0

        self.fileName=fileName
        f=open(self.fileName)

        #Empty Line
        string=f.readline()
        string=f.readline()
    
        while string[:5] !='99999':
            if self.numDic==10000:
                break

            string=f.readline()
            if (string=='99999'):break
            sb=np.zeros(len(list(string.split())))
            
            #self.panelDic=[]

            #for idx,si in enumerate(list(string.split())):
               # sb[idx]=float(si)

            sb[0] = float(string.split()[0]) #Scale
            sb[1] = float(string.split()[1]) #WinShift
            sb[2] = int(string.split()[2])   #Freq. Discr.
            sb[3] = float(string.split()[3]) #Init. Freq.
            sb[4] = float(string.split()[4]) #Final Freq
            sb[5] = int(string.split()[5])   #DicType
            sb[6] = float(string.split()[6]) #Decay/Rho
            sb[7] = float(string.split()[7]) #Rise/Eta
            
            self.panelDic.append(sb)

            self.numDic+=1
        f.close()

    def getDicData(self):
        return self.panelDic
    
    def getNumDicBlock(self):
        return self.numDic

class StructBook():

    def __init__(self):
        
        self.sigType = 0 #Signal Type -> {2:Audio 3:ECG 4:Noise 5:EDA}
        self.nSignal = 0 #Number of Signals or Channels
        self.signalSize = 0 #Signal Size -> len(origSignal)
        self.blockHop = 0  #Hop ->{128 256 512 1024}
        self.blockSize = 0 #Length of the Block ->{128 256 512 1024}
        self.Fs = 0.0 # Sample Frequency
        self.initBlock = 0 
        self.finalBlock = 0
        self.rcell = {} # Struct Book Residue
        self.acell = {} # Struct Book Aproxx Ratio
        self.scell = {} # Struct Book SNR
        self.dcell = {} # Struct Book Dictionary
        self.ncell = {} # Struct Book Net
        self.ecell = {} # Struct Book Energy
    
    def readSBA(self,fileName):
        origSignal,Fs = sf.read(fileName + '.wav')

        f=open(fileName+'_b1-36.sba','r')

        #Sign.Type
        string=f.readline()
        self.sigtype=int(string[14:])

        #Sign. No Signals
        string=f.readline()
        self.nSignal=int(string[14:])

        #Signal Size
        string=f.readline()
        self.signalSize=int(string[14:])

        #Block Hop
        string=f.readline()
        self.blockHop=int(string[14:])

        #Block Size
        string=f.readline()
        self.blockSize=int(string[14:])

        #Samp Freq
        string=f.readline()
        self.Fs=float(string[14:])

        #Init Block
        string=f.readline()
        self.initBlock=int(string[14:])

        #Final Block
        string=f.readline()
        self.finalBlock=int(string[14:])

        ############################
        gabDic = Dictionary.GabDic()
        gabDic.setSignalSize(self.blockSize)
        triangDic = Dictionary.TriangDic()
        triangDic.setSignalSize(self.blockSize)
        bateDic = Dictionary.BateDic()
        bateDic.setSignalSize(self.blockSize)
        i = 0
        ix = 0
        iSignal = 0
        iBlock = 0
        nMaxSignal = 1
        x = np.zeros((self.blockSize,1))

        #energyExp = 0
        #energyGab = 0
        #energyImp = 0
        #energySin = 0
        #enExp = 0
        #enGab = 0
        #enImp = 0
        #enSin = 0
        nBlock = 0

        if len(origSignal) < (self.initBlock-1)*self.blockHop+(self.blockSize):
            for j in  np.arange(0,(len(origSignal) - (self.initBlock-1)*self.blockHop)):
                x[j] = x[j] + origSignal[(j+(self.initBlock-1)*self.blockHop)]
        else:
            x = origSignal[((self.initBlock-1)*self.blockHop):(self.initBlock-1)*self.blockHop+(self.blockSize)]



        probDicType = np.zeros((self.blockSize,1))

        self.rcell={} 
        self.acell={}
        self.scell={}
        self.dcell={}
        self.ncell={}
        self.ecell={}


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

                #energyExp = 0
                #energyGab = 0
                #energyImp = 0
                #energySin = 0
                #enExp = 0
                #enGab = 0
                #enImp = 0
                #enSin = 0
                i = 0
                ix = 0
                rmtx = []
                amtx = []
                dmtx = []
                smtx = []
                nmtx=[]
                emtx = np.zeros((3,512))
                #emtx = zeros(4,512);
                x = np.zeros((self.blockSize,1))
                iBlock +=1
                string = f.readline()
                nBlock = int(string[7:])
                string = f.readline()
                normBlock = float(string[7:])
                string = f.readline()
                if len(origSignal) < (nBlock-1)*self.blockHop+(self.blockSize):
                    for j in np.arange((len(origSignal) - (nBlock-1)*self.blockHop)):
                        x[j] = x[j] + origSignal[(j+(nBlock-1)*self.blockHop)]
                else:
                    x = origSignal[((nBlock-1)*self.blockHop):(nBlock-1)*self.blockHop+(self.blockSize)]
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
                    parm=AtomClass.Atom()
                    sb[idx]=float(si)
                #print(sb)
                if ( (sb[0]!=99999) and (sb[0]!=88888) and (sb[0]!=77777) 
                    and (nBlock>=self.initBlock) and (nBlock<=self.finalBlock)):

                    if sb[9]==4:
                        parm.setAtom(rho=sb[2],xi=sb[3],phase=sb[4],u=sb[5],a=sb[6],b=sb[7],eta=sb[8])
                        gabDic.setRealAtom(parm,self.blockSize)
                        realAtom = gabDic.getRealAtom()
                        dmtx.append(sb[9])

                    elif sb[9]==5:
                        parm.setAtom(rho=sb[2],xi=sb[3],phase=sb[4],u=sb[5],a=sb[6],b=sb[7],eta=sb[8])
                        triangDic.setRealAtom(parm,self.blockSize)
                        realAtom = triangDic.getRealAtom()
                        dmtx.append(sb[9])

                    elif sb[9]==6:
                        parm.setAtom(rho=sb[2],xi=sb[3],phase=sb[4],u=sb[5],a=sb[6],b=sb[7],eta=sb[8])
                        bateDic.setRealAtom(parm,self.blockSize)
                        realAtom = bateDic.getRealAtom()
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

                if((ix == 512 or sb[16]>=13 ) and nBlock>= self.initBlock and
                nBlock<=self.finalBlock):

                    self.rcell[iBlock] = rmtx
                    self.acell[iBlock] = amtx
                    self.scell[iBlock] = smtx
                    self.dcell[iBlock] = dmtx
                    self.ncell[iBlock] = nmtx
                    self.ecell[iBlock] = emtx

                i+=1
                ix+=1

        
        f.close()
        return self.rcell,self.dcell,self.acell
        #def readsba(fileName,origSignal,signal,initBlock,finalBlock):
    def getSignalType(self):
        return self.sigType
    def getNSignal(self):
        return self.nSignal
    def getSignalSize(self):
        return self.signalSize
    def getBlockHop(self):
        return self.blockHop
    def getBlockSize(self):
        return self.blockSize
    def getFs(self):
        return self.Fs
    def getInitBlock(self):
        return self.initBlock
    def getFinalBlock(self):
        return self.finalBlock
    def getResidue(self):
        return self.rcell
    def getAproxRatio(self):
        return self.acell
    def getSNR(self):
        return self.scell
    def getDic(self):
        return self.dcell
    def getNet(self):
        return self.ncell
    def getEnergy(self):
        return self.ecell

#readDecomp=FileDecompBlockRange()
#filename='panelBlockRange.dat'
#readDecomp.loadData(filename)
##print(readDecomp.finalBlock)
##if readDecomp.finalBlock==9999:
##    readDecomp.finalBlock=math.ceil(origSignal.size/readDecomp.blockSize)
#import pandas as pd
#dicData = FileDictionary()
#dicData.loadData('panelDictionary.dat')
#panelDic = dicData.getDicData()
#df=pd.DataFrame(panelDic,columns=["Scale","WinShift","FreqDiscr","InitFreq(Hz)","FinalFreq(Hz)","DicType","Decay","Rise"])
#print(df)
#print(df.loc[df['Decay']==0.03])


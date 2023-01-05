#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import signal
from scipy.stats import chi2
from scipy.special import gamma
#import pandas as pd

#######################

class Dictionary():

    def __init__(self,):
        self.m_signalSize=0
        self.m_complexAtom=[]
        self.m_realAtom=[]
        self.rAtomNorm=0
    
    def setSignalSize(self,signalSize):
        self.signalSize=signalSize
    

    def setComplexAtom(self,parameters):
        pass

    def setRealAtom(self,parameters):
        pass

    def getSignalSize(self,parameters):
        pass
    
    def getComplexAtom(self,parameters):
        pass

    def getRealAtom(self,parameters):
        pass

class BateDic():

    def __init__(self):
        self.rho = 0
        self.xi  = 0
        self.phi = 0
        self.u   = 0
        self.a   = 0
        self.b   = 0
        self.beta = 0 
        self.signalSize=0
        self.complexAtom=[]
        self.realAtom=[]
        self.normr=0.0
    
    def setSignalSize(self,signalSize):
        self.signalSize=signalSize
    
    def getSignalSize(self,parameters):
        return self.signalSize    

    def setComplexAtom(self,parameters,N):
        self.rho = parameters.rho
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        self.beta = parameters.eta

        self.complexAtom=np.zeros(N,dtype='complex')
        delta=0
        realPeak=0
        imagPeak=0

        if (self.rho>=0):
            self.u = self.a
            for n in np.arange(N):
                
                if((n<self.u) or (n>self.b)):
                    self.realAtom[n] = 0
                else:
                    if(self.xi!=0):
                        
                        self.complexAtom[n] = complex(( math.exp( -self.rho * (n-self.u) ) - math.exp( -self.beta * (n-self.u) ) ) * math.cos ( (self.xi*n) + self.phi )
                            ,(math.exp(-self.rho*n-self.u)-math.exp(-self.beta*(n-self.u))*math.sin(self.xi*n)))
                    else:
                        self.complexAtom[n] = complex(1,0.0)

                if ( math.fabs(realPeak) < math.fabs(self.complexAtom[n].real) ) :
                    realPeak = self.complexAtom[n].real
                
                if ( math.fabs(imagPeak) < math.fabs(self.complexAtom[n].imag) ) :
                    imagPeak = self.complexAtom[n].imag
                    
        else:
            self.u = N - 1 - self.b
            for n in np.arange(N):
                self.complexAtom[N-n]=0
                if( n >= self.u ):
                    if(self.xi!=0):
                        self.complexAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( (self.xi*(N-1-n)) + self.phi )
                    else:
                        self.complexAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( self.phi )
                    
                    if ( math.fabs(peak) < math.fabs(self.complexAtom[N-n]) ): 
                        peak = self.complexAtom[N-n]



    

    def getComplexAtom(self):
        return self.complexAtom

    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        self.beta = parameters.eta

        self.realAtom = np.zeros(N)
        delta=0
        peak=0
        if (self.rho>0):
            self.u = self.a
            for n in np.arange(N):
                self.realAtom[n]=0
                if((n<self.u) or (n>self.b)):
                    self.realAtom[n] = 0
                else:
                    if(self.xi!=0):
                        self.realAtom[n] = ( math.exp( -self.rho * (n-self.u) ) - math.exp( -self.beta * (n-self.u) ) ) * math.cos ( (self.xi*n) + self.phi )
                    else:
                        self.realAtom[n] = ( math.exp( -self.rho * ((n-self.u)) ) - math.exp( -self.beta * (n-self.u) ) ) * math.cos ( self.phi )
            
                if ( math.fabs(peak) < math.fabs(self.realAtom[n]) ) :
                    peak = self.realAtom[n]
                if (math.fabs(peak) < 1e-10): 
                    for n in np.arange(N): 
                        self.realAtom[n] = 0
        else:
            self.u = N - 1 - self.b
            for n in np.arange(N):
                self.realAtom[N-n]=0
                if( n >= self.u ):
                    if(self.xi!=0):
                        self.realAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( (self.xi*(N-1-n)) + self.phi )
                    else:
                        self.realAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( self.phi )
                    
                    if ( math.fabs(peak) < math.fabs(self.realAtom[N-n]) ): 
                        peak = self.realAtom[N-n]
            
        if (np.linalg.norm(self.realAtom)!=0):
            self.normr = np.linalg.norm(self.realAtom)
            self.realAtom = self.realAtom/np.linalg.norm(self.realAtom)

    def getRealAtom(self):
        return self.realAtom
    
    def getAtomNorm(self):
        return self.normr

    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio
    
class GabDic():

    def __init__(self):
        self.s   = 0
        self.rho = 0
        self.xi  = 0
        self.phi = 0
        self.u   = 0
        self.a   = 0
        self.b   = 0
        self.beta = 0 
        self.signalSize=0
        self.complexAtom=[]
        self.realAtom=[]
        self.normr=0.0
    
    def setSignalSize(self,signalSize):
        self.signalSize=signalSize
    
    def getSignalSize(self,parameters):
        return self.signalSize    

    def setComplexAtom(self,parameters,N):
        
        self.rho = parameters.rho
        self.s   = parameters.s
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        

        self.complexAtom=np.zeros(N,dtype='complex')
        delta=0
        peak=0
        if (self.rho==0):
            
            for n in np.arange(N):
                
                if(self.xi!=0):
                    self.complexAtom[n] = complex(math.pow(2,0.25)*math.cos(self.xi*n),math.pow(2,0.25)*math.sin(self.xi*n))
                else:               
                    # self.complexAtom[n] = complex(math.pow(2,0.25)*1.0,0.0)
                    self.complexAtom[n] = complex(1.0,0.0)
                     
                if ( math.fabs(peak) < math.fabs(self.complexAtom[n].real) ) :
                    realPeak = self.complexAtom[n].real
                
                if ( math.fabs(peak) < math.fabs(self.complexAtom[n].imag) ) :
                    imagPeak = self.complexAtom[n].imag
                    
        else:
            
            for n in np.arange(N):
          
                if( self.xi != 0 ):
                    self.complexAtom[n] = complex(math.pow(2,0.25) * math.exp(-np.pi*math.pow((n-self.u)*self.rho,2)) * math.cos(self.xi*n),
                                             math.pow(2,0.25) * math.exp(-np.pi*math.pow((n-self.u)*self.rho,2)) *math.sin(self.xi*n))               
                else:              
                    self.complexAtom[n] = complex(math.pow(2,0.25) * math.exp(-np.pi*math.pow((n-self.u)*self.rho,2)),0.0)
                    
                if ( math.fabs(peak) < math.fabs(self.complexAtom[n].real) ) :
                    realPeak = self.complexAtom[n].real
                
                if ( math.fabs(peak) < math.fabs(self.complexAtom[n].imag) ) :
                    imagPeak = self.complexAtom[n].imag


    

    def getComplexAtom(self):
        return self.complexAtom

    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho
        self.s   = parameters.s
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        self.beta = parameters.eta

        self.realAtom = np.zeros(N)
        delta=0
        peak=0

        for n in np.arange(N):

            if(self.xi!=0):
                self.realAtom[n] = math.pow(2,0.25) * math.exp(-np.pi*math.pow((n-self.u)*self.rho,2)) * math.cos((self.xi*n)+self.phi)
            else:
                self.realAtom[n] =math.pow(2,0.25) * math.exp(-np.pi*math.pow((n-self.u)*self.rho,2)) * math.cos(self.phi)
 
            if ( math.fabs(peak) < math.fabs(self.realAtom[n]) ) :
                peak = self.realAtom[n]
            if (math.fabs(peak) < 1e-10): 
                for n in np.arange(N): 
                    self.realAtom[n] = 0
    
            
        if (np.linalg.norm(self.realAtom)!=0):
            self.normr = np.linalg.norm(self.realAtom)
            self.realAtom = self.realAtom/np.linalg.norm(self.realAtom)

    def getRealAtom(self):
        return self.realAtom
    
    def getAtomNorm(self):
        return self.normr

    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio

class TriangDic():

    def __init__(self):
        self.s   = 0
        self.rho = 0
        self.xi  = 0
        self.phi = 0
        self.u   = 0
        self.a   = 0
        self.b   = 0
        self.beta = 0 
        self.signalSize=0
        self.complexAtom=[]
        self.realAtom=[]
        self.normr=0.0
    
    def setSignalSize(self,signalSize):
        self.signalSize=signalSize
    
    def getSignalSize(self,parameters):
        return self.signalSize    

    def setComplexAtomExp(self,parameters,N):
        
        self.rho = parameters.rho
        self.s   = parameters.s
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b

        realPeak = 0
        imagPeak = 0
        delta = 0

        self.complexAtom=np.zeros(N,dtype='complex')

        if( self.rho>=0):
            self.u=self.a
            for n in np.arange(N):

                if(n>=self.u):
                    if(self.xi!=0):
                        self.complexAtom[n] = complex(math.exp(-self.rho*(n-self.u))*math.cos(self.xi*n),
                                            math.exp(-self.rho*(n-self.u))*math.sin(self.xi*n))
                    else:
                        self.complexAtom[n] = complex(math.exp(-self.rho*(n-self.u)),0.0)
                
                if( math.fabs(realPeak) < math.fabs(self.complexAtom[n].real) ) :
                    realPeak = self.complexAtom[n].real
            
                if ( math.fabs(imagPeak) < math.fabs(self.complexAtom[n].imag) ) :
                    imagPeak = self.complexAtom[n].imag
        else:
            self.u = N - 1 - self.b
            for n in np.arange(N):
                self.complexAtom[N-n]=0
                if( n >= self.u ):
                    if(self.xi!=0):
                        self.complexAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( (self.xi*(N-1-n)) + self.phi )
                    else:
                        self.complexAtom[N-n] = ( math.exp( -self.rho * (self.beta*n-self.u) ) - math.exp( -self.rho * (delta*n-self.u) ) ) * math.cos ( self.phi )
                    
                    if ( math.fabs(peak) < math.fabs(self.complexAtom[N-n]) ): 
                        peak = self.complexAtom[N-n]    

        for n in np.arange(self.a):
            self.complexAtom[n] = complex(0.0,0.0)
        for n in np.arange(self.b+1,N):
            self.complexAtom[n] = complex(0.0,0.0)
    def getComplexAtomExp(self):
        return self.complexAtom
    def setComplexAtom(self,parameters,N):
        
        self.rho = parameters.rho
        self.s   = parameters.s
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        
        TL = self.u-self.a
        TC = self.u
        TR = self.u + self.s - 1 - self.a
        h  = 2/(TR-TL)

        self.complexAtom=np.zeros(N,dtype='complex')
        delta=0
        realPeak=0
        imagPeak=0

        for n in np.arange(N):
            
            if(n <= TL and TC !=TL):
                self.complexAtom[n] = complex(0,0)
            elif(TL < n and n < TC):               
                self.complexAtom[n] = complex(h * ((n - TL)/(TC-TL))*math.cos( self.xi * n ),math.sin(self.xi*n))
            elif(n == TC):
                self.complexAtom[n] = complex(h*math.cos( self.xi * n ),math.sin(self.xi*n))
            elif(TC < n and n < TR):
                self.complexAtom[n] = complex(h * ((TR-n)/(TR - TC))*math.cos(self.xi*n) , math.sin(self.xi*n) )
            elif(n >= TR and TC != TR):
                self.complexAtom[n] = complex(0,0)

            if( math.fabs(realPeak) < math.fabs(self.complexAtom[n].real) ) :
                realPeak = self.complexAtom[n].real
            
            if ( math.fabs(imagPeak) < math.fabs(self.complexAtom[n].imag) ) :
                imagPeak = self.complexAtom[n].imag
                        
        for n in np.arange(TL):
            self.complexAtom[n] = complex(0,0)
        for n in np.arange(math.ceil(TR),N):
            self.complexAtom[n] = complex(0,0)          
    
    
    def getComplexAtom(self):
        return self.complexAtom

    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho
        self.s   = parameters.s
        self.xi  = parameters.xi
        self.phi = parameters.phase
        self.u   = parameters.u
        self.a   = parameters.a
        self.b   = parameters.b
        self.beta = parameters.eta

        TL = self.u-self.a
        TC = self.u
        TR = self.u + self.s - 1 - self.a
        h  = 2/(TR-TL)

        self.realAtom=np.zeros(N)
        delta=0
        peak=0           
        for n in np.arange(N):
                
            if(n <= TL and TC !=TL):
                self.realAtom[n] = 0
            elif(TL < n and n < TC):               
                self.realAtom[n] = h * ((n-TL)/(TC - TL))*math.cos(self.xi*n + self.phi)
            elif(n == TC):
                self.realAtom[n] = h*math.cos( self.xi * n + self.phi )
            elif(TC < n and n < TR):
                self.realAtom[n] = h * ((TR-n)/(TR - TC))*math.cos(self.xi*n + self.phi)
            elif(n >= TR and TC != TR):
                self.realAtom[n] = 0

            if( math.fabs(peak) < math.fabs(self.realAtom[n]) ) :
                peak = self.realAtom[n]
        
        if ( math.fabs(peak) < 1e-10):
            self.realAtom = np.zeros(N)
        else:
            self.realAtom=self.realAtom/np.linalg.norm(self.realAtom)
    
    
    def getRealAtom(self):
        return self.realAtom
    
    def getAtomNorm(self):
        return self.normr

    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio

class SigmoDic():

    def __init__(self):
        self.s = 0
        self.rho = 0
        self.xi = 0
        self.phi = 0
        self.u = 0
        self.beta = 0
        self.signalSize = 0
        self.complexAtom = []
        self.realAtom = []
        self.normr = 0.0

    def setSignalSize(self,signalSize):
        self.signalSize = signalSize
    
    def getSignalSize(self,parameters):
        return self.signalSize
    
    def setComplexAtom(self,parameters,N):
        self.rho = parameters.rho
        self.xi = parameters.xi
        self.phi = parameters.phase
        self.u = parameters.u
        self.eta = parameters.eta
        # self.s = parameters.s
       
        self.complexAtom = np.zeros(N,dtype='complex')
        
        for n in np.arange(N):
            self.complexAtom[n] = 0
            
            if(self.xi != 0):
                self.complexAtom[n] = complex(
                    math.exp(-(self.rho*(n-self.u)))/(1+((self.eta*(n-self.u)))**-2)**2
                    *math.cos((n*self.xi)+self.phi)
                    ,math.exp(-(self.rho*(n-self.u)))/(1+((self.eta*(n-self.u)))**-2)**2
                    *math.sin((n*self.xi)+self.phi))
            else:
                self.complexAtom[n] = complex(1,0.0)
        
    def getComplexAtom(self):
        return self.complexAtom
    
    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho
        self.xi = parameters.xi
        self.phi = parameters.phase
        self.u = parameters.u
        self.eta = parameters.eta
        # self.s = parameters.s
        self.b = parameters.b
        peak=0

        self.realAtom = np.zeros(N)

        for n in np.arange(N):
            #self.realAtom[n]=0
            if n<self.u or n>self.b:
                self.realAtom[n] = 0
            else:
                if(self.xi != 0):
                    self.realAtom[n] = (math.exp(-(self.rho*(n-self.u)))/(1+(self.eta*(n-self.u))**-2)**2) * math.cos((n*self.xi)+self.phi)
                else:
                    self.realAtom[n] = (math.exp(-(self.rho*(n-self.u)))/(1+(self.eta*(n-self.u))**-2)**2)*math.cos(self.phi)
                    #print(self.realAtom[n])
            if ( math.fabs(peak) < math.fabs(self.realAtom[n]) ) :
                    peak = self.realAtom[n]
            if (math.fabs(peak) < 1e-10): 
                for n in np.arange(N): 
                    self.realAtom[n] = 0
                    
        if (np.linalg.norm(self.realAtom)!=0):
            self.normr = np.linalg.norm(self.realAtom)
            self.realAtom = self.realAtom/np.linalg.norm(self.realAtom)


    def getRealAtom(self):
        return self.realAtom
    
    def getAtomNorm(self):
        return self.normr

    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio
 
 
class ChiDic():
    def __init__(self):
        self.rho = 0
        self.u = 0
        self.signalSize = 0
        self.complexAtom = []
        self.realAtom = []
        self.normr = []
        
    def setSignalSize(self,signalSize):
        self.signalSize = signalSize
        
    def getSignalSize(self,parameters): 
        return self.signalSize
        
    def setComplexAtom(self,parameters,N): 
        self.rho = parameters.rho #Scale
        self.u = parameters.u
        self.xi = parameters.xi
        self.phi = parameters.phase
        self.eta = parameters.eta
        self.b = parameters.b
        
        self.complexAtom = np.zeros(N,dtype = 'complex')
        
        for n in np.arange(N):
           #for k in np.arange(self.rho):
            if n<self.u or n>self.b:
                self.realAtom[n] = 0
            else:
                if (self.xi != 0):
                    self.complexAtom[n] = complex(chi2.pdf(n,self.eta,self.u,1/self.rho)*math.cos(self.xi*n+self.phi),
                    chi2.pdf(n,self.eta,self.u,1/self.rho)*math.sin(self.xi*n+self.phi))
                else:
                    self.complexAtom[n] = complex(1.0,0)

    def getComplexAtom(self):
        return self.complexAtom
        
    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho #Funciona como escala
        self.eta = parameters.eta #Funciona como grau de liberdade K
        self.u = parameters.u
        self.xi = parameters.xi
        self.phi = parameters.phase
        #self.s = parameters.s
        
        self.realAtom = np.zeros(N)
        #for n in np.arange(N):
           #self.realAtom[n] = ((n-self.u)^((self.eta/2)-1)*np.exp(-(n-self.u)/2))/(self.s*2^(self.eta/2)*gamma(self.eta/2))*np.cos(self.xi*n+self.phi)
        
        n=np.arange(N)
        self.realAtom = chi2.pdf(n,self.eta,self.u,1/self.rho)*np.cos(self.xi*n+self.phi) 
        
        if (np.linalg.norm(self.realAtom)!=0):
            self.normr = np.linalg.norm(self.realAtom)
            self.realAtom = self.realAtom/np.linalg.norm(self.realAtom) 
                 
                    
    def getRealAtom(self):
        return self.realAtom
        
    def getAtomNorm(self):
        return self.normr
    
    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio

class ChiDic2():
    def __init__(self):
        self.rho = 0
        self.u = 0
        self.signalSize = 0
        self.complexAtom = []
        self.realAtom = []
        self.normr = []
        
    def setSignalSize(self,signalSize):
        self.signalSize = signalSize
        
    def getSignalSize(self,parameters): 
        return self.signalSize
        
    def setComplexAtom(self,parameters,N): 
        self.rho = parameters.rho #Scale
        self.u = parameters.u
        self.xi = parameters.xi
        self.phi = parameters.phase
        self.eta = parameters.eta
        self.b = parameters.b
        
        self.complexAtom = np.zeros(N,dtype = 'complex')
        
        for n in np.arange(N):
           #for k in np.arange(self.rho):
            if n<self.u or n>self.b:
                self.realAtom[n] = 0
            else:
                if (self.xi != 0):
                    self.complexAtom[n] = complex(chi2.pdf(n,self.eta,self.u,self.rho)*math.cos(self.xi*n+self.phi),
                    chi2.pdf(n,self.eta)*math.sin(self.xi*n+self.phi))
                else:
                    self.complexAtom[n] = complex(1.0,0)

    def getComplexAtom(self):
        return self.complexAtom
        
    def setRealAtom(self,parameters,N):
        self.rho = parameters.rho #Funciona como escala
        self.eta = parameters.eta #Funciona como grau de liberdade K
        self.u = parameters.u
        self.xi = parameters.xi
        self.phi = parameters.phase
        #self.s = parameters.s
        
        self.realAtom = np.zeros(N)
        for n in np.arange(N):
            self.realAtom[n] = (((self.rho*(n-self.u))**((self.eta/2)-1)*math.exp(-(self.rho*(n-self.u))/2))/(2**(self.eta/2)*math.gamma(self.eta/2)))*np.cos(self.xi*n+self.phi)
            #self.realAtom[n] = ((n-self.u)^((self.eta/2)-1)*np.exp(-(n-self.u)/2))/(self.s*2^(self.eta/2)*gamma(self.eta/2))*np.cos(self.xi*n+self.phi)
        
        # n=np.arange(N)
        # self.realAtom = chi2.pdf(n,self.eta,self.u,self.rho)*np.cos(self.xi*n+self.phi) 
        
        if (np.linalg.norm(self.realAtom)!=0):
            self.normr = np.linalg.norm(self.realAtom)
            self.realAtom = self.realAtom/np.linalg.norm(self.realAtom) 
                 
                    
    def getRealAtom(self):
        return self.realAtom
        
    def getAtomNorm(self):
        return self.normr
    
    def adjustParameters(self,residue,chosenParm):

        if (int(1e10*chosenParm.phase)==0):
            chosenParm.phase=0.0
        if (chosenParm.phase >= (2*math.pi)):
            chosenParm.phase -= 2*math.pi
        if (chosenParm.phase < 0 ):
            chosenParm.phase += 2*math.pi
        if (chosenParm.innerProd < 0.0):
            chosenParm.phase += math.pi
            chosenParm.innerProd = - chosenParm.innerProd
    
    def getApproxRatio(self,signalSize):

        lambda_med_ger=[0.3,0.22,0.18,0.13,0.12,0.09,0.065]

        if (signalSize == 64):
            tolAppRatio = lambda_med_ger[0]
        elif (signalSize == 128):
            tolAppRatio = lambda_med_ger[1]
        elif (signalSize == 256):
            tolAppRatio = lambda_med_ger[2]
        elif ( signalSize == 512):
            tolAppRatio = lambda_med_ger[3]
        elif ( signal == 1024):
            tolAppRatio = lambda_med_ger[4]
        elif (signalSize == 2048):
            tolAppRatio = lambda_med_ger[5]
        elif (signalSize == 4096):
            tolAppRatio = lambda_med_ger[6]
        else:
            tolAppRatio=0.0
        return tolAppRatio
    

    
#N=512
#parms=[0.2,0,0,N/2,N/2,N-1,0.4]
#realAtom,_=bateAtom.setRealAtom(parms,N)
#print(realAtom)
#plt.plot(realAtom)
#plt.show()
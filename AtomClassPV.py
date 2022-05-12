class Atom():
    
    def __init__(self):
        self.innerProd=0.0
        self.s=0.0
        self.rho=0.0
        self.xi=0.0
        self.phase=0.0
        self.u=0
        self.a=0
        self.b=0
        self.nextAtom=0
        self.prevAtom=0
        self.origAtomIndex=0
        self.eta=0.0
        self.dicType=0
        
    def __str__(self):
        return "innerProd=" + str(self.innerProd) + ", rho=" + str(self.rho) + "Tau=" + str(self.u) + "eta=" +str(self.eta) + "dicType=" + str(self.dicType)
    def setAtom(self,rho=0.0,xi=0.0,phase=0.0,u=0,a=0,b=0,eta=0.0,innerProd=0.0,s=0.0,nextAtom=0,prevAtom=0,origAtomIndex=0,dicType=0):
        self.innerProd=innerProd
        self.s=s
        self.rho=rho
        self.xi=xi
        self.phase=phase
        self.u=u
        self.a=a
        self.b=b
        self.nextAtom=nextAtom
        self.prevAtom=prevAtom
        self.origAtomIndex=origAtomIndex
        self.eta=eta
        self.dicType=dicType

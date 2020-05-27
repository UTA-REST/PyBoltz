import math

cdef class Ang:
    """
    This object is used to set angle cuts on angular distribution and renormalise forward scattering probability.
    """
    def __init__(self):
        self.AngCut = 0.0
        self.ScatteringParameter1=0.0
        self.ScatteringParameter2=0.0

    def CalcAngCut(self):
        '''Function used to calculate the Angle cuts on the angular distribution.'''
        self.AngCut=1
        self.ScatteringParameter2=self.ScatteringParameter1
        if self.ScatteringParameter1 <=1:
            return self.AngCut
        cdef double Pi,CNS,ThetaC,FAC
        Pi = math.acos(-1)
        CNS = self.ScatteringParameter1-0.5
        ThetaC = math.asin(2*math.sqrt(CNS-CNS**2))
        FAC =(1-math.cos(ThetaC)/(math.sin(ThetaC)**2))
        self.ScatteringParameter2 = (CNS*FAC)+0.5
        self.AngCut = ThetaC*2./Pi


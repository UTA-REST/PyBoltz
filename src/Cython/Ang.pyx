import math

cdef class Ang:
    """
    This object is used to set angle cuts on angular distribution and renormalise forward scattering probability.
    """
    def __init__(self):
        self.ANGC = 0.0
        self.PSCT1=0.0
        self.PSCT2=0.0

    def AngCut(self):
        self.ANGC=1
        self.PSCT2=self.PSCT1
        if self.PSCT1 <=1:
            return self.ANGC
        cdef double API,RADS,CNS,THETAC,FAC
        API = math.acos(-1)
        RADS = 2/API
        CNS = self.PSCT1-0.5
        THETAC = math.asin(2*math.sqrt(CNS-CNS**2))
        FAC =(1-math.cos(THETAC)/(math.sin(THETAC)**2))
        self.PSCT2 = (CNS*FAC)+0.5
        self.ANGC = THETAC*RADS


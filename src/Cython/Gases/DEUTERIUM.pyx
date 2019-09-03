from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange
cimport GasUtil

sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
cdef void Gas22(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Deuterium gas.
    """
    gd = np.load('gases.npy').item()

    cdef double XEN[53],YXSEC[53],XROT0[40],YROT0[40],XROT1[42],YROT1[42],XROT2[31],YROT2[31],XROT3[31],YROT3[31],XROT4[31],YROT4[31],
    cdef double XROT5[30],YROT5[30],XVIB1[35],YVIB1[35],XVIB2[35],YVIB2[35],XVIB3[16],YVIB3[16],XVIB4[16],YVIB4[16],XEXC1[20],YEXC1[20],
    cdef double XEXC2[23],YEXC2[23],XATT[18],YATT[18],XION[72],YION[72]

    XEN=gd['gas22/XEN']
    YXSEC=gd['gas22/YXSEC']
    XROT0=gd['gas22/XROT0']
    YROT0=gd['gas22/YROT0']
    XROT1=gd['gas22/XROT1']
    YROT1=gd['gas22/YROT1']
    XROT2=gd['gas22/XROT2']
    YROT2=gd['gas22/YROT2']
    XROT3=gd['gas22/XROT3']
    YROT3=gd['gas22/YROT3']
    XROT4=gd['gas22/XROT4']
    YROT4=gd['gas22/YROT4']
    XROT5=gd['gas22/XROT5']
    YROT5=gd['gas22/YROT5']
    XVIB1=gd['gas22/XVIB1']
    YVIB1=gd['gas22/YVIB1']
    XVIB2=gd['gas22/XVIB2']
    YVIB2=gd['gas22/YVIB2']
    XVIB3=gd['gas22/XVIB3']
    YVIB3=gd['gas22/YVIB3']
    XVIB4=gd['gas22/XVIB4']
    YVIB4=gd['gas22/YVIB4']
    XEXC1=gd['gas22/XEXC1']
    YEXC1=gd['gas22/YEXC1']
    XEXC2=gd['gas22/XEXC2']
    YEXC2=gd['gas22/YEXC2']
    XATT=gd['gas22/XATT']
    YATT=gd['gas22/YATT']
    XION=gd['gas22/XION']
    YION=gd['gas22/YION']

    cdef double PJ[7],B0,FROT[8],SUM
    cdef int i,j,k,I,J,NL

    # CALCULATE FRACTIONAL POPULATION DENSITY FOR ROTATIONAL STATES
    B0=0.00377272

    for I in range(1,8,2):
        PJ[I-1] = 3*(2*I+1)*exp(-1*I*(I+1)*B0/object.AKT)

    for I in range(2,7,2):
        PJ[I-1] = 6*(2*I+1)*exp(-1*I*(I+1)*B0/object.AKT)

    SUM = 6.0

    for I in range(7):
        SUM+=PJ[I]

    FROT[0] = 6.0/SUM

    for I in range(1,8):
        FROT[I] = PJ[I-1]/SUM

    object.NION = 1
    object.NATT = 1
    object.NIN = 15
    object.NNULL = 0

    for J in range(6):
        object.KEL[J] = 0

    for J in range(object.NIN):
        object.KIN[J] = 0


    cdef int NDATA,NROT0,NROT1,NROT2,NROT3,NROT4,NROT5,NVIB1,NVIB2,NVIB3,NVIB4,NEXC1,NEXC2,NIOND,NATT1

    NDATA=53
    NROT0=40
    NROT1=42
    NROT2=31
    NROT3=31
    NROT4=31
    NROT5=30
    NVIB1=35
    NVIB2=35
    NVIB3=16
    NVIB4=16
    NEXC1=20
    NEXC2=23
    NIOND=72
    NATT1=18

    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27,EOBY

    object.E = [0.0, 1.0, 15.427, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (4.028204 * AMU)

    EOBY = 8.3

    object.EIN = gd['gas22/EIN']

    cdef double EN

    EN = -1*object.EnergyStep/2.0

    for I in range(4000):
        EN+=object.EnergyStep
        object.Q[1][I] = GasUtil.CALQIONREG(EN, NDATA, YXSEC, XEN)

        object.Q[2][I] = 0.0
        if EN>object.E[2]:
            object.Q[2][I] = GasUtil.CALQIONREG(EN, NIOND, YION, XION)

        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0
        if EN>XATT[0]:
            object.Q[3][I] = GasUtil.CALQIONREG(EN, NATT1, YATT, XATT)
            object.QATT[0][I] = object.Q[3][I]

        object.Q[5][I] = 0.0
        object.Q[4][I] = 0.0

        # SUPERELASTIC 2-0
        object.QIN[0][I] = 0.0
        if EN>0.0:
            object.QIN[0][I] = GasUtil.CALQIONREG(EN +object.EIN[3], NROT0, YROT0, XROT0)
            object.QIN[0][I]*=((object.EIN[3]+EN)/EN)*FROT[2]*0.2

        # SUPERELASTIC 3-1
        object.QIN[1][I] = 0.0
        if EN>0.0:
            object.QIN[1][I] = GasUtil.CALQIONREG(EN +object.EIN[4], NROT1, YROT1, XROT1)
            object.QIN[1][I]*=((object.EIN[4]+EN)/EN)*FROT[3]*(3.0/7.0)

        # SUPERELASTIC 4-2
        object.QIN[2][I] = 0.0
        if EN>0.0:
            object.QIN[2][I] = GasUtil.CALQIONREG(EN +object.EIN[5], NROT2, YROT2, XROT2)
            object.QIN[2][I]*=((object.EIN[5]+EN)/EN)*FROT[4]*(5.0/9.0)

        # ROTATION 0-2
        object.QIN[3][I] = 0.0
        if EN>object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQIONREG(EN, NROT0, YROT0, XROT0)*FROT[0]

        # ROTATION 1-3
        object.QIN[4][I] = 0.0
        if EN>object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQIONREG(EN, NROT1, YROT1, XROT1)*FROT[1]

        # ROTATION 2-4
        object.QIN[5][I] = 0.0
        if EN>object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQIONREG(EN, NROT2, YROT2, XROT2)*FROT[2]

        # ROTATION 3-5
        object.QIN[6][I] = 0.0
        if EN>object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQIONREG(EN, NROT3, YROT3, XROT3)*FROT[3]

        # ROTATION 4-6 + 6-8
        object.QIN[7][I] = 0.0
        if EN>object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQIONREG(EN, NROT4, YROT4, XROT4)*(FROT[4]+FROT[6])

        # ROTATION 5-7 + 7-9
        object.QIN[8][I] = 0.0
        if EN>object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQIONREG(EN, NROT5, YROT5, XROT5)*(FROT[5]+FROT[7])


        object.QIN[9][I] = 0.0
        if EN>object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQIONREG(EN, NVIB1, YVIB1, XVIB1)

        object.QIN[10][I] = 0.0
        if EN>object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQIONREG(EN, NVIB2, YVIB2, XVIB2)

        object.QIN[11][I] = 0.0
        if EN>object.EIN[11]:
            object.QIN[11][I] = GasUtil.CALQIONREG(EN, NVIB3, YVIB3, XVIB3)

        object.QIN[12][I] = 0.0
        if EN>object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQIONREG(EN, NVIB4, YVIB4, XVIB4)

        object.QIN[13][I] = 0.0
        if EN>object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQIONREG(EN, NEXC1, YEXC1, XEXC1)


        object.QIN[14][I] = 0.0
        if EN>object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQIONREG(EN, NEXC2, YEXC2, XEXC2)

        object.Q[0][I] = object.Q[1][I]+object.Q[2][I]+object.Q[3][I]+object.QIN[13][I]+object.QIN[14][I]

        if EN<200:
            for J in range(13):
                object.Q[1][I]-=object.QIN[J][I]

    

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return















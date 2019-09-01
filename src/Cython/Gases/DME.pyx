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
cdef void Gas25(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for DME gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[54],YXSEC[54],XION[29],YION[29],XATT[16],YATT[16],XVIB3[19],YVIB3[19],XVIB4[28],YVIB4[28],XVIB5[25]
    cdef double XVIB6[19],YVIB6[19],XEXC[27],YEXC[27],XEXC1[35],YEXC1[35],YVIB5[25]

    cdef int NDATA,NVIB3,NVIB4,NVIB5,NVIB6,NIOND,NATT1,NEXC,NEXC1
    XEN=gd['gas25/XEN']
    YXSEC=gd['gas25/YXSEC']
    XION=gd['gas25/XION']
    YION=gd['gas25/YION']
    XATT=gd['gas25/XATT']
    YATT=gd['gas25/YATT']
    XVIB3=gd['gas25/XVIB3']
    YVIB3=gd['gas25/YVIB3']
    XVIB4=gd['gas25/XVIB4']
    YVIB4=gd['gas25/YVIB4']
    XVIB5=gd['gas25/XVIB5']
    YVIB5=gd['gas25/YVIB5']
    XVIB6=gd['gas25/XVIB6']
    YVIB6=gd['gas25/YVIB6']
    XEXC=gd['gas25/XEXC']
    YEXC=gd['gas25/YEXC']
    XEXC1=gd['gas25/XEXC1']
    YEXC1=gd['gas25/YEXC1']
    object.EIN = gd['gas25/EIN']
    # ---------------------------------------------------------------------
    # UPDATES DME97 WITH MONTE CARLO SIMULATION OF STEADY STATE TOWNSEND
    #  VALUE FOR ALPHA.
    # UPDATES DME94 WITH CORRECT VIBRATIONAL ANALYSIS FROM SVERDLOV.
    # UPDATES DME92 WITH BETTER FIT TO FANO AND EV/ION PAIR
    # ---------------------------------------------------------------------
    cdef double AVIB1,AVIB2
    cdef int i,j,I,J
    object.NION = 1
    object.NATT = 1
    object.NIN = 8
    object.NNULL = 0
    AVIB1=<float>(0.06)
    AVIB2=<float>(0.35)
    for J in range(6):
        object.KEL[J] = 0

    for J in range(object.NIN):
        object.KIN[J] = 0

    NDATA=54
    NVIB3=19
    NVIB4=28
    NVIB5=25
    NVIB6=19
    NIOND=29
    NATT1=16
    NEXC=27
    NEXC1=35

    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27,EOBY

    object.E = [0.0, 1.0, <float>(10.04), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * EMASS / (<float>(46.06904) * AMU)

    EOBY = <float>(10.04)

    cdef double APOP, EN,EFAC
    APOP = exp(object.EIN[0]/object.AKT)

    EN = -1*object.ESTEP/<float>(2.0)
    for I in range(4000):
        EN+=object.ESTEP

        object.Q[1][I] = GasUtil.CALQIONREG(EN, NDATA, YXSEC, XEN)

        object.Q[2][I] = 0.0
        if EN>object.E[2]:
            object.Q[2][I] = GasUtil.CALQIONREG(EN, NIOND, YION, XION)


        object.Q[3][I] = 0
        object.QATT[0][I] = 0.0
        if EN>XATT[0]:
            object.Q[3][I] = GasUtil.CALQION(EN, NATT1, YATT, XATT)*1e-5
            object.QATT[0][I] = object.Q[3][I]

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        #SUPERELASTIC OF VIBRATION
        object.QIN[0][I] = 0.0
        if EN != 0.0:
            EFAC = sqrt(1.0 - (object.EIN[0]/EN))
            object.QIN[0][I] = AVIB1 *log((EFAC+<float>(1.0))/(EFAC-<float>(1.0)))/EN
            object.QIN[0][I]  = object.QIN[0][I] * APOP/(<float>(1.0)+APOP)*1e-16

        object.QIN[1][I] = 0.0
        if EN >object.EIN[1]:
            EFAC = sqrt(<float>(1.0) - (object.EIN[1]/EN))
            object.QIN[1][I] = AVIB1 *log((EFAC+<float>(1.0))/(<float>(1.0)-EFAC))/EN
            object.QIN[1][I]  = object.QIN[1][I] * 1/(<float>(1.0)+APOP)*1e-16

        object.QIN[2][I] = 0.0
        if EN >object.EIN[2]:
            object.QIN[2][I] = GasUtil.CALPQ3(EN, NVIB3, YVIB3, XVIB3)
            EFAC = sqrt(1.0 - (object.EIN[2]/EN))
            object.QIN[2][I] = (object.QIN[2][I]+ AVIB2 *log((EFAC+<float>(1.0))/(<float>(1.0)-EFAC))/EN)*1e-16

        object.QIN[3][I] = 0.0
        if EN >object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQIONREG(EN, NVIB4, YVIB4, XVIB4)

        object.QIN[4][I] = 0.0
        if EN >object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQIONREG(EN, NVIB5, YVIB5, XVIB5)

        object.QIN[5][I] = 0.0
        if EN >object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQIONREG(EN, NVIB6, YVIB6, XVIB6)

        object.QIN[6][I] = 0.0
        if EN >object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQIONREG(EN, NEXC, YEXC, XEXC)
            
        object.QIN[7][I] = 0.0
        if EN >object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQIONREG(EN, NEXC1, YEXC1, XEXC1)

        object.Q[0][I] =0.0
        for J in range(1,4):
            object.Q[0][I]+=object.Q[J][I]

        for J in range(8):
            object.Q[0][I]+=object.QIN[J][I]


    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
            



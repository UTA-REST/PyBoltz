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

    cdef double PJ[7],B0,FROT[8],Sum
    cdef int i,j,k,I,J,NL

    # CALCULATE FRACTIONAL POPULATION DENSITY FOR ROTATIONAL STATES
    B0=<float>(0.00377272)

    for I in range(1,8,2):
        PJ[I-1] = 3*(2*I+1)*exp(-1*I*(I+1)*B0/object.ThermalEnergy)

    for I in range(2,7,2):
        PJ[I-1] = 6*(2*I+1)*exp(-1*I*(I+1)*B0/object.ThermalEnergy)

    Sum = 6.0

    for I in range(7):
        Sum+=PJ[I]

    FROT[0] = 6.0/Sum

    for I in range(1,8):
        FROT[I] = PJ[I-1]/Sum

    object.N_Ionization = 1
    object.N_Attachment = 1
    object.N_Inelastic = 15
    object.N_Null = 0

    for J in range(6):
        object.AngularModel[J] = 0

    for J in range(object.N_Inelastic):
        object.KIN[J] = 0


    cdef int NDATA,NROT0,NROT1,NROT2,NROT3,NROT4,NROT5,NVIB1,NVIB2,NVIB3,NVIB4,NEXC1,NEXC2,N_IonizationD,N_Attachment1

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
    N_IonizationD=72
    N_Attachment1=18

    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27,EOBY

    object.E = [0.0, 1.0, <float>(15.427), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float>(4.028204) * AMU)

    EOBY = 8.3

    object.EnergyLevels = gd['gas22/EnergyLevels']

    cdef double EN

    EN = -1*object.EnergyStep/2.0

    for I in range(4000):
        EN+=object.EnergyStep
        object.Q[1][I] = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YXSEC, XEN)

        object.Q[2][I] = 0.0
        if EN>object.E[2]:
            object.Q[2][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationD, YION, XION)

        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        if EN>XATT[0]:
            object.Q[3][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_Attachment1, YATT, XATT)
            object.AttachmentCrossSection[0][I] = object.Q[3][I]

        object.Q[5][I] = 0.0
        object.Q[4][I] = 0.0

        # SUPERELASTIC 2-0
        object.InelasticCrossSectionPerGas[0][I] = 0.0
        if EN>0.0:
            object.InelasticCrossSectionPerGas[0][I] = GasUtil.CALIonizationCrossSectionREG(EN +object.EnergyLevels[3], NROT0, YROT0, XROT0)
            object.InelasticCrossSectionPerGas[0][I]*=((object.EnergyLevels[3]+EN)/EN)*FROT[2]*0.2

        # SUPERELASTIC 3-1
        object.InelasticCrossSectionPerGas[1][I] = 0.0
        if EN>0.0:
            object.InelasticCrossSectionPerGas[1][I] = GasUtil.CALIonizationCrossSectionREG(EN +object.EnergyLevels[4], NROT1, YROT1, XROT1)
            object.InelasticCrossSectionPerGas[1][I]*=((object.EnergyLevels[4]+EN)/EN)*FROT[3]*(3.0/7.0)

        # SUPERELASTIC 4-2
        object.InelasticCrossSectionPerGas[2][I] = 0.0
        if EN>0.0:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALIonizationCrossSectionREG(EN +object.EnergyLevels[5], NROT2, YROT2, XROT2)
            object.InelasticCrossSectionPerGas[2][I]*=((object.EnergyLevels[5]+EN)/EN)*FROT[4]*(5.0/9.0)

        # ROTATION 0-2
        object.InelasticCrossSectionPerGas[3][I] = 0.0
        if EN>object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT0, YROT0, XROT0)*FROT[0]

        # ROTATION 1-3
        object.InelasticCrossSectionPerGas[4][I] = 0.0
        if EN>object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT1, YROT1, XROT1)*FROT[1]

        # ROTATION 2-4
        object.InelasticCrossSectionPerGas[5][I] = 0.0
        if EN>object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT2, YROT2, XROT2)*FROT[2]

        # ROTATION 3-5
        object.InelasticCrossSectionPerGas[6][I] = 0.0
        if EN>object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT3, YROT3, XROT3)*FROT[3]

        # ROTATION 4-6 + 6-8
        object.InelasticCrossSectionPerGas[7][I] = 0.0
        if EN>object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT4, YROT4, XROT4)*(FROT[4]+FROT[6])

        # ROTATION 5-7 + 7-9
        object.InelasticCrossSectionPerGas[8][I] = 0.0
        if EN>object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALIonizationCrossSectionREG(EN, NROT5, YROT5, XROT5)*(FROT[5]+FROT[7])


        object.InelasticCrossSectionPerGas[9][I] = 0.0
        if EN>object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB1, YVIB1, XVIB1)

        object.InelasticCrossSectionPerGas[10][I] = 0.0
        if EN>object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB2, YVIB2, XVIB2)

        object.InelasticCrossSectionPerGas[11][I] = 0.0
        if EN>object.EnergyLevels[11]:
            object.InelasticCrossSectionPerGas[11][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB3, YVIB3, XVIB3)

        object.InelasticCrossSectionPerGas[12][I] = 0.0
        if EN>object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB4, YVIB4, XVIB4)

        object.InelasticCrossSectionPerGas[13][I] = 0.0
        if EN>object.EnergyLevels[13]:
            object.InelasticCrossSectionPerGas[13][I] = GasUtil.CALIonizationCrossSectionREG(EN, NEXC1, YEXC1, XEXC1)


        object.InelasticCrossSectionPerGas[14][I] = 0.0
        if EN>object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][I] = GasUtil.CALIonizationCrossSectionREG(EN, NEXC2, YEXC2, XEXC2)

        object.Q[0][I] = object.Q[1][I]+object.Q[2][I]+object.Q[3][I]+object.InelasticCrossSectionPerGas[13][I]+object.InelasticCrossSectionPerGas[14][I]

        if EN<200:
            for J in range(13):
                object.Q[1][I]-=object.InelasticCrossSectionPerGas[J][I]

    

    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    return















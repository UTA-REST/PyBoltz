from Gases.CF4 cimport Gas1
from Gases.ARGON cimport Gas2
from Gases.HELIUM4 cimport Gas3
from Gases.HELIUM3 cimport Gas4
from Gases.NEON cimport Gas5
from Gases.KRYPTON cimport Gas6
from Gases.XENON cimport Gas7
from Gases.CH4 cimport Gas8
from Gases.ETHANE cimport Gas9
from Gases.PROPANE cimport Gas10
from Gases.ISOBUTANE cimport Gas11
from Gases.CO2 cimport Gas12
from Gases.H2O cimport Gas14
from Gases.OXYGEN cimport Gas15
from Gases.NITROGEN cimport Gas16
from Gases.HYDROGEN cimport Gas21
from Gases.DEUTERIUM cimport Gas22
from Gases.DME cimport Gas25
from libc.string cimport memset

from Gas cimport Gas

cdef void callGASF(Gas* GAS):
    if GAS.NGS == 1:
        Gas1(GAS)
    elif GAS.NGS == 2:
        Gas2(GAS)
    elif GAS.NGS == 3:
        Gas3(GAS)
    elif GAS.NGS == 4:
        Gas4(GAS)
    elif GAS.NGS == 5:
        Gas5(GAS)
    elif GAS.NGS == 6:
        Gas6(GAS)
    elif GAS.NGS == 7:
        Gas7(GAS)
    elif GAS.NGS == 8:
        Gas8(GAS)
    elif GAS.NGS == 9:
        Gas9(GAS)
    elif GAS.NGS == 10:
        Gas10(GAS)
    elif GAS.NGS == 11:
        Gas11(GAS)
    elif GAS.NGS == 12:
        Gas12(GAS)
    elif GAS.NGS == 14:
        Gas14(GAS)
    elif GAS.NGS == 15:
        Gas15(GAS)
    elif GAS.NGS == 16:
        Gas16(GAS)
    elif GAS.NGS == 21:
        Gas21(GAS)
    elif GAS.NGS == 22:
        Gas22(GAS)
    elif GAS.NGS == 25:
        Gas25(GAS)

cdef class Gasmix:
    def InitWithInfo(self, NGS, QIN, NIN, PENFRA, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NumberOfGases, NSTEP,
                     NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN,PIR2):
        '''This functions simply initiates the gas data from the parameters. This functions fills the output arrays to zeros.'''
        cdef int i,j;
        for i in range(6):
            self.Gases[i].NGS = NGS[i]
            for j in range(250):
                self.Gases[i].QIN[j][:] = QIN[i][j]
            self.Gases[i].NIN = NIN[i]
            for j in range(3):
                self.Gases[i].PENFRA[j][:] = PENFRA[i][j]
            self.Gases[i].EG = EG
            self.Gases[i].EROOT = EROOT
            self.Gases[i].QT1 = QT1
            self.Gases[i].QT2 = QT2
            self.Gases[i].QT3 = QT3
            self.Gases[i].QT4 = QT4
            self.Gases[i].DEN = DEN
            self.Gases[i].DENS = DENS
            self.Gases[i].NumberOfGases = NumberOfGases
            self.Gases[i].NSTEP = NSTEP
            self.Gases[i].NANISO = NANISO
            self.Gases[i].EFINAL = EFINAL
            self.Gases[i].AKT = AKT
            self.Gases[i].ESTEP = ESTEP
            self.Gases[i].ARY = ARY
            self.Gases[i].TEMPC = TEMPC
            self.Gases[i].TORR = TORR
            self.Gases[i].IPEN = IPEN
            self.Gases[i].PIR2 = PIR2
            memset(self.Gases[i].Q, 0, 6*4000 * sizeof(double))
            memset(self.Gases[i].QION, 0, 30*4000 * sizeof(double))
            memset(self.Gases[i].PEQION, 0, 30*4000 * sizeof(double))
            memset(self.Gases[i].QATT, 0, 8*4000 * sizeof(double))
            memset(self.Gases[i].QNULL, 0, 10*4000 * sizeof(double))






    def setCommons(self, NGS, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NumberOfGases, NSTEP,
                   NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN,PIR2):
        '''This functions is used to fill the common main gas mixing inputs.'''
        for i in range(6):
            self.Gases[i].NGS = NGS[i]
            self.Gases[i].EG[:] = EG[:]
            self.Gases[i].EROOT[:] = EROOT[:]
            self.Gases[i].QT1[:] = QT1[:]
            self.Gases[i].QT2[:] = QT2[:]
            self.Gases[i].QT3[:] = QT3[:]
            self.Gases[i].QT4[:] = QT4[:]
            self.Gases[i].DEN[:] = DEN[:]
            self.Gases[i].DENS = DENS
            self.Gases[i].NumberOfGases = NumberOfGases
            self.Gases[i].NSTEP = NSTEP
            self.Gases[i].NANISO = NANISO
            self.Gases[i].EFINAL = EFINAL
            self.Gases[i].AKT = AKT
            self.Gases[i].ESTEP = ESTEP
            self.Gases[i].ARY = ARY
            self.Gases[i].TEMPC = TEMPC
            self.Gases[i].TORR = TORR
            self.Gases[i].IPEN = IPEN
            memset(self.Gases[i].Q, 0, 6*4000 * sizeof(double))
            memset(self.Gases[i].QION, 0, 30*4000 * sizeof(double))
            memset(self.Gases[i].PEQION, 0, 30*4000 * sizeof(double))
            memset(self.Gases[i].QATT, 0, 8*4000 * sizeof(double))
            memset(self.Gases[i].QNULL, 0, 10*4000 * sizeof(double))
    def Run(self):
        '''This functions calls the corresponding gas functions.'''
        cdef int i
        cdef Gas temp
        for i in range(6):

            callGASF(&self.Gases[i])

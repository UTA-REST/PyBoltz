from CF4 import Gas1
from ARGON import Gas2
import h5py
from types import MethodType
import multiprocessing as mp
from multiprocessing import Pool
from Gas import Gas


class Gasmix:

    def __init__(self):
        # First Setup
        self.Gases = [Gas() for i in range(6)]


    def InitWithInfo(self, NGS, Q, QIN, NIN, E, EI, KIN, QION, PEQION, EION, EB, PEQEL, PEQIN, KEL, PENFRA, NC0, EC0,
                     WK,
                     EFL, NG1, EG1, NG2, EG2, QATT, QNULL, SCLN, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NGAS, NSTEP,
                     NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN):
        self.__init__()
        # First Setup
        for i in range(6):
            self.Gases[i].NGS = NGS[i]
            self.Gases[i].Q = Q[i]
            self.Gases[i].QIN = QIN[i]
            self.Gases[i].NIN = NIN[i]
            self.Gases[i].E = E[i]
            self.Gases[i].EI = EI[i]
            self.Gases[i].KIN = KIN[i]
            self.Gases[i].QION = QION[i]
            self.Gases[i].PEQION = PEQION[i]
            self.Gases[i].EION = EION[i]
            self.Gases[i].EB = EB[i]
            self.Gases[i].PEQEL = PEQEL[i]
            self.Gases[i].PEQIN = PEQIN[i]
            self.Gases[i].KEL = KEL[i]
            self.Gases[i].PENFRA = PENFRA[i]
            self.Gases[i].NC0 = NC0[i]
            self.Gases[i].EC0 = EC0[i]
            self.Gases[i].WK = WK[i]
            self.Gases[i].EFL = EFL[i]
            self.Gases[i].NG1 = NG1[i]
            self.Gases[i].EG1 = EG1[i]
            self.Gases[i].NG2 = NG2[i]
            self.Gases[i].EG2 = EG2[i]
            self.Gases[i].QATT = QATT[i]
            self.Gases[i].QNULL = QNULL[i]
            self.Gases[i].SCLN = SCLN[i]
            self.Gases[i].EG = EG
            self.Gases[i].EROOT = EROOT
            self.Gases[i].QT1 = QT1
            self.Gases[i].QT2 = QT2
            self.Gases[i].QT3 = QT3
            self.Gases[i].QT4 = QT4
            self.Gases[i].DEN = DEN
            self.Gases[i].DENS = DENS
            self.Gases[i].NGAS = NGAS
            self.Gases[i].NSTEP = NSTEP
            self.Gases[i].NANISO = NANISO
            self.Gases[i].EFINAL = EFINAL
            self.Gases[i].AKT = AKT
            self.Gases[i].ESTEP = ESTEP
            self.Gases[i].ARY = ARY
            self.Gases[i].TEMPC = TEMPC
            self.Gases[i].TORR = TORR
            self.Gases[i].IPEN = IPEN

    def setCommons(self, NGS, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NGAS, NSTEP,
                   NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN):
        for i in range(6):
            self.Gases[i].NGS = NGS[i]
            self.Gases[i].EG = EG
            self.Gases[i].EROOT = EROOT
            self.Gases[i].QT1 = QT1
            self.Gases[i].QT2 = QT2
            self.Gases[i].QT3 = QT3
            self.Gases[i].QT4 = QT4
            self.Gases[i].DEN = DEN
            self.Gases[i].DENS = DENS
            self.Gases[i].NGAS = NGAS
            self.Gases[i].NSTEP = NSTEP
            self.Gases[i].NANISO = NANISO
            self.Gases[i].EFINAL = EFINAL
            self.Gases[i].AKT = AKT
            self.Gases[i].ESTEP = ESTEP
            self.Gases[i].ARY = ARY
            self.Gases[i].TEMPC = TEMPC
            self.Gases[i].TORR = TORR
            self.Gases[i].IPEN = IPEN

    def Run(self):
        result=[]
        p = Pool()
        for i in range(6):
            if self.Gases[i].NGS != 0:
                result.append( p.apply_async(globals()['Gas' + str(self.Gases[i].NGS)], [self.Gases[i]]))
        p.close()
        p.join()
        for i in range(6):
            self.Gases[i]=result[i].get()

from ARGON cimport Gas2
from CF4 cimport Gas1

from Gas cimport Gas

cdef void callGASF(Gas* GAS):
    if GAS.NGS == 1:
        Gas1(GAS)
    elif GAS.NGS == 2:
        Gas2(GAS)


cdef class Gasmix:
    def InitWithInfo(self, NGS, QIN, NIN, PENFRA, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NGAS, NSTEP,
                     NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN):
        # First Setup
        for i in range(6):
            self.Gases[i].NGS = NGS[i]
            self.Gases[i].QIN = QIN[i]
            self.Gases[i].NIN = NIN[i]
            self.Gases[i].PENFRA = PENFRA[i]
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
            self.Gases[i].EG[:] = EG[:]
            self.Gases[i].EROOT[:] = EROOT[:]
            self.Gases[i].QT1[:] = QT1[:]
            self.Gases[i].QT2[:] = QT2[:]
            self.Gases[i].QT3[:] = QT3[:]
            self.Gases[i].QT4[:] = QT4[:]
            self.Gases[i].DEN[:] = DEN[:]
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
        '''result=[]
        p = Pool()
        for i in range(6):
           if self.Gases[i].NGS != 0:
              result.append( p.apply_async(callGASF(), [self.Gases[i]]))
        p.close()
        p.join()
        for i in range(6):
          if self.Gases[i].NGS != 0:
              self.Gases[i]=result[i].get()'''
        cdef int i
        cdef Gas temp
        for i in range(6):
            callGASF(&self.Gases[i])

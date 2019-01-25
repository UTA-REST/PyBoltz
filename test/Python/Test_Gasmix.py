import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')
from Gasmix import Gasmix

t1 = time.time()
TestObj = Gasmix()
NGS = []
for i in range(6):
    NGS.append(0)
NGS[0]=2
EROOT = []
QT1 = []
QT2 = []
QT3 = []
QT4 = []
DEN = [1 for i in range(4000)]
DENS = 0
NGAS = 1
NSTEP = 4000
NANISO = 2
ESTEP = 1.25e-4
EG = [6.25e-5 + i * (ESTEP) for i in range(4000)]
EFINAL = 0.5
AKT = 2.6037269846599997e-2
ARY = 0
TEMPC = 0
TORR = 0
IPEN = 0
TestObj.setCommons(NGS, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NGAS, NSTEP,
                   NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN)
if __name__ == '__main__':
    TestObj.Run()
print(TestObj.Gases[0].Q[0][0])
print(TestObj.Gases[1].Q[0][0])
print(TestObj.Gases[2].Q[0][0])
print(TestObj.Gases[3].Q[0][0])
print(TestObj.Gases[4].Q[0][0])
print(TestObj.Gases[0].Q[0][0])

print("hi")
t2 = time.time()
print("time:")
print(t2 - t1)


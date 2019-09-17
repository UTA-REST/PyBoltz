from PyBoltz cimport PyBoltz
import math

from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset

import Setups
from Mixer_mert import Mixer_mert
from MixerT_mert import MixerT_mert
import EnergyLimits
import Monte
from Monte import *
cdef class PyBoltz_mert(PyBoltz):

   def GetSimFunctions(self,BFieldMag,BFieldAngle,EnableThermalMotion):
        """
        This function picks which sim functions to use, depending on applied fields and thermal motion.
        """
        ELimFunc       = EnergyLimits.EnergyLimit
        MonteCarloFunc = Monte.MONTE
        if(self.EnableThermalMotion!=0):
            MixerFunc = MixerT_mert
            if BFieldMag == 0:
                ELimFunc       = EnergyLimits.EnergyLimitT
                MonteCarloFunc = Monte.MONTET
            elif BFieldAngle == 0 or BFieldAngle == 180:
                ELimFunc       = EnergyLimits.EnergyLimitT
                MonteCarloFunc = Monte.MONTEAT
            elif BFieldAngle == 90:
                ELimFunc       = EnergyLimits.EnergyLimitBT
                MonteCarloFunc = Monte.MONTEBT
            else:
                ELimFunc       = EnergyLimits.EnergyLimitCT
                MonteCarloFunc = Monte.MONTECT
        else:
            MixerFunc = Mixer_mert
            if BFieldMag == 0:
                ELimFunc       = EnergyLimits.EnergyLimit
                MonteCarloFunc = Monte.MONTE
            elif BFieldAngle == 0 or BFieldAngle == 180:
                ELimFunc       = EnergyLimits.EnergyLimit
                MonteCarloFunc = Monte.MONTEA
            elif BFieldAngle == 90:
                ELimFunc       = EnergyLimits.EnergyLimitB
                MonteCarloFunc = Monte.MONTEB
            else:
                ELimFunc       = EnergyLimits.EnergyLimitC
                MonteCarloFunc = Monte.MONTEC
        return [MixerFunc,ELimFunc,MonteCarloFunc]
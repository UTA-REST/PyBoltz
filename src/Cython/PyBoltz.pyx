import math

from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset

import Setups
import Mixers
import EnergyLimits
import Monte
from Monte import *


cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)

cdef double drand48(double dummy):
    return DRAND48(dummy)

cdef class PyBoltz:
    """

    This is the main object used to start the simulation, as well as store the information of the simulation.
    It has most of the needed arrays, and variables.

    More about PyBoltz:

    `PyBoltz_Documentation <http://cyclo.mit.edu/drift/www/aboutPyBoltz.html/>`_

    .. note::
        If the variable has a "NT" at the end, that variable has the same function as its counterpart without a "NT" at the end.

    """


    def __init__(self):
        '''
        Fill all the variables needed with zeros.This function uses memset as it is fast.
        '''
        memset(self.NullCollisionFreqT, 0, 4000 * 960 * sizeof(double))
        memset(self.EINNT, 0, 960 * sizeof(double))
        memset(self.TotalCollisionFrequencyNT, 0, 4000 * sizeof(double))
        memset(self.IARRYNT, 0, 960 * sizeof(double))
        memset(self.RGASNT, 0, 960 * sizeof(double))
        memset(self.IPNNT, 0, 960 * sizeof(double))
        memset(self.PenningFractionNT, 0, 3*960 * sizeof(double))
        memset(self.WPLNT, 0, 960 * sizeof(double))
        memset(self.MaxCollisionFreqNT, 0, 8 * sizeof(double))
        memset(self.NullCollisionFreqNT, 0, 4000*60 * sizeof(double))
        memset(self.TotalCollisionFrequencyNNT, 0, 4000 * sizeof(double))
        memset(self.SCLENULNT, 0, 60 * sizeof(double))
        memset(self.PSCTNT, 0, 4000* 960 * sizeof(double))
        memset(self.ANGCTNT, 0, 4000* 960 * sizeof(double))
        memset(self.INDEXNT, 0, 960 * sizeof(double))
        memset(self.NC0NT, 0, 960 * sizeof(double))
        memset(self.EC0NT, 0, 960 * sizeof(double))
        memset(self.NG1NT, 0, 960 * sizeof(double))
        memset(self.EG1NT, 0, 960 * sizeof(double))
        memset(self.NG2NT, 0, 960 * sizeof(double))
        memset(self.EG2NT, 0, 960 * sizeof(double))
        memset(self.WKLMNT, 0, 960 * sizeof(double))
        memset(self.EFLNT, 0, 960 * sizeof(double))
        memset(self.DENSY, 0, 4000 * sizeof(double))
        memset(self.CollisionEnergies, 0, 4000 * sizeof(double))
        memset(self.CollisionTimes, 0, 300 * sizeof(double))
        memset(self.ICOLL, 0, 6 * 5 * sizeof(double))
        memset(self.ICOLLNT, 0, 30 * sizeof(double))
        memset(self.ICOLNN, 0, 6 * 10 * sizeof(double))
        memset(self.ICOLN, 0, 6 * 290 * sizeof(double))
        memset(self.ICOLNNNT, 0, 60  * sizeof(double))
        memset(self.ICOLNNT, 0, 960 * sizeof(double))
        memset(self.AMGAS, 0, 6 * sizeof(double))
        memset(self.VTMB, 0, 6 * sizeof(double))
        memset(self.MaxCollisionFreqTotalG, 0, 6 * sizeof(double))
        memset(self.GasIDs, 0, 6 * sizeof(double))
        memset(self.GasFractions, 0, 6 * sizeof(double))
        memset(self.ANN, 0, 6 * sizeof(double))
        memset(self.VANN, 0, 6 * sizeof(double))
        memset(self.QSUM, 0, 4000 * sizeof(double))
        memset(self.QION, 0, 6 * 4000 * sizeof(double))
        memset(self.QIN, 0, 6 * 250 * 4000 * sizeof(double))
        memset(self.E, 0, 4000 * sizeof(double))
        memset(self.EROOT, 0, 4000 * sizeof(double))
        memset(self.QTOT, 0, 4000 * sizeof(double))
        memset(self.QREL, 0, 4000 * sizeof(double))
        memset(self.QINEL, 0, 4000 * sizeof(double))
        memset(self.NIN, 0, 6 * sizeof(double))
        memset(self.CF, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.TotalCollisionFrequency, 0, 6 * 4000 * sizeof(double))
        memset(self.EIN, 0, 6 * 290 * sizeof(double))
        memset(self.IARRY, 0, 6 * 290 * sizeof(double))
        memset(self.RGAS, 0, 6 * 290 * sizeof(double))
        memset(self.IPN, 0, 6 * 290 * sizeof(double))
        memset(self.WPL, 0, 6 * 290 * sizeof(double))
        memset(self.IPLAST, 0, 6 * sizeof(double))
        memset(self.ISIZE, 0, 6 * sizeof(double))
        memset(self.PenningFraction, 0, 6 * 290 * 3 * sizeof(double))
        memset(self.MaxCollisionFreq, 0, 6 * sizeof(double))
        memset(self.NullCollisionFreq, 0, 6 * 10 * 4000 * sizeof(double))
        memset(self.TotalCollisionFrequencyN, 0, 6 * 4000 * sizeof(double))
        memset(self.SCLENUL, 0, 6 * 10 * sizeof(double))
        memset(self.NPLAST, 0, 6 * sizeof(double))
        memset(self.PSCT, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.ANGCT, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.INDEX, 0, 6 * 290 * sizeof(double))
        memset(self.FCION, 0, 4000 * sizeof(double))
        memset(self.FCATT, 0, 4000 * sizeof(double))
        memset(self.NC0, 0, 6 * 290 * sizeof(double))
        memset(self.EC0, 0, 6 * 290 * sizeof(double))
        memset(self.NG1, 0, 6 * 290 * sizeof(double))
        memset(self.EG1, 0, 6 * 290 * sizeof(double))
        memset(self.NG2, 0, 6 * 290 * sizeof(double))
        memset(self.EG2, 0, 6 * 290 * sizeof(double))
        memset(self.WKLM, 0, 6 * 290 * sizeof(double))
        memset(self.EFL, 0, 6 * 290 * sizeof(double))
        memset(self.QEL, 0, 4000 * sizeof(double))
        memset(self.QSATT, 0, 4000 * sizeof(double))
        memset(self.RNMX, 0, 9 * sizeof(double))
        memset(self.ES, 0, 4000 * sizeof(double))

        # Input parameters / settings
        self.EnableThermalMotion = 0.0
        self.MaxNumberOfCollisions = 0.0
        self.BFieldAngle = 0.0
        self.BFieldMag = 0.0
        self.NumberOfGases = 0
        self.WhichAngularModel = 2
        self.TemperatureCentigrade = 0.0
        self.PressureTorr = 0.0
        self.EnablePenning = 0
        self.EField = 0.0
        
        # Calculated Constants 
        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.PIR2 = 0.0
        self.RhydbergConst = 0.0
        self.EFieldOverBField = 0.0
        self.AngularSpeedOfRotation = 0.0
        self.ThermalEnergy = 0.0
        self.MaxCollisionTime = 100.0
        self.SmallNumber = 1e-20
        self.AngleFromX = 0.0
        self.PresTempCor = 0.0

        # Parameters governing decorrelation length
        self.Decor_Colls = 0      # prev NCOLM
        self.Decor_Step = 0     # prev NCORLN 
        self.Decor_LookBacks = 0  # prev NCORST

        # Dynamically set
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.EnergySteps = 0
        self.AnisotropicDetected = 0
        self.FinalElectronEnergy = 0.0
        self.ElectronEnergyStep = 0
        self.InitialElectronEnergy = 0.0
        self.AngleFromZ = 0.0
        self.TimeSum = 0.0
        self.MaxCollisionFreqTotal = 0.0

        
        # Outputs
        self.MeanElectronEnergyError = 0.0
        self.MeanElectronEnergy = 0.0
        self.VelocityX = 0.0
        self.VelocityY = 0.0
        self.VelocityZ = 0.0
        self.VelocityErrorX = 0.0
        self.VelocityErrorY = 0.0
        self.VelocityErrorZ = 0.0
        self.AttachmentRate = 0.0
        self.IonisationRate = 0.0
        self.IonisationRateError = 0.0
        self.AttachmentRateError = 0.0
        self.LongitudinalDiffusion = 0.0
        self.LongitudinalDiffusionError = 0.0
        self.TransverseDiffusion = 0.0
        self.TransverseDiffusionError = 0.0
        self.LongitudinalDiffusion1 = 0.0
        self.LongitudinalDiffusion1Error = 0.0
        self.TransverseDiffusion1 = 0.0
        self.TransverseDiffusion1Error = 0.0
        self.DiffusionX = 0.0
        self.DiffusionY = 0.0
        self.DiffusionYZ = 0.0
        self.DiffusionYZ = 0.0
        self.DiffusionXY = 0.0
        self.DiffusionXZ = 0.0
        self.ErrorDiffusionX = 0.0
        self.ErrorDiffusionY = 0.0
        self.ErrorDiffusionZ = 0.0
        self.ErrorDiffusionYZ = 0.0
        self.ErrorDiffusionXY = 0.0
        self.ErrorDiffusionXZ = 0.0
        self.FakeIonizations = 0
        self.FAKEI = 0.0
        self.RandomSeed = 0.666
        self.ConsoleOutputFlag = 1
        self.MeanCollisionTime = 0.0
        self.ReducedIonization=0.0
        self.ReducedAttachment=0.0

    def end(self):
        """
        This function is used to convert some of the output values into different units.
        """
        cdef double DUM[6]
        if self.VelocityZ != 0:
            self.TransverseDiffusion1 = sqrt(2.0 * self.TransverseDiffusion / self.VelocityZ) * 10000.0
            self.TransverseDiffusion1Error = math.sqrt(self.TransverseDiffusionError ** 2 + self.VelocityErrorZ ** 2)/2.0

            self.LongitudinalDiffusion1 = sqrt(2.0 * self.LongitudinalDiffusion / self.VelocityZ) * 10000.0
            self.LongitudinalDiffusion1Error = sqrt(self.LongitudinalDiffusionError ** 2 + self.VelocityErrorZ ** 2)/2.0

            self.VelocityZ *=1e-5
            self.VelocityY *=1e-5
            self.VelocityX *=1e-5

            self.ReducedIonization = self.IonisationRate * 760 * self.TemperatureKelvin / (self.PressureTorr * 293.15)
            self.ReducedAttachment = self.AttachmentRate * 760 * self.TemperatureKelvin / (self.PressureTorr * 293.15)


   
    def GetSimFunctions(self,BFieldMag,BFieldAngle,EnableThermalMotion):
        """
        This function picks which sim functions to use, depending on applied fields and thermal motion.
        """
        ELimFunc       = EnergyLimits.EnergyLimit
        MonteCarloFunc = Monte.MONTE
        if(self.EnableThermalMotion!=0):
            MixerFunc = Mixers.MixerT
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
            MixerFunc = Mixers.Mixer
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

    def Start(self):
        """
        This is the main function that starts the magboltz simulation/calculation.

        The simulation starts by calculating the momentum cross section values for the requested gas mixture. If FinalElectronEnergy
        is equal to 0.0 it will then keep calling the EnergyLimit functions and the Mixer functions to find the electron
        Integration limit.

        Finally PyBoltz calls the Monte carlo functions, which is where the main simulation happens. The outputs are stored
        in the the parent object of this function.

        For more info on the main output variables check the git repository readme:
        `PyBoltz repository <https://github.com/UTA-REST/MAGBOLTZ-py/>`_
        """

        ELimNotYetFixed=1

        cdef double ReducedField
        cdef int i = 0

        # Get the appropriate set of simulation functions given configuration keys
        MixerFunc, ELimFunc, MonteCarloFunc = self.GetSimFunctions(self.BFieldMag,self.BFieldAngle,self.EnableThermalMotion)

        # Set up the simulation
        Setups.Setup(self)

        # Find the electron upper energy limit
        if self.FinalElectronEnergy == 0.0:
            # Given no specified upper energy limit, find it iteratively
            self.FinalElectronEnergy = 0.5
            ReducedField = self.EField * (self.TemperatureCentigrade + 273.15) / (self.PressureTorr * 293.15)
            if ReducedField > 15:
                self.FinalElectronEnergy = 8.0
            self.InitialElectronEnergy = self.FinalElectronEnergy / 50.0
            while ELimNotYetFixed == 1:
                MixerFunc(self)
                ELimNotYetFixed = ELimFunc(self)

                if ELimNotYetFixed == 1:
                    self.FinalElectronEnergy = self.FinalElectronEnergy * math.sqrt(2)
                    self.InitialElectronEnergy = self.FinalElectronEnergy / 50
        else:
            # Given a specified upper energy limit, use it
            MixerFunc(self)
        
        if self.ConsoleOutputFlag : print("Calculated the final energy = " + str(self.FinalElectronEnergy))

        # Run the simulation
        MonteCarloFunc.run(self)

        # Closeout and end
        self.end()
        return
             

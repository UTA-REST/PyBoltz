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
        memset(self.CFNT, 0, 4000 * 960 * sizeof(double))
        memset(self.EINNT, 0, 960 * sizeof(double))
        memset(self.TCFNT, 0, 4000 * sizeof(double))
        memset(self.IARRYNT, 0, 960 * sizeof(double))
        memset(self.RGASNT, 0, 960 * sizeof(double))
        memset(self.IPNNT, 0, 960 * sizeof(double))
        memset(self.PENFRANT, 0, 3*960 * sizeof(double))
        memset(self.WPLNT, 0, 960 * sizeof(double))
        memset(self.TCFMAXNT, 0, 8 * sizeof(double))
        memset(self.CFNNT, 0, 4000*60 * sizeof(double))
        memset(self.TCFNNT, 0, 4000 * sizeof(double))
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
        memset(self.ETPL, 0, 8 * sizeof(double))
        memset(self.XTPL, 0, 8 * sizeof(double))
        memset(self.YTPL, 0, 8 * sizeof(double))
        memset(self.ZTPL, 0, 8 * sizeof(double))
        memset(self.YZTPL, 0, 8 * sizeof(double))
        memset(self.XZTPL, 0, 8 * sizeof(double))
        memset(self.XYTPL, 0, 8 * sizeof(double))
        memset(self.VYTPL, 0, 8 * sizeof(double))
        memset(self.VXTPL, 0, 8 * sizeof(double))
        memset(self.TTPL, 0, 8 * sizeof(double))
        memset(self.XXTPL, 0, 8 * sizeof(double))
        memset(self.YYTPL, 0, 8 * sizeof(double))
        memset(self.ZZTPL, 0, 8 * sizeof(double))
        memset(self.VZTPL, 0, 8 * sizeof(double))
        memset(self.NETPL, 0, 8 * sizeof(double))
        memset(self.NESST, 0, 9 * sizeof(double))
        memset(self.DENSY, 0, 4000 * sizeof(double))
        memset(self.SPEC, 0, 4000 * sizeof(double))
        memset(self.TIME, 0, 300 * sizeof(double))
        memset(self.ICOLL, 0, 6 * 5 * sizeof(double))
        memset(self.ICOLLNT, 0, 30 * sizeof(double))
        memset(self.ICOLNN, 0, 6 * 10 * sizeof(double))
        memset(self.ICOLN, 0, 6 * 290 * sizeof(double))
        memset(self.ICOLNNNT, 0, 60  * sizeof(double))
        memset(self.ICOLNNT, 0, 960 * sizeof(double))
        memset(self.AMGAS, 0, 6 * sizeof(double))
        memset(self.VTMB, 0, 6 * sizeof(double))
        memset(self.TCFMXG, 0, 6 * sizeof(double))
        memset(self.NumberOfGasesN, 0, 6 * sizeof(double))
        memset(self.FRAC, 0, 6 * sizeof(double))
        memset(self.ANN, 0, 6 * sizeof(double))
        memset(self.VANN, 0, 6 * sizeof(double))
        memset(self.RI, 0, 8 * sizeof(double))
        memset(self.EPT, 0, 8 * sizeof(double))
        memset(self.VZPT, 0, 8 * sizeof(double))
        memset(self.TTEST, 0, 8 * sizeof(double))
        memset(self.XS, 0, 2000 * sizeof(double))
        memset(self.YS, 0, 2000 * sizeof(double))
        memset(self.ZS, 0, 2000 * sizeof(double))
        memset(self.TS, 0, 2000 * sizeof(double))
        memset(self.DCX, 0, 2000 * sizeof(double))
        memset(self.DCY, 0, 2000 * sizeof(double))
        memset(self.DCZ, 0, 2000 * sizeof(double))
        memset(self.IPL, 0, 2000 * sizeof(double))
        memset(self.ESPL, 0, 8 * sizeof(double))
        memset(self.XSPL, 0, 8 * sizeof(double))
        memset(self.TMSPL, 0, 8 * sizeof(double))
        memset(self.TTMSPL, 0, 8 * sizeof(double))
        memset(self.RSPL, 0, 8 * sizeof(double))
        memset(self.RRSPL, 0, 8 * sizeof(double))
        memset(self.RRSPM, 0, 8 * sizeof(double))
        memset(self.YSPL, 0, 8 * sizeof(double))
        memset(self.ZSPL, 0, 8 * sizeof(double))
        memset(self.TSPL, 0, 8 * sizeof(double))
        memset(self.XXSPL, 0, 8 * sizeof(double))
        memset(self.YYSPL, 0, 8 * sizeof(double))
        memset(self.ZZSPL, 0, 8 * sizeof(double))
        memset(self.VZSPL, 0, 8 * sizeof(double))
        memset(self.TSSUM, 0, 8 * sizeof(double))
        memset(self.TSSUM2, 0, 8 * sizeof(double))
        memset(self.QSUM, 0, 4000 * sizeof(double))
        memset(self.QION, 0, 6 * 4000 * sizeof(double))
        memset(self.QIN, 0, 6 * 250 * 4000 * sizeof(double))
        memset(self.E, 0, 4000 * sizeof(double))
        memset(self.EROOT, 0, 4000 * sizeof(double))
        memset(self.QTOT, 0, 4000 * sizeof(double))
        memset(self.QREL, 0, 4000 * sizeof(double))
        memset(self.QINEL, 0, 4000 * sizeof(double))
        memset(self.NIN, 0, 6 * sizeof(double))
        memset(self.LION, 0, 6 * sizeof(double))
        memset(self.LIN, 0, 6 * 250 * sizeof(double))
        memset(self.ALION, 0, 6 * sizeof(double))
        memset(self.ALIN, 0, 6 * 250 * sizeof(double))
        memset(self.CF, 0, 6 * 290 * 4000 * sizeof(double))
        memset(self.TCF, 0, 6 * 4000 * sizeof(double))
        memset(self.EIN, 0, 6 * 290 * sizeof(double))
        memset(self.IARRY, 0, 6 * 290 * sizeof(double))
        memset(self.RGAS, 0, 6 * 290 * sizeof(double))
        memset(self.IPN, 0, 6 * 290 * sizeof(double))
        memset(self.WPL, 0, 6 * 290 * sizeof(double))
        memset(self.IPLAST, 0, 6 * sizeof(double))
        memset(self.ISIZE, 0, 6 * sizeof(double))
        memset(self.PENFRA, 0, 6 * 290 * 3 * sizeof(double))
        memset(self.TCFMAX, 0, 6 * sizeof(double))
        memset(self.CFN, 0, 6 * 10 * 4000 * sizeof(double))
        memset(self.TCFN, 0, 6 * 4000 * sizeof(double))
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
        memset(self.XSS, 0, 2000 * sizeof(double))
        memset(self.YSS, 0, 2000 * sizeof(double))
        memset(self.ZSS, 0, 2000 * sizeof(double))
        memset(self.TSS, 0, 2000 * sizeof(double))
        memset(self.ESS, 0, 2000 * sizeof(double))
        memset(self.DCXS, 0, 2000 * sizeof(double))
        memset(self.DCYS, 0, 2000 * sizeof(double))
        memset(self.DCZS, 0, 2000 * sizeof(double))
        memset(self.IPLS, 0, 2000 * sizeof(double))
        memset(self.IFAKET, 0, 8 * sizeof(double))
        memset(self.IFAKED, 0, 9 * sizeof(double))
        memset(self.QEL, 0, 4000 * sizeof(double))
        memset(self.QSATT, 0, 4000 * sizeof(double))
        memset(self.RNMX, 0, 9 * sizeof(double))
        memset(self.ES, 0, 4000 * sizeof(double))
        memset(self.ZPLANE, 0, 8 * sizeof(double))
        memset(self.LAST, 0, 6 * sizeof(double))

        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.CONST4 = 0.0
        self.CONST5 = 0.0
        self.NISO = 0
        self.NCOLM = 0
        self.NCORLN = 0
        self.NCORST = 0
        self.NNULL = 0
        self.MaximumCollisionTime = 0.0
        self.MeanElectronEnergyError = 0.0
        self.MeanElectronEnergy = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.LongitudinalDiffusionError = 0.0
        self.TransverseDiffusionError = 0.0
        self.TemperatureKelvin = 0.0
        self.ALPP = 0.0
        self.ATTP = 0.0
        self.SSTMIN = 0.0
        self.VDOUT = 0.0
        self.VDERR = 0.0
        self.WSOUT = 0.0
        self.WSERR = 0.0
        self.DLOUT = 0.0
        self.DLERR = 0.0
        self.NMAXOLD = 0.0
        self.DTOUT = 0.0
        self.DTERR = 0.0
        self.ALPHSST = 0.0
        self.EFieldOverBField = 0.0
        self.AngularSpeedOfRotation = 0.0
        self.PIR2 = 0.0
        self.BFieldAngle = 0.0
        self.BFieldMag = 0.0
        self.NumberOfGases = 0
        self.NSTEP = 0
        self.NANISO = 2
        self.FinalElectronEnergy = 0.0
        self.ElectronEnergyStep = 0
        self.ThermalEnergy = 0.0
        self.ARY = 0.0
        self.TemperatureCentigrade = 0.0
        self.PressureTorr = 0.0
        self.EnablePenning = 0
        self.NSCALE = 0
        self.MaxCollisionTime = 100.0
        self.SmallNumber = 0.0
        self.InitialElectronEnergy = 0.0
        self.AngleFromZ = 0.0
        self.AngleFromX = 0.0
        self.EField = 0.0
        self.MaxNumberOfCollisions = 0.0
        self.IonisationRate = 0.0
        self.TCFMX = 0.0
        self.EnableThermalMotion = 0.0
        self.ITMAX = 0
        self.CORR = 0.0
        self.ATTOINT = 0.0
        self.ATTERT = 0.0
        self.AIOERT = 0.0
        self.ALPHERR = 0.0
        self.ATTSST = 0.0

        self.ATTERR = 0.0
        self.IZFINAL = 0.0
        self.RALPHA = 0.0
        self.RALPER = 0.0
        self.TODENE = 0.0
        self.RATTOF = 0.0
        self.RATOFER = 0.0
        self.ALPHAST = 0.0
        self.VDST = 0.0
        self.TSTEP = 0.0
        self.ZSTEP = 0.0
        self.TFINAL = 0.0
        self.RATTOFER = 0.0
        self.ZFINAL = 0.0
        self.ITFINAL = 0.0
        self.IPRIM = 0.0
        self.TimeSum = 0.0
        self.ATTOION = 0.0
        self.ATTIOER = 0.0
        self.ATTATER = 0.0
        self.DTOVMB = 0.0
        self.DTMN = 0.0
        self.DFTER1 = 0.0
        self.DLOVMB = 0.0
        self.DLMN = 0.0
        self.DFLER1 = 0.0
        # common output blocks
        self.VelocityX = 0.0
        self.VelocityY = 0.0
        self.VelocityZ = 0.0
        self.VelocityErrorX = 0.0
        self.VelocityErrorY = 0.0
        self.VelocityErrorZ = 0.0
        self.AttachmentRate = 0.0
        self.IonisationRateError = 0.0
        self.AttachmentRateError = 0.0
        self.LongitudinalDiffusion = 0.0
        self.TransverseDiffusion = 0.0
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
        self.IFAKE = 0
        self.FAKEI = 0.0
        self.RSTART = 0.666
        self.ConsoleOutputFlag = 1
        self.MeanCollisionTime = 0.0

    def end(self):
        """
        This function is used to convert some of the output values into different units.
        """
        cdef double DUM[6]
        if self.VelocityZ != 0:
            self.DTOVMB = self.TransverseDiffusion * self.EField / self.VelocityZ
            self.DTMN = sqrt(2.0 * self.TransverseDiffusion / self.VelocityZ) * 10000.0
            self.DFTER1 = math.sqrt(self.TransverseDiffusionError ** 2 + self.VelocityErrorZ ** 2)
            self.DFTER1 = self.DFTER1 / 2.0

            self.DLOVMB = self.LongitudinalDiffusion * self.EField / self.VelocityZ
            self.DLMN = sqrt(2.0 * self.LongitudinalDiffusion / self.VelocityZ) * 10000.0
            self.DFLER1 = sqrt(self.LongitudinalDiffusionError ** 2 + self.VelocityErrorZ ** 2)
            self.DFLER1 = self.DFLER1 / 2.0
            self.VelocityZ *=1e-5
            self.VelocityY *=1e-5
            self.VelocityX *=1e-5


   
    def GetSimFunctions(self,BFieldMag,BFieldAngle,EnableThermalMotion):
        """
        This function picks which sim functions to use, depending on applied fields and thermal motion.
        """
        SetupFunc      = Setups.Setup
        ELimFunc       = EnergyLimits.EnergyLimit
        MonteCarloFunc = Monte.MONTE
        if(self.EnableThermalMotion!=0):
            SetupFunc = Setups.SetupT
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
            SetupFunc = Setups.Setup
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
        return [SetupFunc,MixerFunc,ELimFunc,MonteCarloFunc]  

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

        cdef double EOB
        cdef int i = 0

        # Get the appropriate set of simulation functions given configuration keys
        SetupFunc,MixerFunc, ELimFunc, MonteCarloFunc = self.GetSimFunctions(self.BFieldMag,self.BFieldAngle,self.EnableThermalMotion)

        # Set up the simulation
        SetupFunc(self)

        # Find the electron upper energy limit
        if self.FinalElectronEnergy == 0.0:
            # Given no specified upper energy limit, find it iteratively
            self.FinalElectronEnergy = 0.5
            EOB = self.EField * (self.TemperatureCentigrade + 273.15) / (self.PressureTorr * 293.15)
            if EOB > 15:
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

        # Express outputs in the right units
        self.TemperatureKelvin = 273.15 + self.TemperatureCentigrade
        self.ALPP = self.IonisationRate * 760 * self.TemperatureKelvin / (self.PressureTorr * 293.15)
        self.ATTP = self.AttachmentRate * 760 * self.TemperatureKelvin / (self.PressureTorr * 293.15)
        self.SSTMIN = 40
        self.end()
        return
             

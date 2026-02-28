import math
import random
import numpy as np
from PyGasMix.Gasmix import Gasmix


class Boltz:
    """
    Main object used to start the simulation and store simulation data.
    Pure Python port of PyBoltz/Boltz.pyx.
    """

    def __init__(self):
        # NT arrays (no thermal motion)
        self.CollisionFrequencyNT = np.zeros((4000, 960))
        self.EnergyLevelsNT = np.zeros(960)
        self.TotalCollisionFrequencyNT = np.zeros(4000)
        self.InteractionTypeNT = np.zeros(960)
        self.RGasNT = np.zeros(960)
        self.ElectronNumChangeNT = np.zeros(960)
        self.PenningFractionNT = np.zeros((3, 960))
        self.MaxCollisionFreqNT = np.zeros(8)
        self.NullCollisionFreqNT = np.zeros((4000, 60))
        self.TotalCollisionFrequencyNullNT = np.zeros(4000)
        self.ScaleNullNT = np.zeros(60)
        self.ScatteringParameterNT = np.zeros((4000, 960))
        self.AngleCutNT = np.zeros((4000, 960))
        self.AngularModelNT = np.zeros(960)
        self.NC0NT = np.zeros(960)
        self.EC0NT = np.zeros(960)
        self.CollisionsPerGasPerTypeNT = np.zeros(30)
        self.ICOLNNT = np.zeros(960)
        self.WPLNT = np.zeros(960)
        self.WPL = np.zeros((6, 290))
        self.ICOLNNNT = np.zeros(60)
        self.NumMomCrossSectionPointsNullNT = 0.0
        self.ISIZENT = 0.0
        self.RandomMaxBoltzArray = np.zeros(9)

        # Regular (thermal motion) arrays
        self.CollisionEnergies = np.zeros(4000)
        self.CollisionTimes = np.zeros(300)
        self.CollisionsPerGasPerType = np.zeros((6, 5))
        self.ICOLNN = np.zeros((6, 10))
        self.ICOLN = np.zeros((6, 290))
        self.AMGAS = np.zeros(6)
        self.EC0 = np.zeros((6, 290))
        self.VTMB = np.zeros(6)
        self.MaxCollisionFreqTotalG = np.zeros(6)
        self.GasIDs = np.zeros(6)
        self.GasFractions = np.zeros(6)
        self.MoleculesPerCm3PerGas = np.zeros(6)
        self.VMoleculesPerCm3PerGas = np.zeros(6)
        self.CrossSectionSum = np.zeros(4000)
        self.ElasticCrossSection = np.zeros(4000)
        self.AttachmentSectionSum = np.zeros(4000)
        self.E = np.zeros(4000)
        self.SqrtEnergy = np.zeros(4000)
        self.TotalCrossSection = np.zeros(4000)
        self.RelativeIonMinusAttachCrossSection = np.zeros(4000)
        self.InelasticCrossSection = np.zeros(4000)
        self.FakeIonizationsT = np.zeros(8)
        self.FakeIonizationsD = np.zeros(9)
        self.LAST = np.zeros(6)
        self.N_Inelastic = np.zeros(6)
        self.NumMomCrossSectionPoints = np.zeros(6)
        self.ISIZE = np.zeros(6)
        self.MaxCollisionFreq = np.zeros(6)
        self.NumMomCrossSectionPointsNull = np.zeros(6)
        self.AngularModel = np.zeros((6, 290))
        self.NC0 = np.zeros((6, 290))
        self.EnergyLevels = np.zeros((6, 290))
        self.InteractionType = np.zeros((6, 290))
        self.RGas = np.zeros((6, 290))
        self.ElectronNumChange = np.zeros((6, 290))
        self.IonizationCrossSection = np.zeros((6, 4000))
        self.InelasticCrossSectionPerGas = np.zeros((6, 250, 4000))
        self.LIN = np.zeros((6, 250))
        self.ALIN = np.zeros((6, 250))
        self.CollisionFrequency = np.zeros((6, 4000, 290))
        self.TotalCollisionFrequency = np.zeros((6, 4000))
        self.PenningFraction = np.zeros((6, 3, 290))
        self.NullCollisionFreq = np.zeros((6, 4000, 10))
        self.TotalCollisionFrequencyNull = np.zeros((6, 4000))
        self.ScaleNull = np.zeros((6, 10))
        self.ScatteringParameter = np.zeros((6, 4000, 290))
        self.AngleCut = np.zeros((6, 4000, 290))
        self.IonCollisionFreq = np.zeros(4000)
        self.AttCollisionFreq = np.zeros(4000)

        # Collision type counters
        self.ElasticColli = np.zeros(6, dtype=int)
        self.InelasticColli = np.zeros(6, dtype=int)
        self.AttachmentColli = np.zeros(6, dtype=int)
        self.IonisationColli = np.zeros(6, dtype=int)
        self.SuperElasticColli = np.zeros(6, dtype=int)

        # Townsend plane output arrays
        self.SXPlanes = np.zeros(9)
        self.SYPlanes = np.zeros(9)
        self.SZPlanes = np.zeros(9)
        self.STPlanes = np.zeros(9)
        self.SEPlanes = np.zeros(9)
        self.SX2Planes = np.zeros(9)
        self.SY2Planes = np.zeros(9)
        self.SZ2Planes = np.zeros(9)
        self.SVZPlanes = np.zeros(9)
        self.STSPlanes = np.zeros(9)
        self.STS2Planes = np.zeros(9)
        self.SRPlanes = np.zeros(9)
        self.SR2Planes = np.zeros(9)
        self.TMSPL = np.zeros(9)
        self.TTMSPL = np.zeros(9)
        self.TXPlanes = np.zeros(9)
        self.TYPlanes = np.zeros(9)
        self.TZPlanes = np.zeros(9)
        self.TTPlanes = np.zeros(9)
        self.TEPlanes = np.zeros(9)
        self.TX2Planes = np.zeros(9)
        self.TY2Planes = np.zeros(9)
        self.TZ2Planes = np.zeros(9)
        self.TVZPlanes = np.zeros(9)
        self.NumberOfElectronsPlanes = np.zeros(9)
        self.RealIonisation = np.zeros(9)
        self.EnergyPT = np.zeros(9)
        self.VelocityZPT = np.zeros(9)
        self.AverageTimePT = np.zeros(9)
        self.NumberOfElectronSST = np.zeros(11)
        self.FakeIonizationsT = np.zeros(8)
        self.FakeIonizationsD = np.zeros(9)

        # Input parameters / settings
        self.Enable_Thermal_Motion = 0.0
        self.MaxNumberOfCollisions = 0.0
        self.BField_Angle = 0.0
        self.BField_Mag = 0.0
        self.NumberOfGases = 0
        self.Which_Angular_Model = 2
        self.TemperatureCentigrade = 0.0
        self.Pressure_Torr = 0.0
        self.Enable_Penning = 0
        self.EField = 0.0
        self.Num_Samples = 10
        self.Decor_Colls = 0
        self.Decor_Step = 0
        self.Decor_Lookbacks = 0

        # Calculated constants
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

        # Dynamically set
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.EnergySteps = 0
        self.AnisotropicDetected = 0
        self.Max_Electron_Energy = 0.0
        self.ElectronEnergyStep = 0
        self.InitialElectronEnergy = 0.0
        self.AngleFromZ = 0.0
        self.TimeSum = 0.0
        self.MaxCollisionFreqTotal = 0.0
        self.BFieldMode = 0
        self.NumMomCrossSectionPointsNT = 0.0
        self.MeanCollisionTime = 0.0

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
        self.DiffusionZ = 0.0
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
        self.Random_Seed = 54217137
        self.Console_Output_Flag = 1
        self.AlphaSST = 0.0
        self.AlphaSSTErr = 0.0
        self.AttachmentSST = 0.0
        self.AttachmentSSTErr = 0.0
        self.ReducedIonization = 0.0
        self.ReducedAttachment = 0.0
        self.ReducedIonizationErr = 0.0
        self.ReducedAttachmentErr = 0.0
        self.Steady_State_Threshold = 40.0
        self.TemperatureKelvin = 0.0
        self.MaximumCollisionTime = 0.0
        self.AN = 0.0
        self.VAN = 0.0

        # Steady state / Townsend outputs
        self.VelocitySvzSST = 0.0
        self.VelocitySvzSSTErr = 0.0
        self.WSOUT = 0.0
        self.WSERR = 0.0
        self.LongitudinalDiffSST = 0.0
        self.LongitudinalDiffSSTErr = 0.0
        self.TransverseDiffSST = 0.0
        self.TransverseDiffSSTErr = 0.0
        self.AlphaSST = 0.0
        self.AlphaSSTErr = 0.0
        self.AttachmentSST = 0.0
        self.AttachmentSSTErr = 0.0
        self.AttachmentOverIonisation = 0.0
        self.AttachmentOverIonisationErr = 0.0
        self.AttachmentErr = 0.0
        self.AttachmentOverIonisationPT = 0.0
        self.AttachmentErrPT = 0.0
        self.AttachmentOverIonisationErrPT = 0.0

        # TOF outputs
        self.ReducedAlphaTOF = 0.0
        self.ReducedAlphaTOFErr = 0.0
        self.TOFEnergy = 0.0
        self.TOFEnergyErr = 0.0
        self.VelocityTOFPT = 0.0
        self.VelocityTOFPTErr = 0.0
        self.LongitudinalDiffusionTOF = 0.0
        self.LongitudinalDiffusionTOFErr = 0.0
        self.TransverseDiffusionTOF = 0.0
        self.TransverseDiffusionTOFErr = 0.0
        self.VelocityTOF = 0.0
        self.VelocityTOFErr = 0.0
        self.ReducedAttachmentTOF = 0.0
        self.ReducedAttachmentTOFErr = 0.0

        # Alpha / ATT
        self.ALPHA = 0.0
        self.ATT = 0.0
        self.ALPER = 0.0
        self.ATTER = 0.0
        self.ALPTEST = 0.0

        # Townsend simulation settings
        self.Steady_State_Threshold = 40.0
        self.FakeIonisationsEstimate = 0.0
        self.TimeStep = 0.0
        self.MaxTime = 0.0
        self.NumberOfTimeSteps = 0
        self.NumberOfSpaceSteps = 0
        self.TotalTimePrimary = 0.0
        self.TotalTimeSecondary = 0.0
        self.IPrimary = 0
        self.TotalSpaceZPrimary = 0.0
        self.TotalSpaceZSecondary = 0.0
        self.SpaceStepZ = 0.0
        self.MaxSpaceZ = 0.0

        self.MixObject = Gasmix()

    def reset(self):
        self.TimeSum = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.TotalTimePrimary = 0.0
        self.TotalTimeSecondary = 0.0
        self.TotalSpaceZPrimary = 0.0
        self.TotalSpaceZSecondary = 0.0
        self.CollisionsPerGasPerType[:] = 0.0
        self.ICOLN[:] = 0.0
        self.ICOLNN[:] = 0.0
        self.CollisionEnergies[:] = 0.0
        self.CollisionTimes[:] = 0.0

    def end(self):
        """Convert output values into different units."""
        self.TemperatureKelvin = self.TemperatureCentigrade + 273.15

        if self.VelocityZ != 0:
            self.TransverseDiffusion1 = (
                math.sqrt(2.0 * self.TransverseDiffusion / self.VelocityZ) * 10000.0)
            self.TransverseDiffusion1Error = math.sqrt(
                self.TransverseDiffusionError ** 2 + self.VelocityErrorZ ** 2) / 2.0

            self.LongitudinalDiffusion1 = (
                math.sqrt(2.0 * self.LongitudinalDiffusion / self.VelocityZ) * 10000.0)
            self.LongitudinalDiffusion1Error = math.sqrt(
                self.LongitudinalDiffusionError ** 2 + self.VelocityErrorZ ** 2) / 2.0

            self.VelocityZ *= 1e-5
            self.VelocityY *= 1e-5
            self.VelocityX *= 1e-5

            self.ReducedIonization = self.IonisationRate / (
                760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))
            self.ReducedIonizationErr = self.IonisationRateError / (
                760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))

            self.ReducedAttachment = self.AttachmentRate / (
                760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))
            self.ReducedAttachmentErr = self.AttachmentRateError / (
                760 * self.TemperatureKelvin / (self.Pressure_Torr * 293.15))

        if self.Enable_Thermal_Motion == 1:
            for I in range(6):
                self.ElasticColli[I] = int(self.CollisionsPerGasPerType[I][0])
                self.InelasticColli[I] = int(self.CollisionsPerGasPerType[I][3])
                self.AttachmentColli[I] = int(self.CollisionsPerGasPerType[I][2])
                self.IonisationColli[I] = int(self.CollisionsPerGasPerType[I][1])
                self.SuperElasticColli[I] = int(self.CollisionsPerGasPerType[I][4])
        else:
            for I in range(6):
                self.ElasticColli[I] = int(self.CollisionsPerGasPerTypeNT[0])
                self.InelasticColli[I] = int(self.CollisionsPerGasPerTypeNT[3])
                self.AttachmentColli[I] = int(self.CollisionsPerGasPerTypeNT[2])
                self.IonisationColli[I] = int(self.CollisionsPerGasPerTypeNT[1])
                self.SuperElasticColli[I] = int(self.CollisionsPerGasPerTypeNT[4])

    def GetSimFunctions(self, BFieldMag, BFieldAngle, EnableThermalMotion):
        """
        Pick which simulation functions to use depending on fields and thermal motion.
        Only the no-B-field and no-thermal variants are fully ported.
        Set self.use_numba = True before calling Start() to use the Numba JIT backend.
        """
        from PyBoltzPy import Mixers, EnergyLimits
        if getattr(self, 'use_numba', False):
            from PyBoltzPy.MonteFuncs import MONTE_numba as MONTE
        else:
            from PyBoltzPy.MonteFuncs import MONTE

        if EnableThermalMotion != 0:
            raise NotImplementedError("Thermal motion not supported in PyBoltzPy")

        MixerFunc = Mixers.Mixer
        if BFieldMag == 0:
            self.BFieldMode = 1
            ELimFunc = EnergyLimits.EnergyLimit
            MonteCarloFunc = MONTE
        elif BFieldAngle == 0 or BFieldAngle == 180:
            self.BFieldMode = 2
            ELimFunc = EnergyLimits.EnergyLimit
            MonteCarloFunc = MONTE
        else:
            raise NotImplementedError("B-field at arbitrary angle not supported in PyBoltzPy")

        TownsendFunc = None
        return [MixerFunc, ELimFunc, MonteCarloFunc, TownsendFunc]

    def SetExtraParameters(self, params):
        self.MixObject.ExtraParameters = params

    def Start(self):
        """
        Main function that starts the simulation.
        """
        random.seed(self.Random_Seed)
        ELimNotYetFixed = 1

        MixerFunc, ELimFunc, MonteCarloFunc, TownsendFunc = self.GetSimFunctions(
            self.BField_Mag, self.BField_Angle, self.Enable_Thermal_Motion)

        from PyBoltzPy import Setups
        Setups.Setup(self)

        if self.Max_Electron_Energy == 0.0:
            self.Max_Electron_Energy = 0.5
            ReducedField = self.EField * (self.TemperatureCentigrade + 273.15) / (
                self.Pressure_Torr * 293.15)
            if ReducedField > 15:
                self.Max_Electron_Energy = 8.0
            self.InitialElectronEnergy = self.Max_Electron_Energy / 50.0
            while ELimNotYetFixed == 1:
                MixerFunc(self)
                ELimNotYetFixed = ELimFunc(self)
                if ELimNotYetFixed == 1:
                    self.Max_Electron_Energy = self.Max_Electron_Energy * math.sqrt(2)
                    self.InitialElectronEnergy = self.Max_Electron_Energy / 50
        else:
            MixerFunc(self)

        print("")
        if self.Console_Output_Flag:
            print("Calculated the final energy = " + str(self.Max_Electron_Energy))

        ELimNotYetFixed = 1
        while ELimNotYetFixed == 1:
            try:
                MonteCarloFunc.run(self)
            except ValueError as err:
                if "ENERGY INTEGRATION RANGE" in str(err).upper():
                    print("Electron max energy " + str(round(self.Max_Electron_Energy, 2)) +
                          " exceeded, increasing to " +
                          str(round(self.Max_Electron_Energy * math.sqrt(2), 2)) +
                          " and trying again")
                    self.Max_Electron_Energy = self.Max_Electron_Energy * math.sqrt(2)
                    self.reset()
                    MixerFunc(self)
                else:
                    raise err
            else:
                ELimNotYetFixed = 0

        self.end()

        if abs(self.ReducedIonization - self.ReducedAttachment) >= self.Steady_State_Threshold:
            if self.ReducedIonization == 0:
                print("Steady State Threshold crossed. Not running SST as ionisation rate is zero.")
                return
            if self.Console_Output_Flag:
                print("\n**Crossed the set Steady state simulation threshold = {}\n".format(
                    self.Steady_State_Threshold))
            if TownsendFunc is not None:
                TownsendFunc.run(self)

        return

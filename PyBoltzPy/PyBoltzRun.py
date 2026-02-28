import numpy as np
from PyBoltzPy.Boltz import Boltz


class PBRes:
    err = 0
    val = 0

    def __init__(self, a=0, b=0):
        self.val = a
        self.err = b * a / 100.0

    def __str__(self):
        return "value: " + str(self.val) + "; error: " + str(self.err)


class PyBoltzRun:
    PBSettings = {
        'Gases': ['NEON', 'CO2'],
        'Fractions': [90, 10],
        'Max_collisions': 4e7,
        'EField_Vcm': 100,
        'Max_electron_energy': 0,
        'Temperature_C': 23,
        'Pressure_Torr': 750.062,
        'BField_Tesla': 0,
        'BField_angle': 0,
        'Angular_dist_model': 1,
        'Enable_penning': 0,
        'Enable_thermal_motion': 1,
        'ConsoleOutputFlag': 0,
        'Decor_Colls': 0,
        'Decor_LookBacks': 0,
        'Decor_Step': 0,
        'NumSamples': 10,
    }

    Gases = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON',
             'CH4', 'ETHANE', 'PROPANE', 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN',
             'NITROGEN', np.nan, np.nan, np.nan, np.nan, 'HYDROGEN', 'DEUTERIUM',
             np.nan, np.nan, 'DME']

    def ListGases(self):
        for g in self.Gases:
            if type(g) == str:
                print(g, self.GasCode(g))

    def GasCode(self, GasName):
        return self.Gases.index(GasName)

    def GasName(self, Code):
        return self.Gases[Code]

    def ProcessInputs(self, MBObject, Inputs):
        for key in self.PBSettings.keys():
            if key not in Inputs:
                print("Input " + str(key) + " not set, using default " + str(self.PBSettings[key]))
                Inputs[key] = self.PBSettings[key]
        if len(Inputs['Gases']) != len(Inputs['Fractions']):
            print("Error! Gas and fraction lists not the same length")
            return False
        if len(Inputs['Gases']) > 6:
            print("Error! Too many gases. Max is 6.")
            return False
        if abs(sum(Inputs['Fractions']) - 100) > 1e-6:
            print("Error! Gas fractions don't add to 100%")
            return False
        MBObject.EField = Inputs['EField_Vcm']
        MBObject.NumberOfGases = len(Inputs['Gases'])
        GasIDs = np.zeros(6, dtype='int')
        GasFractions = np.zeros(6, dtype='float')
        for i in range(len(Inputs['Gases'])):
            GasIDs[i] = self.GasCode(Inputs['Gases'][i])
            GasFractions[i] = Inputs['Fractions'][i]
        MBObject.GasIDs = GasIDs
        MBObject.GasFractions = GasFractions
        MBObject.MaxNumberOfCollisions = Inputs['Max_collisions']
        MBObject.Enable_Penning = Inputs['Enable_penning']
        MBObject.Enable_Thermal_Motion = Inputs['Enable_thermal_motion']
        MBObject.Max_Electron_Energy = Inputs['Max_electron_energy']
        MBObject.TemperatureCentigrade = Inputs['Temperature_C']
        MBObject.Pressure_Torr = Inputs['Pressure_Torr']
        MBObject.BField_Mag = Inputs['BField_Tesla']
        MBObject.BField_Angle = Inputs['BField_angle']
        MBObject.Console_Output_Flag = Inputs['ConsoleOutputFlag']
        MBObject.Which_Angular_Model = Inputs['Angular_dist_model']
        MBObject.Decor_Colls = Inputs['Decor_Colls']
        MBObject.Decor_Lookbacks = Inputs['Decor_LookBacks']
        MBObject.Decor_Step = Inputs['Decor_Step']
        MBObject.Num_Samples = Inputs['NumSamples']
        return True

    def ProcessOutputs(self, MBObject):
        Outputs = {}
        Outputs['Drift_vel'] = PBRes(
            np.array([MBObject.VelocityX, MBObject.VelocityY, MBObject.VelocityZ]),
            np.array([MBObject.VelocityErrorX, MBObject.VelocityErrorY, MBObject.VelocityErrorZ]))
        Outputs['DT'] = PBRes(MBObject.TransverseDiffusion, MBObject.TransverseDiffusionError)
        Outputs['DL'] = PBRes(MBObject.LongitudinalDiffusion, MBObject.LongitudinalDiffusionError)
        Outputs['DT1'] = PBRes(MBObject.TransverseDiffusion1, MBObject.TransverseDiffusion1Error)
        Outputs['DL1'] = PBRes(MBObject.LongitudinalDiffusion1, MBObject.LongitudinalDiffusion1Error)
        Outputs['MeanEnergy'] = PBRes(MBObject.MeanElectronEnergy, MBObject.MeanElectronEnergyError)
        DTensor = [
            [MBObject.DiffusionX, MBObject.DiffusionXY, MBObject.DiffusionXZ],
            [MBObject.DiffusionXY, MBObject.DiffusionY, MBObject.DiffusionYZ],
            [MBObject.DiffusionXZ, MBObject.DiffusionYZ, MBObject.DiffusionZ],
        ]
        DTensorErr = [
            [MBObject.ErrorDiffusionX, MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionXZ],
            [MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionY, MBObject.ErrorDiffusionYZ],
            [MBObject.ErrorDiffusionXZ, MBObject.ErrorDiffusionYZ, MBObject.ErrorDiffusionZ],
        ]
        Outputs['DTensor'] = PBRes(np.array(DTensor), np.array(DTensorErr))
        Outputs['ReducedIonization'] = PBRes(MBObject.ReducedIonization, MBObject.ReducedIonizationErr)
        Outputs['ReducedAttachment'] = PBRes(MBObject.ReducedAttachment, MBObject.ReducedAttachmentErr)
        Outputs['AlphaSST'] = PBRes(MBObject.AlphaSST, MBObject.AlphaSSTErr)
        return Outputs

    def Run(self, MySettings):
        MBObject = Boltz()
        MBObject.use_numba = MySettings.pop('use_numba', False)
        Status = self.ProcessInputs(MBObject, MySettings)
        if Status:
            MBObject.Start()
        Outputs = self.ProcessOutputs(MBObject)
        return Outputs

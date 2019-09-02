import numpy as np
from PyBoltz import PyBoltz

#Data type to store results with uncertainties
class PBRes:
    err=0
    val=0
    def __init__(self, a=0,b=0):
        self.val=a
        self.err=b
    def __str__(self):
        return "value: " + str(self.val)+ "; error: "+ str(self.err)

# PyBoltzRun helper class
class PyBoltzRun:
    
    #Default settings for running PyBolz
    PBSettings   ={'Gases'                 :['XENON','HELIUM4'],
                   'Fractions'             :[90,10],
                   'Max_collisions'        :1,
                   'EField_Vcm'            :100, 
                   'Max_electron_energy'   :0,
                   'Temperature_C'         :23,
                   'Pressure_Torr'         :750.062,
                   'BField_Tesla'          :0,
                   'BField_angle'          :0,
                   'Angular_dist_model'    :1,
                   'Enable_penning'        :0,
                   'Enable_thermal_motion' :1,
                   'ConsoleOutputFlag'                    :0}
    # Available Gases
    Gases = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON', 'CH4', 'ETHANE', 'PROPANE'
         , 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN', 'NITROGEN', np.nan, np.nan, np.nan, np.nan
         , 'HYDROGEN', 'DEUTERIUM', np.nan, np.nan, 'DME']

    # Print list of available gases
    def ListGases(self):
        for g in self.Gases:
            if(type(g)==str):
                print(g,self.GasCode(g))

    # Convert GasName into MagBoltz GasCode            
    def GasCode(self,GasName):
        return self.Gases.index(GasName)

    # Convert MagBoltz GasCode in GasName
    def GasName(self,Code):
        return Gases[Code]
    
    # Load Input Dictionary into MagBoltz object
    def ProcessInputs(self,MBObject, Inputs):
        for key in self.PBSettings.keys():
            if not key in Inputs:
                print("Input "+str(key)+ " not set, using default "+str(self.PBSettings[key]))
                Inputs[key]=self.PBSettings[key]
        if(len(Inputs['Gases'])!=len(Inputs['Fractions'])):
            print("Error! Gas and fraction lists not the same length")
            return False
        if(len(Inputs['Gases'])>6):
            print("Error! Too many gases. Max is 6.")
            return False
        if(abs(sum(Inputs['Fractions'])-100)>1e-6):
            print("Error! Gas fractions don't add to 100%")
            return False
        MBObject.EField=Inputs['EField_Vcm']
        MBObject.NumberOfGases=len(Inputs['Gases'])
        NumberOfGasesN=np.zeros(6,dtype='int')
        FRAC=np.zeros(6,dtype='float')
        for i in range(len(Inputs['Gases'])):
            NumberOfGasesN[i] = self.GasCode(Inputs['Gases'][i])
            FRAC[i]  = Inputs['Fractions'][i]
        MBObject.NumberOfGasesN  = NumberOfGasesN
        MBObject.FRAC   = FRAC
        MBObject.MaxNumberOfCollisions   = Inputs['Max_collisions']
        MBObject.EnablePenning   = Inputs['Enable_penning']
        MBObject.EnableThermalMotion  = Inputs['Enable_thermal_motion']
        MBObject.FinalElectronEnergy = Inputs['Max_electron_energy']
        MBObject.TemperatureCentigrade  = Inputs['Temperature_C']
        MBObject.PressureTorr   = Inputs['Pressure_Torr']
        MBObject.BFieldMag   = Inputs['BField_Tesla']
        MBObject.BFieldAngle = Inputs['BField_angle']
        MBObject.ConsoleOutputFlag     = Inputs['ConsoleOutputFlag']
        MBObject.WhichAngularModel = Inputs['Angular_dist_model']
        return True

    # Extract Outputs into Output Dictionary
    def ProcessOutputs(self, MBObject):
        Outputs={}
        Outputs['Drift_vel']      = PBRes([MBObject.VelocityX,MBObject.VelocityY,MBObject.VelocityZ],[MBObject.VelocityErrorX,MBObject.VelocityErrorY,MBObject.VelocityErrorZ])
        Outputs['DT']             = PBRes(MBObject.TransverseDiffusion, MBObject.TransverseDiffusionError)
        Outputs['DL']             = PBRes(MBObject.LongitudinalDiffusion, MBObject.LongitudinalDiffusionError)
        Outputs['DT1']            = PBRes(MBObject.TransverseDiffusion1,MBObject.TransverseDiffusion1Error)
        Outputs['DL1']            = PBRes(MBObject.LongitudinalDiffusion1,MBObject.LongitudinalDiffusion1Error)

        Outputs['MeanEnergy']    = PBRes(MBObject.MeanElectronEnergy,MBObject.MeanElectronEnergyError)
        DTensor     = [[MBObject.DiffusionX, MBObject.DiffusionXY, MBObject.DiffusionXZ],
                       [MBObject.DiffusionXY, MBObject.DiffusionY, MBObject.DiffusionYZ],
                       [MBObject.DiffusionXZ, MBObject.DiffusionYZ, MBObject.DiffusionZ]]
        DTensorErr  = [[MBObject.ErrorDiffusionX, MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionXZ],
                       [MBObject.ErrorDiffusionXY, MBObject.ErrorDiffusionY, MBObject.ErrorDiffusionYZ],
                       [MBObject.ErrorDiffusionXZ, MBObject.ErrorDiffusionYZ, MBObject.ErrorDiffusionZ]]
        Outputs['DTensor']       = PBRes(DTensor, DTensorErr)

        return Outputs

    # Run PyBoltz with chosen settings
    def Run(self,MySettings):
        MBObject = PyBoltz()
        Status=self.ProcessInputs(MBObject,MySettings)
        if(Status):
            MBObject.Start()
        Outputs = self.ProcessOutputs(MBObject)
        return Outputs

import numpy as np
from PyBoltz import PyBoltz

#Data type to store results with uncertainties
# Initialized with val and % error
class PBRes:
    '''Class used to store the output of PyBoltz with a val and error.'''
    err=0
    '''The error variable.'''
    val=0
    '''The value variable.'''
    def __init__(self, a=0,b=0):
        self.val=a
        self.err=b*a/100.
    def __str__(self):
        return "value: " + str(self.val)+ "; error: "+ str(self.err)

# PyBoltzRun helper class
class PyBoltzRun:
    '''Class used to be the wrapper object of PyBoltz.'''
    #Default settings for running PyBolz
    PBSettings   ={'Gases'                 :['NEON','CO2'],
                   'Fractions'             :[90,10],
                   'Max_collisions'        :4e7,
                   'EField_Vcm'            :100, 
                   'Max_electron_energy'   :0,
                   'Temperature_C'         :23,
                   'Pressure_Torr'         :750.062,
                   'BField_Tesla'          :0,
                   'BField_angle'          :0,
                   'Angular_dist_model'    :1,
                   'Enable_penning'        :0,
                   'Enable_thermal_motion' :1,
                   'ConsoleOutputFlag'     :0,
                   'Decor_Colls'           :0,
                   'Decor_LookBacks'       :0,
                   'Decor_Step'            :0,
                   'NumSamples'            :10}
    '''Dictionary used to store the inputs/settings for the PyBoltz simulation.'''
    # Available Gases
    Gases = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON', 'CH4', 'ETHANE', 'PROPANE'
         , 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN', 'NITROGEN', np.nan, np.nan, np.nan, np.nan
         , 'HYDROGEN', 'DEUTERIUM', np.nan, np.nan, 'DME']
    '''Array of gases in PyBoltz.'''

    # Print list of available gases
    def ListGases(self):
        '''Function used to print all the gases names in PyBoltz.'''
        for g in self.Gases:
            if(type(g)==str):
                print(g,self.GasCode(g))

    # Convert GasName into MagBoltz GasCode            
    def GasCode(self,GasName):
        '''Function used to get the ID of the gas. The ID is simply the index of that gas in that array.'''
        return self.Gases.index(GasName)

    # Convert MagBoltz GasCode in GasName
    def GasName(self,Code):
        '''Function used to return the name of the Gas ID given.'''
        return Gases[Code]
    
    # Load Input Dictionary into MagBoltz object
    def ProcessInputs(self,MBObject, Inputs):
        '''Function used to setup the PyBoltz Object with the given inputs in the PBSettings dictionary.'''
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
        GasIDs=np.zeros(6,dtype='int')
        GasFractions=np.zeros(6,dtype='float')
        for i in range(len(Inputs['Gases'])):
            GasIDs[i] = self.GasCode(Inputs['Gases'][i])
            GasFractions[i]  = Inputs['Fractions'][i]
        MBObject.GasIDs  = GasIDs
        MBObject.GasFractions   = GasFractions
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
        MBObject.Decor_Colls = Inputs['Decor_Colls']
        MBObject.Decor_LookBacks = Inputs['Decor_LookBacks']
        MBObject.Decor_Step = Inputs['Decor_Step']
        MBObject.NumSamples = Inputs['NumSamples']

        return True

    # Extract Outputs into Output Dictionary
    def ProcessOutputs(self, MBObject):
        '''Function used to fill the Outputs dictionary with the results from the PyBoltz object.'''
        Outputs={}
        Outputs['Drift_vel']      = PBRes(
            np.array([MBObject.VelocityX,MBObject.VelocityY,MBObject.VelocityZ]),
            np.array([MBObject.VelocityErrorX,MBObject.VelocityErrorY,MBObject.VelocityErrorZ]))
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
        Outputs['DTensor']       = PBRes(np.array(DTensor), np.array(DTensorErr))

        return Outputs

    # Run PyBoltz with chosen settings
    def Run(self,MySettings):
        '''Function used to run the PyBoltz simulation. Note that the PBSettings dictionary needs to be set up.'''
        MBObject = PyBoltz()
        Status=self.ProcessInputs(MBObject,MySettings)
        if(Status):
            MBObject.Start()
        Outputs = self.ProcessOutputs(MBObject)
        return Outputs

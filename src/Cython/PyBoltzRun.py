import numpy as np
from Magboltz import Magboltz

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
                   'OF'                    :0}
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
        MBObject.EFIELD=Inputs['EField_Vcm']
        MBObject.NumberOfGases=len(Inputs['Gases'])
        NumberOfGasesN=np.zeros(6,dtype='int')
        FRAC=np.zeros(6,dtype='float')
        for i in range(len(Inputs['Gases'])):
            NumberOfGasesN[i] = self.GasCode(Inputs['Gases'][i])
            FRAC[i]  = Inputs['Fractions'][i]
        MBObject.NumberOfGasesN  = NumberOfGasesN
        MBObject.FRAC   = FRAC
        MBObject.NMAX   = Inputs['Max_collisions']
        MBObject.IPEN   = Inputs['Enable_penning']
        MBObject.EnableThermalMotion  = Inputs['Enable_thermal_motion']
        MBObject.EFINAL = Inputs['Max_electron_energy']
        MBObject.TEMPC  = Inputs['Temperature_C']
        MBObject.TORR   = Inputs['Pressure_Torr']
        MBObject.BFieldMag   = Inputs['BField_Tesla']
        MBObject.BFieldAngle = Inputs['BField_angle']
        MBObject.OF     = Inputs['OF']
        MBObject.NANISO = Inputs['Angular_dist_model']
        return True

    # Extract Outputs into Output Dictionary
    def ProcessOutputs(self, MBObject):
        Outputs={}
        Outputs['Drift_vel']      = PBRes([MBObject.WX,MBObject.WY,MBObject.WZ],[MBObject.DWX,MBObject.DWY,MBObject.DWZ])
        Outputs['DT']             = PBRes(MBObject.DIFTR, MBObject.DFTER)
        Outputs['DL']             = PBRes(MBObject.DIFLN, MBObject.DFLER)
        Outputs['DT1']            = PBRes(MBObject.DTMN,MBObject.DFTER1)
        Outputs['DL1']            = PBRes(MBObject.DLMN,MBObject.DFLER1)

        Outputs['MeanEnergy']    = PBRes(MBObject.AVE,MBObject.DEN)
        DTensor     = [[MBObject.DIFXX, MBObject.DIFXY, MBObject.DIFXZ],
                       [MBObject.DIFXY, MBObject.DIFYY, MBObject.DIFYZ],
                       [MBObject.DIFXZ, MBObject.DIFYZ, MBObject.DIFZZ]]
        DTensorErr  = [[MBObject.DXXER, MBObject.DXYER, MBObject.DXZER],
                       [MBObject.DXYER, MBObject.DYYER, MBObject.DYZER],
                       [MBObject.DXZER, MBObject.DYZER, MBObject.DZZER]]            
        Outputs['DTensor']       = PBRes(DTensor, DTensorErr)

        return Outputs

    # Run PyBoltz with chosen settings
    def Run(self,MySettings):
        MBObject = Magboltz()
        Status=self.ProcessInputs(MBObject,MySettings)
        if(Status):
            MBObject.Start()
        Outputs = self.ProcessOutputs(MBObject)
        return Outputs

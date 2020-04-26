from Gases.CF4 cimport Gas1
from Gases.ARGON cimport Gas2
from Gases.HELIUM4 cimport Gas3
from Gases.HELIUM3 cimport Gas4
from Gases.NEON cimport Gas5
from Gases.KRYPTON cimport Gas6
from Gases.XENON cimport Gas7
from Gases.CH4 cimport Gas8
from Gases.ETHANE cimport Gas9
from Gases.PROPANE cimport Gas10
from Gases.ISOBUTANE cimport Gas11
from Gases.CO2 cimport Gas12
from Gases.H2O cimport Gas14
from Gases.OXYGEN cimport Gas15
from Gases.NITROGEN cimport Gas16
from Gases.HYDROGEN cimport Gas21
from Gases.DEUTERIUM cimport Gas22
from Gases.DME cimport Gas25

from Gases.XENONMERT cimport Gas61
from Gases.XENON_PAC cimport Gas107
from Gases.XENON_PRESSURE cimport Gas95
from Gases.XENON_DIMER cimport Gas77
from libc.string cimport memset
from Gas cimport Gas
import sys
cdef void callGASF(Gas*GAS, Params):
    if GAS.GasNumber == 1:
        Gas1(GAS)
    elif GAS.GasNumber == 2:
        Gas2(GAS)
    elif GAS.GasNumber == 3:
        Gas3(GAS)
    elif GAS.GasNumber == 4:
        Gas4(GAS)
    elif GAS.GasNumber == 5:
        Gas5(GAS)
    elif GAS.GasNumber == 6:
        Gas6(GAS)
    elif GAS.GasNumber == 7:
        Gas7(GAS)
    elif GAS.GasNumber == 8:
        Gas8(GAS)
    elif GAS.GasNumber == 9:
        Gas9(GAS)
    elif GAS.GasNumber == 10:
        Gas10(GAS)
    elif GAS.GasNumber == 11:
        Gas11(GAS)
    elif GAS.GasNumber == 12:
        Gas12(GAS)
    elif GAS.GasNumber == 14:
        Gas14(GAS)
    elif GAS.GasNumber == 15:
        Gas15(GAS)
    elif GAS.GasNumber == 16:
        Gas16(GAS)
    elif GAS.GasNumber == 21:
        Gas21(GAS)
    elif GAS.GasNumber == 22:
        Gas22(GAS)
    elif GAS.GasNumber == 25:
        Gas25(GAS)
    elif GAS.GasNumber == 61:
        Gas61(GAS, Params['A'], Params['D'], Params['F'], Params['A1'], Params['Lambda'], Params['EV0'])
    elif GAS.GasNumber == 95:
        Gas95(GAS, Params['Press_Fact'])
    elif GAS.GasNumber == 107:
        Gas107(GAS)
    elif GAS.GasNumber == 77:
        Gas77(GAS)

cdef class Gasmix:
    """
    The Gasmix object is used to coordinate the calling of different gas functions. It contains an array of six Gas object structs.
    This object is used by the Mixer functions to get the cross section outputs.
    """

    def InitWithInfo(self, GasNumber, InelasticCrossSectionPerGas, N_Inelastic, PenningFraction, EG, SqrtEnergy,
                     NumberOfGases, EnergySteps,
                     WhichAngularModel, EnergyStep, FinalEnergy, ThermalEnergy, TemperatureC, Pressure, PIR2,
                     RhydbergConst):
        '''This functions simply initiates the gas data from the parameters. This functions fills the output arrays to zeros.'''
        cdef int i, j;
        for i in range(6):
            self.Gases[i].GasNumber = GasNumber[i]
            for j in range(250):
                self.Gases[i].InelasticCrossSectionPerGas[j][:] = InelasticCrossSectionPerGas[i][j]
            self.Gases[i].N_Inelastic = N_Inelastic[i]
            for j in range(3):
                self.Gases[i].PenningFraction[j][:] = PenningFraction[i][j]
            self.Gases[i].EG = EG
            self.Gases[i].SqrtEnergy = SqrtEnergy
            self.Gases[i].NumberOfGases = NumberOfGases
            self.Gases[i].EnergySteps = EnergySteps
            self.Gases[i].WhichAngularModel = WhichAngularModel
            self.Gases[i].FinalEnergy = FinalEnergy
            self.Gases[i].ThermalEnergy = ThermalEnergy
            self.Gases[i].EnergyStep = EnergyStep
            self.Gases[i].TemperatureC = TemperatureC
            self.Gases[i].Pressure = Pressure
            self.Gases[i].PIR2 = PIR2
            self.Gases[i].RhydbergConst = RhydbergConst
            memset(self.Gases[i].Q, 0, 6 * 4000 * sizeof(double))
            memset(self.Gases[i].IonizationCrossSection, 0, 30 * 4000 * sizeof(double))
            memset(self.Gases[i].PEIonizationCrossSection, 0, 30 * 4000 * sizeof(double))
            memset(self.Gases[i].AttachmentCrossSection, 0, 8 * 4000 * sizeof(double))
            memset(self.Gases[i].NullCrossSection, 0, 10 * 4000 * sizeof(double))

    def reset(self):
        '''Function used to zero out the main output arrays.'''
        for i in range(6):
            memset(self.Gases[i].Q, 0, 6 * 4000 * sizeof(double))
            memset(self.Gases[i].IonizationCrossSection, 0, 30 * 4000 * sizeof(double))
            memset(self.Gases[i].PEIonizationCrossSection, 0, 30 * 4000 * sizeof(double))
            memset(self.Gases[i].AttachmentCrossSection, 0, 8 * 4000 * sizeof(double))
            memset(self.Gases[i].NullCrossSection, 0, 10 * 4000 * sizeof(double))

    def Run(self):
        '''This functions calls the corresponding gas functions.'''
        cdef int i
        cdef Gas temp
        for i in range(6):
            callGASF(&self.Gases[i], self.ExtraParameters)

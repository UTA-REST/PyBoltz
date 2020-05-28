import time
from PyBoltz.OdieRun import *

print("")
print("PyBoltz, adapted by B. Al. Atoum, A.D. McDonald, and B.J.P. Jones")
print("  from the original FORTRAN code MagBoltz by S. Biagi")

print("The PyBoltz to Garfield++ converter, Odie, written by A. B. Cudd.")
print("-----------------------------------------------------------------")

#Set up helper object
Odie = OdieRun()

# Configure settings for our simulation
base_settings = {
    'Gases': ['ARGON', 'CH4'],
    'Fractions': [90.0, 10.0],
    'Max_collisions': 4e6,
    'EField_Vcm': 100,
    'Max_electron_energy': 0,
    'Temperature_C': 23,
    'Pressure_Torr': 750.062,
    'BField_Tesla': 0,
    'BField_angle': 0,
    'Angular_dist_model': 2,
    'Enable_penning': 0,
    'Enable_thermal_motion': 1,
    'ConsoleOutputFlag': 1
}

t_start = time.time()

#Load base settings into Odie, this is required for
#running on a grid.
Odie.LoadSettings(base_settings, PrintSettings=True)

#Odie also can read in the settings from a JSON file
#Odie.LoadSettings(JSONFileName="input.json", PrintSettings=True)

#Generate output for a grid of possible EFields
#in four steps from 50 V/cm to 200 V/cm on a linear scale
GridOutput = Odie.GenerateGasGrid(50, 200, 4, LogScale=False)

#Generate output for a grid of possible EFields
#in four steps from 50 V/cm to 200 V/cm on a log scale,
#and in three steps from 0 to 1 Telsa for the BField (linear scale)
#GridOutput = Odie.GenerateGasGrid(50, 200, 4, LogScale=True, 0, 1, 3)

#Finally, a grid of EFields, BFields, and E-B Angles can be generated
#GridOutput = Odie.GenerateGasGrid(
#    minE=50, maxE=200, nE=4, LogScale=False, minB=0, maxB=1, nB=3, minA=0, maxA=90, nA=3
#)

t_end = time.time()

#Write output to Garfield++ style gas file
Odie.WriteGasFile("ar_90_ch4_10.gas", GridOutput)

print("Simulation time: {}\n".format(t_end - t_start))

#The output for each grid point is indexed by a tuple of the EField,
#E-B Angle, and BField. In general this loop will iterate over all points.
print("Printing (some) gas properties...")
for e in Odie.GridSettings['EFields']:
    for b in Odie.GridSettings['BFields']:
        for a in Odie.GridSettings['EBAngles']:
            Output = GridOutput[(e, b, a)]

            print("\nE={} V/cm, B={} T, A={}".format(e, b, a))
            print(
                "Vz: {:.3f} +/- {:.3f}".format(
                    Output['Drift_vel'].val[2], Output['Drift_vel'].err[2]
                )
            )
            print("DL: {:.3f} +/- {:.3f}".format(Output['DL'].val, Output['DL'].err))
            print("DT: {:.3f} +/- {:.3f}".format(Output['DT'].val, Output['DT'].err))

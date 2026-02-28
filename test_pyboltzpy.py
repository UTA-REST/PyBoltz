import sys, time, traceback
sys.path.insert(0, '.')

try:
    from PyBoltzPy.Boltz import Boltz
    from PyBoltzPy import Setups, Mixers, EnergyLimits
    print("Step 1: imports OK")

    b = Boltz()
    print("Step 2: Boltz() OK")

    # Set up like PyBoltzRun does
    import numpy as np
    b.EField = 100
    b.NumberOfGases = 1
    GasIDs = np.zeros(6, dtype='int')
    GasFractions = np.zeros(6, dtype='float')
    GasIDs[0] = 2  # ARGON index
    GasFractions[0] = 100.0
    b.GasIDs = GasIDs
    b.GasFractions = GasFractions
    b.MaxNumberOfCollisions = 5e4
    b.Enable_Penning = 0
    b.Enable_Thermal_Motion = 0
    b.Max_Electron_Energy = 0
    b.TemperatureCentigrade = 23
    b.Pressure_Torr = 750
    b.BField_Mag = 0
    b.BField_Angle = 0
    b.Console_Output_Flag = 1
    b.Which_Angular_Model = 2
    b.Decor_Colls = 0
    b.Decor_Lookbacks = 0
    b.Decor_Step = 0
    b.Num_Samples = 10
    print("Step 3: params set OK")

    Setups.Setup(b)
    print("Step 4: Setup OK")
    print(f"  CONST1={b.CONST1}, CONST3={b.CONST3}, PresTempCor={b.PresTempCor}")

    # Set up energy limit loop
    b.Max_Electron_Energy = 0.5
    b.InitialElectronEnergy = b.Max_Electron_Energy / 50.0
    b.BFieldMode = 1

    print("Step 5: Calling Mixer...")
    Mixers.Mixer(b)
    print("Step 5: Mixer OK")
    print(f"  NumMomCrossSectionPointsNT={b.NumMomCrossSectionPointsNT}")
    print(f"  MaxCollisionFreqNT={b.MaxCollisionFreqNT}")

except Exception as e:
    traceback.print_exc()
    print(f"Error: {e}")

"""
End-to-end smoke test: PyBoltzPy full pipeline (Mixer -> EnergyLimit -> MONTE -> end).
Uses small collision count so it completes quickly.
"""
import sys, time
sys.path.insert(0, '.')

from PyBoltzPy.PyBoltzRun import PyBoltzRun

settings = {
    'Gases': ['ARGON'],
    'Fractions': [100],
    'Max_collisions': 5e4,
    'EField_Vcm': 100,
    'Max_electron_energy': 0.5,
    'Temperature_C': 23,
    'Pressure_Torr': 750,
    'BField_Tesla': 0,
    'BField_angle': 0,
    'Angular_dist_model': 2,
    'Enable_penning': 0,
    'Enable_thermal_motion': 0,
    'ConsoleOutputFlag': 0,
    'Decor_Colls': 0,
    'Decor_Step': 0,
    'Decor_LookBacks': 0,
    'NumSamples': 5,
}

print("Starting PyBoltzPy end-to-end test..."); sys.stdout.flush()
t0 = time.time()
out = PyBoltzRun().Run(settings)
elapsed = time.time() - t0

print(f"Completed in {elapsed:.1f}s")
vz = out['Drift_vel'].val[2]
print(f"VelocityZ = {vz:.4f} cm/us")
print(f"MeanEnergy = {out['MeanEnergy'].val:.4f} eV")
print(f"DT = {out['DT'].val:.4f}")
print(f"DL = {out['DL'].val:.4f}")

# Rough sanity check: Ar at 100 V/cm, 750 Torr should give ~0.04-0.1 cm/us
if 0.01 < vz < 1.0:
    print("PASS: VelocityZ in plausible range for Ar at 100 V/cm")
else:
    print(f"WARN: VelocityZ={vz} outside expected range 0.01-1.0 cm/us")

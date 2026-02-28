"""
Benchmark: Cython PyBoltz vs pure-Python PyBoltzPy
Runs Argon at 100 V/cm, 750 Torr with no B-field, no thermal motion.
"""
import time
import sys

sys.path.insert(0, '/Users/benjpjones/PycharmProjects/PyBoltz')

settings = {
    'Gases': ['ARGON'],
    'Fractions': [100],
    'Max_collisions': 4e7,
    'EField_Vcm': 100,
    'Max_electron_energy': 8.0,  # Fixed high ceiling so both versions use the same energy grid
    'Temperature_C': 23,
    'Pressure_Torr': 750,
    'BField_Tesla': 0,
    'BField_angle': 0,
    'Angular_dist_model': 2,
    'Enable_penning': 0,
    'Enable_thermal_motion': 0,
    'ConsoleOutputFlag': 0,
    'Decor_Colls': 0,
    'Decor_LookBacks': 0,
    'Decor_Step': 0,
    'NumSamples': 10,
}

N_collisions = settings['Max_collisions']

# --- Cython version ---
print(f"Running Cython version ({N_collisions:.0e} collisions)...")
from PyBoltz.PyBoltzRun import PyBoltzRun as CythonRun
t0 = time.time()
cython_out = CythonRun().Run(dict(settings))
t_cython = time.time() - t0
cython_vz = cython_out['Drift_vel'].val[2]
print(f"  Vz = {cython_vz:.4f} cm/us,  time = {t_cython:.2f}s")

# --- Pure Python version ---
print(f"\nRunning pure Python version ({N_collisions:.0e} collisions)...")
from PyBoltzPy.PyBoltzRun import PyBoltzRun as PythonRun
t0 = time.time()
python_out = PythonRun().Run(dict(settings))
t_python = time.time() - t0
python_vz = python_out['Drift_vel'].val[2]
print(f"  Vz = {python_vz:.4f} cm/us,  time = {t_python:.2f}s")

# --- Numba version (first call includes compile time) ---
print(f"\nWarming up Numba JIT (compiling)...")
from PyBoltzPy.Boltz import Boltz as NBoltz
warm_obj = NBoltz()
warm_obj.use_numba = True
from PyBoltzPy.PyBoltzRun import PyBoltzRun as NRun
warm_run = NRun()
warm_settings = dict(settings)
warm_settings['Max_collisions'] = 1e4
t_compile = time.time()
warm_run.ProcessInputs(warm_obj, warm_settings)
warm_obj.Start()
t_compile = time.time() - t_compile
print(f"  Compile + warmup run: {t_compile:.1f}s")

print(f"\nRunning Numba version ({N_collisions:.0e} collisions)...")
t0 = time.time()
numba_out = NRun().Run(dict(settings) | {'use_numba': True})
t_numba = time.time() - t0
numba_vz = numba_out['Drift_vel'].val[2]
print(f"  Vz = {numba_vz:.4f} cm/us,  time = {t_numba:.2f}s")

# --- Results ---
print("\n--- Benchmark Results ---")
print(f"{'Version':<12} {'Collisions':>12}  {'Time':>8}  {'Throughput':>18}  {'Vz (cm/us)':>12}")
print("-" * 70)
cython_rate = N_collisions / t_cython
python_rate = N_collisions / t_python
numba_rate  = N_collisions / t_numba
print(f"{'Cython':<12} {N_collisions:>12.0e}  {t_cython:>7.2f}s  {cython_rate:>16.0f}/s  {cython_vz:>12.4f}")
print(f"{'Python':<12} {N_collisions:>12.0e}  {t_python:>7.2f}s  {python_rate:>16.0f}/s  {python_vz:>12.4f}")
print(f"{'Numba':<12} {N_collisions:>12.0e}  {t_numba:>7.2f}s  {numba_rate:>16.0f}/s  {numba_vz:>12.4f}")
print()
print(f"Python is {cython_rate/python_rate:.1f}x slower than Cython")
print(f"Numba  is {cython_rate/numba_rate:.1f}x slower than Cython  ({python_rate/numba_rate:.1f}x faster than pure Python)")

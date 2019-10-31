import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def line(x, a,b):
    return a * x+b


def GetTarget(maxTime):
    step = (maxTime - 1.0)-1
    X = np.arange(1.0,maxTime,step)
    return X

def CalculateDiffusion(XOutput,YOutput,ZOutput,TimeSumOutput,numElectrons):
    MaxTime = 1e9
    MT = 0.0
    for O in range(numElectrons):
        MT = TimeSumOutput[O][99] - TimeSumOutput[O][0]
        if MT < MaxTime:
            MaxTime = MT
    TIME = np.zeros(100)
    XVALS = np.zeros(100)
    YVALS = np.zeros(100)
    ZVALS = np.zeros(100)
    #targets = np.linspace(0.1, MaxTime, 30)
    targets = GetTarget(MaxTime)
    SigmaDlz = []
    SigmaDty = []
    SigmaDtx = []
    for target in targets:
        Xse = []
        Yse = []
        Zse = []

        for O in range(numElectrons):
            for j in range(100):
                TIME[j] = TimeSumOutput[O][j] - TimeSumOutput[O][0]
                XVALS[j] = XOutput[O][j] - XOutput[O][0]
                YVALS[j] = YOutput[O][j] - YOutput[O][0]
                ZVALS[j] = ZOutput[O][j] - ZOutput[O][0]

            f = interpolate.interp1d(TIME, XVALS)
            Xse.append(f(target))

            f = interpolate.interp1d(TIME, YVALS)
            Yse.append(f(target))

            f = interpolate.interp1d(TIME, ZVALS)
            Zse.append(f(target))

        SigmaDlz.append(np.std(Zse))
        SigmaDty.append(np.std(Yse))
        SigmaDtx.append(np.std(Xse))

    SigmaDlz = np.array(SigmaDlz)
    errorDlz = np.sqrt((2 * SigmaDlz ** 4 / (numElectrons - 1)))

    SigmaDty = np.array(SigmaDty)
    errorDty = np.sqrt((2 * SigmaDty ** 4 / (numElectrons - 1)))

    SigmaDtx = np.array(SigmaDtx)
    errorDtx = np.sqrt((2 * SigmaDtx ** 4 / (numElectrons - 1)))

    print("Fitting the diffusions \n")
    const = 5e15
    print("Longitudinal")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDlz ** 2, sigma=errorDlz)
    print("Dl =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)


    print("Transverse")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDtx ** 2, sigma=errorDtx)
    print("Dtx =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)

    print("Transverse")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDty ** 2, sigma=errorDty)
    print("Dty =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)

    print("Fitting the diffusions \n")
    const = 5e15
    print("Longitudinal")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDlz ** 2, sigma=errorDlz)
    print("Dl =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)

    print("Transverse")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDtx ** 2, sigma=errorDtx)
    print("Dtx =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)

    print("Transverse")
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, targets, SigmaDty ** 2, sigma=errorDty)
    print("Dty =", popt[0] * const, "+/-", pcov[0, 0] ** 0.5 * const)

    plt.figure(figsize=(8, 7))

    plt.errorbar(targets, SigmaDlz ** 2 * 1e15, yerr=errorDlz * 1e15, fmt='s', alpha=0.7, color='darkorchid',
                 label="Longitudinal Z")

    plt.errorbar(targets, SigmaDtx ** 2 * 1e15, yerr=errorDtx * 1e15, fmt='s', alpha=0.7, color='darkblue',
                 label="Transverse X")

    plt.errorbar(targets, SigmaDty ** 2 * 1e15, yerr=errorDty * 1e15, fmt='s', alpha=0.7, color='darkred',
                 label="Transverse Y")

    popt, pcov = curve_fit(line, targets, SigmaDty ** 2 * 1e5, sigma=errorDty * 1e5)
    plt.plot(targets, line(targets, popt[0], popt[1]), color='k')


    plt.grid()
    plt.legend(loc="upper left", fontsize=24)
    plt.xlabel("Time [ns]", fontsize=24)
    plt.ylabel("Variance of swarm [1e15]", fontsize=24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

    plt.figure(figsize=(8, 7))

    plt.errorbar(targets, SigmaDlz ** 2 * 1e5, yerr=errorDlz * 1e5, fmt='s', alpha=0.7, color='darkorchid',
                 label="Longitudinal Z")

    popt , pcov = curve_fit(line, targets, SigmaDlz ** 2 * 1e5, sigma=errorDlz * 1e5)
    plt.plot(targets, line(targets, popt[0], popt[1]), color='k')

    plt.errorbar(targets, SigmaDtx ** 2 * 1e5, yerr=errorDtx * 1e5, fmt='s', alpha=0.7, color='darkblue',
                 label="Transverse X")

    plt.errorbar(targets, SigmaDty ** 2 * 1e5, yerr=errorDty * 1e5, fmt='s', alpha=0.7, color='darkred',
                 label="Transverse Y")

    popt, pcov = curve_fit(line, targets, SigmaDty ** 2 * 1e5, sigma=errorDty * 1e5)
    plt.plot(targets, line(targets, popt[0], popt[1]), color='k')

    plt.grid
    plt.grid()
    plt.legend(loc="upper left", fontsize=24)
    plt.xlabel("Time [Biagis]", fontsize=24)
    plt.ylabel("Variance of swarm [1e5]", fontsize=24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

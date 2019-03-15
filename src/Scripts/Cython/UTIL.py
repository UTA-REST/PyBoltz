####################################
########## Here I will look into the MERT for the cross section
####################################
import numpy as np



def MERT(epsilon, A, D, F, A1):
    a0 = 1  # 5.29e-11  # in m
    hbar = 1  # 197.32697*1e-9 # in eV m
    m = 1  # 511e3     # eV/c**2
    alpha = 27.292 * a0 ** 3
    k = np.sqrt((epsilon) / (13.605 * a0 ** 2))

    eta0 = -A * k * (1 + (4 * alpha) / (3 * a0) * k ** 2 * np.log(k * a0)) \
           - (np.pi * alpha) / (3 * a0) * k ** 2 + D * k ** 3 + F * k ** 4

    eta1 = (np.pi) / (15 * a0) * alpha * k ** 2 - A1 * k ** 3

    Qm = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0) - np.arctan(eta1))) ** 2

    Qt = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0))) ** 2

    return Qm * (5.29e-11) ** 2 * 1e20, Qt * (5.29e-11) ** 2 * 1e20


def WEIGHT_Q(eV, Qm, BashBoltzQm, Lamda, eV0):
    WeightQm = (1 - np.tanh(Lamda * (eV - eV0))) / 2
    WeightBB = (1 + np.tanh(Lamda * (eV - eV0))) / 2

    NewBashQm = BashBoltzQm * WeightBB
    NewMERTQm = Qm * WeightQm
    NewQm = NewBashQm + NewMERTQm
    return NewQm


def HYBRID_X_SECTIONS(MB_EMTx, MB_EMTy, MB_ETx, MB_ETy, A, D, F, A1, Lambda, eV0):
    Qm_MERT, Qt_MERT = MERT(MB_EMTx, A, D, F, A1)
    New_Qm = WEIGHT_Q(MB_EMTx, Qm_MERT, MB_EMTy, Lambda, eV0)
    Qm_MERT, Qt_MERT = MERT(MB_ETx, A, D, F, A1)
    New_Qt = WEIGHT_Q(MB_ETx, Qt_MERT, MB_ETy, Lambda, eV0)

    return MB_EMTx, New_Qm, MB_ETx, New_Qt
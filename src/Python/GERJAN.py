import Magboltz
import numpy as np
import math

import random

def GERJAN(RDUM,API):
    random.seed(RDUM)
    RNMX =np.zeros(6)

    for J in range(0,5,2):
        RAN1 = random.random()
        RAN2 = random.random()
        TWOPI = 2.0 *API
        RNMX[J]=math.sqrt(-np.log(RAN1))*math.cos(RAN2*TWOPI)
        RNMX[J+1] = math.sqrt(-np.log(RAN1)) * math.sin(RAN2 * TWOPI)
    return RNMX



import Magboltz
import numpy as np
import math

from RAND48 import Rand48

def GERJAN(RSTART,API):
    RNMX = [0 for i in range(6)]
    RAND48 = Rand48()

    RAND48.seed(RSTART)
    for J in range(0,5,2):

        RAN1 = RAND48.drand()
        RAN2 = RAND48.drand()
        TWOPI = 2.0 *API
        RNMX[J]=math.sqrt(-np.log(RAN1))*math.cos(RAN2*TWOPI)
        RNMX[J+1] = math.sqrt(-np.log(RAN1)) * math.sin(RAN2 * TWOPI)
    return RNMX



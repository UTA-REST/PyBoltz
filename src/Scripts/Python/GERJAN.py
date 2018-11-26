import Magboltz
import numpy as np
import math
from random import seed
from random import random

def GERJAN(RSTART,API):
    RNMX = [0 for i in range(6)]
    for J in range(0,5,2):
        seed(RSTART)
        RAN1 = random()
        RAN2 = random()
        TWOPI = 2.0 *API
        RNMX[J]=math.sqrt(-np.log(RAN1))*math.cos(RAN2*TWOPI)
        RNMX[J+1] = math.sqrt(-np.log(RAN1)) * math.sin(RAN2 * TWOPI)
    return RNMX



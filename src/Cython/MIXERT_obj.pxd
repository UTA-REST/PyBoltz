cdef class MIXERT_obj:
    cdef public:
        int NIN[6], NATT[6], NNULL[6], NION[6]
        double ECHARG, EHALF, EMASS, KELSUM
        double Q[6][6][4000],QIN[6][250][4000], E[6][6], EIN[6][250], KIN[6][250], QION[6][30][4000], PEQION[6][30][4000], EION[6][30], EB[6][30], PEQEL[6][6][4000]
        double PEQIN[6][250][4000], KEL[6][6], PENFRA[6][3][290], NC0[6][30], EC0[6][30], WK[6][30], EFL[6][30], NG1[6][30], EG1[6][30], NG2[6][30], EG2[6][30], SCLN[6][10]
        double QATTT[6][8][4000], QNULL[6][10][4000]

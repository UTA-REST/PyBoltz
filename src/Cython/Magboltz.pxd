cimport numpy as np
import math
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef double drand48(double dummy)

cdef class Magboltz:
    cdef public:
        double EOVB
        '''This is a constant that is equal to the electric field / magentic field * 1e-9.'''
        double WB
        '''This is the angular speed of rotation see cycltron frequency.'''
        double BFieldAngle
        '''This is the angle between the magnetic field and the electric field.'''
        double BFieldMag
        '''This is the magnitude of the magentic field.'''
        double EFINAL
        '''This is the upper limit for the electron energy integration.'''
        double ESTEP
        '''Magboltz does the electron energy integration in 4000 steps this variable has the difference in energy between each step.'''
        double AKT
        '''This indicates the amount of energy in the gas (it is equal to the Boltzman constant * absolute tempreture).'''
        double ARY
        '''This is Rydberg constant times hc in eV.'''
        double TEMPC
        '''This is the tempreture in degrees Centigrade.'''
        double TORR
        '''This is the pressure in Torr.'''
        double TMAX
        '''Maximum collision time. Default is set to 100.'''
        double SMALL
        '''This constant is equal to 1e-20. Used to be a small constant.'''
        double API
        '''The value of PI.'''
        double ESTART
        '''The lower limit of the electron energy integration.'''
        double THETA
        '''Collision scattering angle from the Z plane.'''
        double PHI
        '''Angle used to calculate initial direction cosines in the XY plane. Angle from the X axis.'''
        double EFIELD
        '''Electric field [V/cm].'''
        double NMAX
        '''Number of the simulated collisions * N.'''
        double ALPHA
        '''Townsend coefficient.'''
        double WX
        '''Drift velocity on the x axis.'''
        double WY
        '''Drift velocity on the y axis.'''
        double WZ
        '''Drift velocity on the z axis.'''
        double DWX
        '''Percentage error on the x drift velocity.'''
        double DWY
        '''Percentage error on the y drift velocity.'''
        double DWZ
        '''Percentage error on the z drift velocity.'''
        double TTOTS
        ''''''
        double ATT
        '''Townsend coefficient.'''
        double ALPER
        '''Error for ALPHA.'''
        double ATTER
        '''Error for ATT'''
        double DIFLN
        '''Longitudinal diffusion.'''
        double DIFTR
        '''Transverse diffusion.'''
        double DIFXX
        '''Diffusion on the x plane.'''
        double DIFYY
        '''Diffusion on the y plane.'''
        double DIFZZ
        '''Diffusion on the z plane.'''
        double DIFYZ
        '''Diffusion on the yz plane.'''
        double DIFXY
        '''Diffusion on the xy plane.'''
        double DIFXZ
        '''Diffusion on the xz plane.'''
        double DXXER
        '''Percentage error for DIFXX.'''
        double DYYER
        '''Percentage error for DIFYY.'''
        double DZZER
        '''Percentage error for DIFZZ.'''
        double DYZER
        '''Percentage error for DIFYZ.'''
        double DXYER
        '''Percentage error for DIFXY.'''
        double DXZER
        '''Percentage error for DIFXZ.'''
        double TMAX1
        '''This variable is used to store the maximum collision time in the simulation.'''
        double DEN
        '''Percentage error for AVE.'''
        double AVE
        '''Mean electron energy.'''
        double XID
        ''''''
        double X
        '''Variable used to represent the x position in the simulation.'''
        double Y
        '''Variable used to represent the y position in the simulation.'''
        double Z
        '''Variable used to represent the z position in the simulation.'''
        double DFLER
        '''Percentage for DIFLN.'''
        double DFTER
        '''Percentage for DIFTR.'''
        double TGAS
        '''Absolute tempreture in Kelvin.'''
        double ALPP
        '''Variable used to represent the ionisation rate.'''
        double ATTP
        '''Variable used to represent the attachement rate.'''
        double SSTMIN
        '''Variable used to represent the minmum difference between ALPP and ATTP to include spatial gradients. SST - Steady State Townsend Parameters.'''
        double VDOUT
        '''VD for Steady State Townsend Parameters.'''
        double VDERR
        '''Percentage error for VDOUT.'''
        double WSOUT
        '''VD for Steady State Townsend Parameters.'''
        double WSERR
        '''Percentage error for WSOUT.'''
        double DLOUT
        '''Steady State Townsend Longitudinal diffusion.'''
        double DLERR
        '''Percentage error for DLOUT.'''
        double NMAXOLD
        '''NMAX for SST - Steady State Townsend'''
        double DTOUT
        '''Steady State Townsend Transverse diffusion.'''
        double DTERR
        '''Percentage error for DTERR.'''
        double ALPHSST
        '''Ionisation rate from SST - Steady State Townsend.'''
        double ATTOINT
        double ATTERT
        double AIOERT
        double ALPHERR
        double ATTSST
        double TTOT
        double ATTERR
        double IZFINAL
        double RALPHA
        '''PT ionisation rate.'''
        double RALPER
        '''Percentage error for RALPHA.'''
        double TODENE
        double TOFENER
        double TOFENE
        double TOFWV
        double TOFWVER
        double TOFDL
        double TOFDLER
        double TOFDT
        double TOFDTER
        double TOFWR
        double TOFWRER
        double RATTOF
        '''PT attachment rate.'''
        double RATOFER
        '''Percentage error for RATTOF'''
        double ALPHAST
        double VDST
        double TSTEP
        double ZSTEP
        double TFINAL
        double RATTOFER
        double ZFINAL
        double ITFINAL
        double IPRIM
        double TOFWVZ, TOFWVZER, TOFWVX, TOFWVXER, TOFWVY, TOFWVYER, TOFDZZ, TOFDZZER, TOFDXX, TOFDXXER, TOFDYY, TOFDYYER, TOFDYZ, TOFDYZER, TOFDXZ, TOFDXZER, TOFDXY, TOFDXYER, TOFWRZ, TOFWRZER, TOFWRY, TOFWRYER, TOFWRX, TOFWRXER, ATTOION, ATTIOER, ATTATER
        double CONST1
        '''Constant that is equal to AWB / 2 * 1e-9.'''
        double CONST2
        '''Constant that is equal to CONST1 * 1e-2.'''
        double CONST3
        '''Constant that is equal to sqrt(0.2 * AWB) * 1e-9.'''
        #TODO:find ALOSCH
        double CONST4
        '''Constant that is equal to CONST3 * ALOSCH * 1e-15.'''
        double CONST5
        '''Constant that is equal to CONST3 / 2.'''
        double TCFMX
        '''Sum of the maximum collision frequency of each gas.'''
        double EnableThermalMotion
        '''Variable used to indicate wethier to include thermal motion or not.'''
        double CORR
        '''Variable used to calculate the correlation constant between the pressure and tempreture. CORR=ABZERO*TORR/(ATMOS*(ABZERO+TEMPC)*100.0D0).'''
        double FAKEI
        double AN
        '''Variable that is equal to 100 * CORR * ALOSCH. Which is the number of molecules/cm^3.'''
        double VAN
        '''Variable that is equal to 100 * CORR * CONST4 * 1e15'''
        double ZTOT
        double ZTOTS
        double ST
        '''Variable used to calculate the sum of the collision times in the simulation.'''
        double RSTART
        '''Random number generator seed. Not used at the moment.'''
        long long NumberOfGases
        '''Number of gases in the mixture.'''
        long long NSTEP
        '''Steps for the electron energy integration.'''
        long long NANISO
        '''Variable used to indicate the type of the elastic angular distribtions.'''
        long long IPEN
        '''Variable used to indicate the inclusion of penning effects. '''
        long long NISO
        '''Anisotropic flag used if anisotropic scattering data is detected.'''
        long long IELOW
        '''Flag used to indicate if the energy limit has been crossed.'''
        long long NCOLM
        long long NCORLN
        '''Long decorrelation length'''
        long long NCORST
        '''Long decorrelation step.'''
        long long NNULL
        '''Number of null collisions.'''
        long long IFAKE
        '''Fake ionisation counter.'''
        long long ITMAX
        '''Number of samples to be taken in the Monte functions.'''
        long long NSCALE
        '''Constant equal to 40000000.'''
        double NESST[9]
        double DENSY[4000]
        double SPEC[4000]
        '''Number of collisions in the Ith energy step.'''
        double TIME[300]
        '''Number of collisions in the Ith time step.'''
        double ICOLL[6][5]
        '''Number of collisions for Ith gas at the Jth type.'''
        double ICOLNN[6][10]
        '''Null scatter sum for the Ith gas and the Jth location.'''
        double ICOLN[6][290]
        double ESPL[8], XSPL[8], TMSPL[8], TTMSPL[8], RSPL[8], RRSPL[8], RRSPM[8], YSPL[8], ZSPL[8], TSPL[8], XXSPL[8], YYSPL[8], ZZSPL[8], VZSPL[8], TSSUM[8], TSSUM2[8]
        double XS[2000], YS[2000], ZS[2000], TS[2000], DCX[2000], DCY[2000], DCZ[2000], IPL[2000], ETPL[8], XTPL[8], YTPL[8], ZTPL[8], YZTPL[8], XZTPL[8], XYTPL[8], VYTPL[8], VXTPL[8], TTPL[8], XXTPL[8], YYTPL[8], ZZTPL[8], VZTPL[8], NETPL[8], ZPLANE[8]
        double XSS[2000], YSS[2000], ZSS[2000], TSS[2000], ESS[2000], DCXS[2000], DCYS[2000], DCZS[2000], IPLS[2000]
        double AMGAS[6]
        double VTMB[6]
        '''Maxwell Boltzman velocity factor for each gas component.'''
        double TCFMXG[6]
        '''Fraction of the maximum collision frequency of each gas over the sum of the maximum collision frequencies of all the gases.'''
        double NumberOfGasesN[6]
        '''Array used to store the number of the 6 gases in the mixture.'''
        double FRAC[6]
        '''Array used to store the percentage of each gas in the mixture.'''
        double ANN[6]
        '''Array used to calculate the number of molecules/cm^3 for each gas.'''
        double VANN[6]
        '''Array used to calculate the VAN for each gas.'''
        double RI[8], EPT[8], VZPT[8], TTEST[8]
        double QSUM[4000], QEL[4000], QSATT[4000], ES[4000]
        double E[4000]
        '''Energy ar each energy step.'''
        double EROOT[4000]
        '''The square root of each energy step.'''
        double QTOT[4000], QREL[4000], QINEL[4000], FCION[4000], FCATT[4000], IFAKET[8], IFAKED[9], RNMX[9], LAST[6]
        double NIN[6]
        '''Number of momentum cross section data points for types other than attachment and ionisation.'''
        double LION[6], ALION[6]
        double IPLAST[6]
        '''Number of momentum cross section data points for each gas.'''
        double ISIZE[6]
        double TCFMAX[6]
        '''Maximum value of collision frequency for each gas.'''
        double NPLAST[6]
        '''Number of momentum cross section data points for null collisions.'''
        double INDEX[6][290], NC0[6][290], EC0[6][290], NG1[6][290], EG1[6][290], NG2[6][290], EG2[6][290], WKLM[6][290], EFL[6][290], EIN[6][290], IARRY[6][290], RGAS[6][290], IPN[6][290], WPL[6][290], QION[6][4000], QIN[6][250][4000], LIN[6][250], ALIN[6][250]
        double CF[6][4000][290]
        '''Collision frequency for each gas at every energy step for every data point.'''
        double TCF[6][4000]
        '''Total collision frequency for each gas at every energy step.'''
        double PENFRA[6][3][290]
        double CFN[6][4000][10]
        '''Null collision frequency for each gas at every energy step for every data point.'''
        double TCFN[6][4000]
        '''Total null collision frequency for each gas at every energy step.'''
        double SCLENUL[6][10], PSCT[6][4000][290], ANGCT[6][4000][290]
        double DTOVMB
        '''Transverse diffusion in EV.'''
        double DTMN
        '''Transverse diffusion in microns/cm^0.5.'''
        double DFTER1
        '''Percentage error for DTOVMB'''
        double DLOVMB
        '''Longitudinal diffusion in EV.'''
        double DLMN
        '''Longitudinal diffusion in microns/cm^0.5'''
        double DFLER1
        '''Percentage error for DLOVMB'''
        double IPLASTNT,ISIZENT
        double PIR2
        int OF
        '''Flag used to stop console printouts'''
        double MCT
        # Variables and arrays used when the thermal motion is not included.
        double CFNT[4000][960],EINNT[960],TCFNT[4000],IARRYNT[960],RGASNT[960],IPNNT[960],WPLNT[960],PENFRANT[3][960],TCFMAXNT[8]
        double CFNNT[4000][60],TCFNNT[4000],SCLENULNT[60],PSCTNT[4000][960],ANGCTNT[4000][960],INDEXNT[960],NC0NT[960],EC0NT[960]
        double NG1NT[960],EG1NT[960],NG2NT[960],EG2NT[960],WKLMNT[960],EFLNT[960]
        double ICOLLNT[30],ICOLNNT[960],ICOLNNNT[60]
        double NPLASTNT

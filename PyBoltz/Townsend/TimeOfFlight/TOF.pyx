from Townsend.CollisionFrequencyCalc import COLF
from Townsend.CollisionFrequencyCalc import COLFT
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from Boltz cimport Boltz

cpdef run(Boltz Object, int ConsoleOuput):
    cdef double FrequencyOfRealColli, FrequencyOfElasticColli, FrequencyOfIonisationColli, FrequencyOfAttachmentColli, FrequencyOfInelasticColli, TotalSumOfColli
    cdef double FakeIonisationCorrectionErr, NumberOfElecPlanes[8], VelocityZ[8],ANST[8]
    cdef double LongitudinalDiffusion[8], DiffusionX[8], DiffusionY[8],TransverseDiffusion2,TransverseDiffusion3,ATER
    cdef int J,I1,I2

    # Get ionisation and attachment frequencies
    if Object.Enable_Thermal_Motion:
        FrequencyOfRealColli, FrequencyOfElasticColli, FrequencyOfIonisationColli, FrequencyOfAttachmentColli, FrequencyOfInelasticColli, TotalSumOfColli = COLFT.run(Object)
    else:
        FrequencyOfRealColli, FrequencyOfElasticColli, FrequencyOfIonisationColli, FrequencyOfAttachmentColli, FrequencyOfInelasticColli, TotalSumOfColli = COLF.run(Object)

    # Attachment Ionisation ratio
    Object.AttachmentOverIonisationPT = FrequencyOfAttachmentColli / FrequencyOfIonisationColli

    FakeIonisationCorrectionErr = abs((Object.FakeIonisationsEstimate + FrequencyOfIonisationColli - FrequencyOfAttachmentColli) / (FrequencyOfIonisationColli - FrequencyOfAttachmentColli))

    # Plane 0
    NumberOfElecPlanes[0] = Object.NumberOfElectronsPlanes[1]
    VelocityZ[0] = Object.TZPlanes[1] / (NumberOfElecPlanes[0] * Object.TimeStep)
    LongitudinalDiffusion[0] = ((Object.TZ2Planes[1] / NumberOfElecPlanes[0]) - (Object.TZPlanes[1] / NumberOfElecPlanes[0]) ** 2) / (2 * Object.TimeStep)
    DiffusionX[0] = ((Object.TX2Planes[1] / NumberOfElecPlanes[0]) - (Object.TXPlanes[1] / NumberOfElecPlanes[0]) ** 2) / (2 * Object.TimeStep)
    DiffusionY[0] = ((Object.TY2Planes[1] / NumberOfElecPlanes[0]) - (Object.TYPlanes[1] / NumberOfElecPlanes[0]) ** 2) / (2 * Object.TimeStep)

    for J in range(2, Object.NumberOfTimeSteps + 1):
        # For each plane after 0  Calculate the velocity, Longitudinal diffusion and the x and y diffusion values.
        NumberOfElecPlanes[J - 1] = Object.NumberOfElectronsPlanes[J]
        VelocityZ[J - 1] = ((Object.TZPlanes[J] / NumberOfElecPlanes[J - 1]) - (Object.TZPlanes[J - 1] / NumberOfElecPlanes[J - 2])) / (Object.TimeStep)

        LongitudinalDiffusion[J - 1] = ((Object.TZ2Planes[J] / NumberOfElecPlanes[J - 1]) - (Object.TZPlanes[J] / NumberOfElecPlanes[J - 1]) ** 2 - (
                Object.TZ2Planes[J - 1] / NumberOfElecPlanes[J - 2]) + (Object.TZPlanes[J - 1] / NumberOfElecPlanes[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

        DiffusionX[J - 1] = ((Object.TX2Planes[J] / NumberOfElecPlanes[J - 1]) - (Object.TXPlanes[J] / NumberOfElecPlanes[J - 1]) ** 2 - (
                Object.TX2Planes[J - 1] / NumberOfElecPlanes[J - 2]) + (Object.TXPlanes[J - 1] / NumberOfElecPlanes[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

        DiffusionY[J - 1] = ((Object.TY2Planes[J] / NumberOfElecPlanes[J - 1]) - (Object.TYPlanes[J] / NumberOfElecPlanes[J - 1]) ** 2 - (
                Object.TY2Planes[J - 1] / NumberOfElecPlanes[J - 2]) + (Object.TYPlanes[J - 1] / NumberOfElecPlanes[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

    for J in range(Object.NumberOfTimeSteps):
        # Convert to right units
        VelocityZ[J]*=1e9
        LongitudinalDiffusion[J] *=1e16
        DiffusionX[J] *=1e16
        DiffusionY[J] *=1e16

    if ConsoleOuput:
        print("\nTime of flight results at " + str(Object.NumberOfTimeSteps) + " sequential time planes")
        print('{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}'.format("Plane #", "DL", "DX", "DY",
                                                                         "WR"))
        for J in range(Object.NumberOfTimeSteps):
            print(
                '{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}'.format(J + 1, LongitudinalDiffusion[J], DiffusionX[J],
                                                                       DiffusionY[J], VelocityZ[J]))

    # If there is more electrons in the first plane than the last plane, this means that there has been some attachment
    # occuring in the simulation.
    if Object.NumberOfElectronsPlanes[1] > Object.NumberOfElectronsPlanes[Object.NumberOfTimeSteps]:
        # For net attachment the results are taken from plane 2

        Object.TOFEnergy = Object.EnergyPT[1]
        Object.TOFEnergyErr = 100 * abs((Object.EnergyPT[1] - Object.EnergyPT[2]) / (2 * Object.EnergyPT[1]))

        Object.VelocityTOFPT = Object.VelocityZPT[1]
        Object.VelocityTOFPTErr = 100 * abs((Object.VelocityZPT[1] - Object.VelocityZPT[2]) / (2 * Object.VelocityZPT[1]))

        Object.LongitudinalDiffusionTOF = LongitudinalDiffusion[1]
        Object.LongitudinalDiffusionTOFErr = 100 * abs((LongitudinalDiffusion[1] - LongitudinalDiffusion[2]) / (2 * LongitudinalDiffusion[1]))

        TransverseDiffusion2 = (DiffusionX[1] + DiffusionY[1]) / 2
        TransverseDiffusion3 = (DiffusionX[2] + DiffusionY[2]) / 2

        Object.TransverseDiffusionTOF = TransverseDiffusion2
        Object.TransverseDiffusionTOFErr = 100 * abs((TransverseDiffusion2 - TransverseDiffusion3) / (2 * TransverseDiffusion2))

        Object.VelocityTOF = VelocityZ[1]
        Object.VelocityTOFErr = 100 * abs((VelocityZ[1] - VelocityZ[2]) / (2 * VelocityZ[1]))

        ANST[1] = Object.NumberOfElectronsPlanes[2]
        ANST[2] = Object.NumberOfElectronsPlanes[3]
        ANST[3] = ANST[2] - sqrt(ANST[1]/ANST[2])
        ANST[4] = log(ANST[1]/ANST[2])
        ANST[5] = log(ANST[1]/ANST[3])
        ANST[6] = ANST[5]/ANST[4]
        ANST[7] = ANST[6]-1
        if Object.AttachmentOverIonisationPT==-1:
            Object.ReducedAlphaTOF = 0.0
            Object.ReducedAlphaTOFErr = 0.0
            Object.ReducedAttachmentTOF = -1 * Object.RealIonisation[1]
            Object.ReducedAttachmentTOFErr = 100 * sqrt(ANST[7] ** 2 + Object.AttachmentErrPT ** 2)
        else:
            Object.ReducedAlphaTOF = Object.RealIonisation[1] / (1 - Object.AttachmentOverIonisationPT)
            Object.ReducedAlphaTOFErr = 100 * sqrt(ANST[7] ** 2 + Object.AttachmentOverIonisationErrPT ** 2)
            Object.ReducedAttachmentTOF = Object.AttachmentOverIonisationPT * Object.RealIonisation[1] / (1.0 - Object.AttachmentOverIonisationPT)
            Object.ReducedAttachmentTOFErr = 100 * sqrt(ANST[7] ** 2 + Object.AttachmentErrPT ** 2)

    else:
        # For net ionisation take results from final plane

        I1 = Object.NumberOfTimeSteps
        Object.TOFEnergy = Object.EnergyPT[I1 - 1]
        Object.TOFEnergyErr = 100 * abs((Object.EnergyPT[I1 - 1] - Object.EnergyPT[I1 - 2]) / (2 * Object.EnergyPT[I1 - 1]))

        Object.VelocityTOFPT = Object.VelocityZPT[I1 - 1]
        Object.VelocityTOFPTErr = 100 * abs((Object.VelocityZPT[I1 - 1] - Object.VelocityZPT[I1 - 2]) / (2 * Object.VelocityZPT[I1 - 1]))

        Object.LongitudinalDiffusionTOF = LongitudinalDiffusion[I1 - 1]
        Object.LongitudinalDiffusionTOFErr = 100 * abs((LongitudinalDiffusion[I1 - 1] - LongitudinalDiffusion[I1 - 2]) / (2 * LongitudinalDiffusion[I1 - 1]))

        TransverseDiffusion2 = (DiffusionX[I1 - 1] + DiffusionY[I1 - 1]) / 2
        TransverseDiffusion3 = (DiffusionX[I1 - 2] + DiffusionY[I1 - 2]) / 2

        Object.TransverseDiffusionTOF = TransverseDiffusion2
        Object.TransverseDiffusionTOFErr = 100 * abs((TransverseDiffusion2 - TransverseDiffusion3) / (2 * TransverseDiffusion2))

        Object.VelocityTOF = VelocityZ[I1 - 1]
        Object.VelocityTOFErr = 100 * abs((VelocityZ[I1 - 1] - VelocityZ[I1 - 2]) / (2 * VelocityZ[I1 - 1]))

        ATER = abs((Object.RealIonisation[I1 - 1] - Object.RealIonisation[I1 - 2]) / (Object.RealIonisation[I1 - 1]))

        Object.ReducedAlphaTOF = Object.RealIonisation[I1 - 1] / (1.0 - Object.AttachmentOverIonisationPT)
        Object.ReducedAlphaTOFErr = 100 * sqrt(ATER ** 2 + Object.AttachmentOverIonisationErrPT ** 2)

        Object.ReducedAttachmentTOF = Object.AttachmentOverIonisationPT * Object.RealIonisation[I1 - 1] / (1.0 - Object.AttachmentOverIonisationPT)
        if Object.AttachmentOverIonisationPT!=0.0:
            Object.ReducedAttachmentTOFErr = 100 * sqrt(ATER ** 2 + Object.AttachmentErrPT ** 2)
        else:
            Object.ReducedAttachmentTOFErr = 0.0

from Gasmix cimport Gasmix
from Gases.XENONMERT cimport Gas61
cdef class Gasmix_mert(Gasmix):
    def Run(self):
        '''This functions calls the corresponding gas functions.'''
        cdef int i
        for i in range(6):
            Gas61(&self.Gases[i],self.A,self.D,self.F,self.A1,self.Lambda,self.EV0)

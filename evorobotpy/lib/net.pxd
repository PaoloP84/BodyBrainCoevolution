cdef extern from "utilities.cpp":
    pass

cdef extern from "evonet.cpp":
    pass

# Declare the class with cdef
cdef extern from "evonet.h":
    cdef cppclass Evonet:
        Evonet() except +
        Evonet(int, int, int, int, int, int, int, int, int, int, int, int, double, int, double, double) except +
        int ninputs, nhiddens, noutputs, nlayers, bias, netType, actFunct, outType, wInit, clip, normalize, randAct, wrange, nbins, low, high
        double* m_act
        double* m_netinput
        void resetNet()
        void seed(int s)
        void copyGenotype(double* genotype)
        void copyInput(float* input)
        void copyOutput(float* output)
        void copyNeuronact(double* na)
        void copyNormalization(double* no)
        void updateNet()
        void getOutput(double* output)
        int computeParameters()
        void initWeights()
        void normphase(int phase)
        void updateNormalizationVectors()
        void setNormalizationVectors()
        void resetNormalizationVectors()


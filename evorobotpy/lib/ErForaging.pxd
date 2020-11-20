cdef extern from "utilities.cpp":
    pass

cdef extern from "foraging.cpp":
    pass

# Declare the class with cdef
#cdef extern from "utilities.h":
    #cdef cppclass RandomGenerator:
        #RandomGenerator() except +
        #void setSeed(int seed)
        #int seed()
        #int getInt(int min, int max)
        #double getDouble(double min, double max)
        #double getGaussian(double var, double mean)

# Declare the class with cdef
cdef extern from "foraging.h":
    cdef cppclass Problem:
        Problem() except +
        int m_trial
        int ninputs
        int noutputs
        double* m_state
        double m_masspole_2, m_length_2
        void seed(int s)
        void reset()
        double step()
        void close()
        void render()
        double isDone()
        void copyObs(float* observation)
        void copyAct(float* action)
        void copyDone(double* done)
        void copyDobj(double* objs)


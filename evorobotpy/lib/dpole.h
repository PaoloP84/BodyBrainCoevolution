#ifndef PROBLEM_H
#define PROBLEM_H

#include "utilities.h"

// Double-pole constants
#define MASSPOLE_1 0.1
#define LENGTH_1 0.5
#define MUP 0.000002
#define MUC 0.0005
#define GRAVITY -9.8
#define MASSCART 1.0
#define FORCE_MAG 10.0
#define TAU 0.01
#define NUM_INPUTS 3
#define NUM_STATES 6
#define THIRTY_SIX_DEGREES 0.628329
#define TRACK_EDGE 2.4

class Problem
{

public:
    // Void constructor
    Problem();
    // Destructor
    ~Problem();
    // Set the seed
    void seed(int s);
    // Reset trial
    void reset();
    // Perform a step of the double-pole
    double step();
    // Close the environment
    void close();
    // View the environment (graphic mode)
    void render();
    // Copy the observations
    void copyObs(float* observation);
    // Copy the action
    void copyAct(float* action);
    // Copy the termination flag
    void copyDone(double* done);
    // Copy the pointer to the vector of objects to be displayed
    void copyDobj(double* objs);

    // State
    double* m_state;
    // Second-pole length
    double m_length_2;
    // Second-pole mass
    double m_masspole_2;
    // number of inputs
    int m_ninputs;
    // number of outputs
    int m_noutputs;
    // maximum value for action
    double m_high;
    // minimum value for action
    double m_low;

private:
    // Initialize
    void initEnvironment(char* filename);
    // Get observations
    void getObs();
    // Check if the task is terminated
    bool outsideBounds();
    // Step
    void doStep(double action, double *st, double *derivs);
    // RK4
    void rk4(float f, double *y, double *dydx, double *yout);
    // Perform the action
    void performAction(float output);
    // Read configuration file
    bool readConfig(char* filename);
    // Random generator
    RandomGenerator* m_rng;

};

#endif

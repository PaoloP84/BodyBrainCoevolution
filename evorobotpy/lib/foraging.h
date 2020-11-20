#ifndef PROBLEM_H
#define PROBLEM_H

#include "utilities.h"


class Problem
{

public:
	// Void constructor
	Problem();
	// Other constructor
	Problem(double length2, double masspole2, int fixed, int ntrials, int nttrials);
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
	// Check whether the episode terminated
	double isDone();


	// number of robots
	int nrobots;
	// pointer to robots
	struct robot *rob;
	// number of inputs
	int ninputs;
	// number of outputs
	int noutputs;

private:
	// create the environment
    void initEnvironment();
	void getObs();
	// Random generator
	RandomGenerator* rng;

};

#endif


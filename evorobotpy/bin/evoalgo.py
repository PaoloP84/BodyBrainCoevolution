#!/usr/bin/python

import numpy as np
import time

mode = 1

# average fitness of the samples
def averageFit(fitness):
    avef = 0.0
    for i in range(len(fitness)):
        avef = avef + fitness[i]
    avef = avef / len(fitness)
    return avef

class EvoAlgo(object):
    def __init__(self, env, policy, seed, filedir):
        # Copy the environment
        self.env = env
        # Copy the policy
        self.policy = policy
        # Copy the seed
        self.seed = seed
        # Copy the directory where files will be saved
        self.filedir = filedir
        # Set fitness initially to a very low value
        self.bestfit = -999999999.0
        # Initialize best solution found
        self.bestsol = None
        # Set generalization initially to a very low value
        self.bestgfit = -999999999.0
        # Initialize best generalizing solution found
        self.bestgsol = None
        # Initialize stat information
        self.stat = np.arange(0, dtype=np.float64)
        # Initialize average fitness
        self.avgfit = 0.0
        # Time from last save
        self.last_save_time = time.time()
        # Test mode
        self.testMode = None
        # List of centroids
        self.centers = []
        
    def reset(self):
         # Set fitness initially to a very low value
        self.bestfit = -999999999.0
        # Initialize best solution found
        self.bestsol = None
        # Set generalization initially to a very low value
        self.bestgfit = -999999999.0
        # Initialize best generalizing solution found
        self.bestgsol = None
        # Initialize stat information
        self.stat = np.arange(0, dtype=np.float64)
        # Initialize average fitness
        self.avgfit = 0.0
        # Time from last save
        self.last_save_time = time.time()
        # Test mode
        self.testMode = None
        # List of centroids
        self.centers = []

    # Set evolutionary variables like batchSize, step size, etc.
    def setEvoVars(self, sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter):
        self.batchSize = sampleSize
        self.stepsize = stepsize
        self.noiseStdDev = noiseStdDev
        self.sameenvcond = sameenvcond
        self.wdecay = wdecay

    def setTestMode(self, mode=None):
        self.testMode = mode

    def run(self, nevals):
        # Run method depends on the algorithm
        raise NotImplementedError

    def test(self, testfile):
        # Set the seed to enable the replication of the generalization test
        #self.policy.nn.seed(self.policy.seed + 1000000)
        # necessary initialization of the renderer
        if (self.policy.displayneurons == 0 and not "Er" in self.policy.environment):
            if mode == 0:
                self.env.render(mode="human")
            if mode == 1 and "Bullet" in self.policy.environment:
                import pybullet as pb
                if self.testMode is not None:
                    # Force mode test
                    pb.inTest(mode=self.testMode, amplitude=30.0, rate=50) # To be fixed!!!
                else:
                    # Post-evaluation test
                    pb.inTest()
                pb.render(mode="human")
        if testfile is not None:
            if self.filedir.endswith("/"):
                fname = self.filedir + testfile
            else:
                fname = self.filedir + "/" + testfile
            # Load the policy to be tested
            if (self.policy.normalize == 0):
                bestgeno = np.load(fname)
            else:
                geno = np.load(fname)
                for i in range(self.policy.ninputs * 2):
                    self.policy.normvector[i] = geno[self.policy.nparams + i]
                bestgeno = geno[0:self.policy.nparams]
                self.policy.nn.setNormalizationVectors()
            # Test the loaded individual
            self.policy.set_trainable_flat(bestgeno)
        else:
            self.policy.reset()
        # Test loaded individual
        if (self.policy.nttrials > 0):
            ntrials = self.policy.nttrials
        else:
            ntrials = self.policy.ntrials
        eval_rews, eval_length = self.policy.rollout(ntrials, render=True, seed=self.policy.get_seed + 100000, timestep_limit=1000)  # eval rollouts don't obey task_data.timestep_limit
        print("Test: steps %d reward %lf" % (eval_length, eval_rews))

    def updateBest(self, fit, ind):
        if fit > self.bestfit:
            self.bestfit = fit
            # in case of normalization store also the normalization vectors
            if (self.policy.normalize == 0):
                self.bestsol = np.copy(ind)
            else:
                self.bestsol = np.append(ind,self.policy.normvector)

    def updateBestg(self, fit, ind):
        if fit > self.bestgfit:
            self.bestgfit = fit
            # in case of normalization store also the normalization vectors
            if (self.policy.normalize == 0):
                self.bestgsol = np.copy(ind)
            else:
                self.bestgsol = np.append(ind,self.policy.normvector)
                
    # called at the end of every generation to display and store data
    def updateInfo(self, gen, steps, fitness, centroid, centroidfit, bestsam, elapsed, nevals):
        self.computeAvg(fitness)
        self.stat = np.append(self.stat, [steps, self.bestfit, self.bestgfit, self.avgfit, centroidfit, bestsam])
        print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f centroid %.2f bestsam %.2f avg %.2f weightsize %.2f' %
                      (self.seed, steps / float(nevals) * 100, gen, steps / 1000000, self.bestfit, self.bestgfit, centroidfit, bestsam, self.avgfit, np.average(np.absolute(centroid))))
        if (gen > 1) and ((time.time() - self.last_save_time) > (self.policy.saveeach * 60)):
            self.save(gen, steps, centroidfit, centroid, bestsam, elapsed)
            self.last_save_time = time.time()        

    def save(self, gen, steps, centroidfit, centroid, bestsam, elapsed):
            print('save data')
            # save best, bestg, and last centroid
            fname = self.filedir + "/bestS" + str(self.seed)
            np.save(fname, self.bestsol)
            fname = self.filedir + "/bestgS" + str(self.seed)
            np.save(fname, self.bestgsol)
            fname = self.filedir + "/centroidS" + str(self.seed)
            if (self.policy.normalize == 0):
                np.save(fname, centroid)
            else:
                np.save(fname, np.append(centroid,self.policy.normvector))
            # Save the list of centroids
            fname = self.filedir + "/centroidsS" + str(self.seed)
            np.save(fname, self.centers)
            # save statistics
            fname = self.filedir + "/statS" + str(self.seed)
            np.save(fname, self.stat)
            # save summary statistics
            fname = self.filedir + "/S" + str(self.seed) + ".fit"
            fp = open(fname, "w")
            fp.write('Seed %d gen %d eval %d bestfit %.2f bestgfit %.2f centroid %.2f bestsam %.2f avg %.2f weightsize %.2f runtime %.2f\n' % (self.seed, gen, steps, self.bestfit, self.bestgfit, centroidfit, bestsam, self.avgfit, np.average(np.absolute(centroid)), elapsed))
            fp.close()

    def computeAvg(self, fitness):
        self.avgfit = averageFit(fitness)


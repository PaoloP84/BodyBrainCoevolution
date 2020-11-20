#!/usr/bin/python
# stefano, help only if no parameters

# Libraries to be imported

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort

mode = 1

# Evolve with ES algorithm taken from Salimans et al. (2017)
class Salimans(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        center = self.policy.get_trainable_flat()

        # Add center to the list of centers
        self.centers = np.append(self.centers, center)
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        batchSize = self.batchSize
        if batchSize == 0:
            # 4 + floor(3 * log(N))
            batchSize = int(4 + math.floor(3 * math.log(nparams)))
        # Symmetric weights in the range [-0.5,0.5]
        weights = zeros(batchSize)

        ceval = 0                    # current evaluation
        cgen = 0                # current generation
        # Parameters for Adam policy
        m = zeros(nparams)
        v = zeros(nparams)
        epsilon = 1e-08 # To avoid numerical issues with division by zero...
        beta1 = 0.9
        beta2 = 0.999
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        rs = np.random.RandomState(self.seed)

        print("Salimans: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.sameenvcond, nparams))

        # Change morphology flag
        changed = False

        # Current seed
        cseed = None

        # Steps run before storing a new centroid
        runsteps = 100000
        # Number of centroids stored
        ncenters = 1

        # main loop
        elapsed = 0
        while (ceval < maxsteps):
            cgen += 1

            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = rs.randn(batchSize, nparams)
            # We generate simmetric variations for the offspring
            candidate = np.arange(nparams, dtype=np.float64)
            # Evaluate offspring
            fitness = zeros(batchSize * 2)
            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            cseed = self.policy.get_seed + cgen
            if mode == 0:
                self.env.seed(cseed)
            if mode == 1 and "Bullet" in self.policy.environment:
                import pybullet as pb
                pb.seed(seed=cseed)
            self.policy.nn.seed(cseed)
            # Evaluate offspring
            for b in range(batchSize):
                for bb in range(2):
                    if (bb == 0):
                        for g in range(nparams):
                            candidate[g] = center[g] + samples[b,g] * self.noiseStdDev
                    else:
                        for g in range(nparams):
                            candidate[g] = center[g] - samples[b,g] * self.noiseStdDev
                    # Set policy parameters 
                    self.policy.set_trainable_flat(candidate) 
                    # Sample of the same generation experience the same environmental conditions
                    if (self.sameenvcond == 1):
                        if mode == 0:
                            self.env.seed(cseed)#self.policy.get_seed + cgen)
                        if mode == 1 and "Bullet" in self.policy.environment:
                            import pybullet as pb
                            pb.seed(seed=cseed)
                        self.policy.nn.seed(cseed)
                    # Evaluate the offspring
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, timestep_limit=1000)
                    # Get the fitness
                    fitness[b*2+bb] = eval_rews
                    # Update the number of evaluations
                    ceval += eval_length
                    # Update data if the current offspring is better than current best
                    self.updateBest(fitness[b*2+bb], candidate)

            # Sort by fitness and compute weighted mean into center
            fitness, index = ascendent_sort(fitness)
            # Now me must compute the symmetric weights in the range [-0.5,0.5]
            utilities = zeros(batchSize * 2)
            for i in range(batchSize * 2):
                utilities[index[i]] = i
            utilities /= (batchSize * 2 - 1)
            utilities -= 0.5
            # Now we assign the weights to the samples
            for i in range(batchSize):
                idx = 2 * i
                weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

            # Evaluate the centroid
            if (self.sameenvcond == 1):
                if mode == 0:
                    self.env.seed(cseed)#self.policy.get_seed + cgen)
                if mode == 1 and "Bullet" in self.policy.environment:
                    import pybullet as pb
                    pb.seed(seed=cseed)
                self.policy.nn.seed(cseed)
            self.policy.set_trainable_flat(center)
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, timestep_limit=1000)
            centroidfit = eval_rews
            ceval += eval_length
            # Update data if the centroid is better than current best
            self.updateBest(centroidfit, center)

            # Evaluate generalization
            if (self.policy.nttrials > 0):
                if centroidfit > fitness[batchSize * 2 - 1]:
                    # the centroid is tested for generalization
                    candidate = np.copy(center)
                else:
                    # the best sample is tested for generalization
                    bestsamid = index[batchSize * 2 - 1]
                    if ((bestsamid % 2) == 0):
                        bestid = int(bestsamid / 2)
                        for g in range(nparams):
                            candidate[g] = center[g] + samples[bestid][g] * self.noiseStdDev
                    else:
                        bestid = int(bestsamid / 2)
                        for g in range(nparams):
                            candidate[g] = center[g] - samples[bestid][g] * self.noiseStdDev

                cseed = self.policy.get_seed + 100000

                if mode == 0:
                    self.env.seed(cseed)
                if mode == 1 and "Bullet" in self.policy.environment:
                    import pybullet as pb
                    pb.seed(seed=cseed)
                self.policy.nn.seed(cseed)
                self.policy.set_trainable_flat(candidate) 
                eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, timestep_limit=1000)
                gfit = eval_rews
                ceval += eval_length
                # eveltually store the new best generalization individual
                self.updateBestg(gfit, candidate)

            # Compute the gradient
            g = 0.0
            i = 0
            while i < batchSize:
                gsize = -1
                if batchSize - i < 500:
                    gsize = batchSize - i
                else:
                    gsize = 500
                g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                i += gsize
            # Normalization over the number of samples
            g /= (batchSize * 2)
            # Weight decay
            if (self.wdecay == 1):
                globalg = -g + 0.005 * center
            else:
                globalg = -g
            # ADAM policy
            # Compute how much the center moves
            a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            m = beta1 * m + (1.0 - beta1) * globalg
            v = beta2 * v + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * m / (sqrt(v) + epsilon)
            # update center
            center += dCenter

            # Add center to the list of centers
            if ceval >= runsteps:
                self.centers = np.append(self.centers, center)
                ncenters += 1
                print("Seed %d steps %d vs %d ncenters %d" % (self.seed, ceval, runsteps, ncenters))
                runsteps += 100000

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, center, centroidfit, fitness[batchSize * 2 - 1], elapsed, maxsteps)

            if self.policy.warmup == 1 and self.policy.changeMorphology == 1:
                if ceval >= int(maxsteps * self.policy.warmupLength) and not changed:
                    # Check whether or not morphology of the robot needs to be changed
                    if mode == 1 and "Bullet" in self.policy.environment:
                        import pybullet as pb
                        pb.changeMorphology()
                        changed = True

        # save data
        self.save(cgen, ceval, centroidfit, center, fitness[batchSize * 2 - 1], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))


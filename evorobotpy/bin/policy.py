#!/usr/bin/python

"""

   This class implements the policy
   i.e. the agent's neural network controller
   Require the net.so library obtained by compiling with cython: evonet.cpp, evonet.h, net.pxd and net.pyx 

"""

import numpy as np
import net
import configparser
import time
import sys
import os
import pybullet_data
import renderWorld

mode = 1

renderon = True
try:
    import pyglet
except ImportError:
    renderon = False
    print("Warning: renderWorld unavailable. It requires pyglet")
    pass
          
if (renderon):
    import renderWorld

class Policy(object):
    def __init__(self, environment, env, ninputs, noutputs, low, high, ob, ac, filename, seed):
        # Copy environment
        self.environment = environment
        self.env = env
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.ninputs = ninputs 
        self.noutputs = noutputs
        # Initialize parameters to default values
        self.ntrials = 1     # evaluation trials
        self.nttrials = 0    # post-evaluation trials
        self.maxsteps = 1000 # max number of steps (used from ERPolicy only)
        self.nhiddens = 50   # number of hiddens
        self.nlayers = 1     # number of hidden layers 
        self.bias = 0        # whether we have biases
        self.out_type = 2    # output type (1=logistic,2=tanh,3=linear,4=binary)
        self.architecture =0 # Feed-forward, recurrent, or full-recurrent network
        self.afunction = 2   # activation function
        self.nbins = 1       # number of bins 1=no-beans
        self.winit = 0       # weight initialization: Xavier, normc, uniform
        self.action_noise = 0# noise applied to actions
        self.normalize = 0   # Do not normalize observations
        self.clip = 0        # clip observation
        self.displayneurons=0# Gym policies can display or the robot or the neurons activations
        self.wrange = 1.0    # weight range, used in uniform initialization only
        self.taskId = 0      # identifier of the task (to be passed to pybullet)
        self.penalty = 0     # penalty (to be passed to pybullet)
        self.penaltyValue = 0.0# Penalty value
        self.warmup = 0      # warm-up phase flag
        self.warmupLength = 0.5# Length of warm-up phase (0.5 --> 50%)
        self.changeMorphology=0# Change morphology flag
        self.changeRate = 0.0# Percentage of change (i.e., increase/decrease) of morphology
        self.paramNoise = 0.0# Noise on additional parameters
        self.fixedSize = 0   # Flags whether or not we keep robot's body sizes fixed
        self.fixedRange = 0  # Flags whether or not we keep robot's joint ranges fixed
        self.fixedVolume = 0 # Flags whether or not we keep robot's body volumes fixed
        self.motorCoeff = 1.0# coefficient of the motor
        self.spring = 0      # whether joints have springs
        self.energy = 0      # Maximum energy produceable by the robot
        self.energyCoeff = 1.0# Energy coefficient (for energy penalty)
        self.energyMode = 0  # Energy calculation mode (0: based on actions (average squared), 1: based on torques (sum of abs))
        self.nullTorque = 0  # Whether or not we set null torques if applied to joints at limit
        self.limitMotors = 0 # Flags whether or not we limit the motor coeffs
        # Read configuration file
        self.readConfig(filename)
        # Display info
        print("Evaluation: Episodes %d Test Episodes %d MaxSteps %d" % (self.ntrials, self.nttrials, self.maxsteps))
        # Initialize the neural network
        self.nn = net.PyEvonet(self.ninputs, (self.nhiddens * self.nlayers), self.noutputs, self.nlayers, self.bias, self.architecture, self.afunction, self.out_type, self.winit, self.clip, self.normalize, self.action_noise, self.wrange, self.nbins, low, high)
        # Initialize policy parameters
        self.nparams = self.nn.computeParameters()
        self.netnparams = self.nparams
        self.naddparams = 0
        if mode == 1 and "Bullet" in self.environment:
            if self.changeMorphology == 1:
                # We compute additional parameters for:
                # - Halfcheetah
                # - Walker2D
                # - Embryo
                # - Embryo_hard
                # - Embryo_challenging
                if self.taskId == 2:
                    # Additional parameters
                    self.naddparams = 34 # 8 body sizes + 1 torso length + 1 head length + 6 * 2 body positions + 6 * 2 joint ranges
                if self.taskId == 3:
                    # Additional parameters (walker must be symmetric)
                    self.naddparams = 14 # 4 body sizes + 4 body lengths + 3 * 2 joint ranges
                if self.taskId == 6:
                    # Additional parameters
                    self.naddparams = 32 # 7 body sizes + 7 body lengths + 6 rotation angles + 6 * 2 joint ranges
                if self.taskId == 7:
                    # Additional parameters
                    self.naddparams = 62 # 13 body sizes + 13 body lengths + 12 rotation angles + 12 * 2 joint ranges
                if self.taskId == 8:
                    # Additional parameters
                    self.naddparams = 122 # 25 body sizes + 25 body lengths + 24 rotation angles + 24 * 2 joint ranges
            if self.limitMotors == 1:
                self.naddparams += 6 # 6 params to limit 6 joint motors
            if self.spring == 1:
                self.naddparams += 6 # 6 springs related to 6 joints
            self.nparams += self.naddparams
        self.params = np.arange(self.nparams, dtype=np.float64)
        # Initialize normalization vector
        if (self.normalize == 1):
            self.normvector = np.arange(self.ninputs*2, dtype=np.float64)
        else:
            self.normvector = None
        # allocate neuron activation vector
        if (self.nbins == 1):
            self.nact = np.arange(self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs, dtype=np.float64)
        else:
            self.nact = np.arange(self.ninputs + (self.nhiddens * self.nlayers) + (self.noutputs * self.nbins), dtype=np.float64)            
        # Allocate space for observation and action
        self.ob = ob
        self.ac = ac
        # Copy pointers
        self.nn.copyGenotype(self.params)
        self.nn.copyNeuronact(self.nact)
        if (self.normalize == 1):
            self.nn.copyNormalization(self.normvector)
        # Initialize weights
        self.nn.seed(self.seed)
        self.nn.initWeights()
        if mode == 0:
            self.nn.copyInput(self.ob)
            self.nn.copyOutput(self.ac)
        # Copy pointers to pybullet in case of pybullet environment
        if mode == 1 and "Bullet" in self.environment:
            import pybullet as pb
            self.env.seed(self.seed)
            # Initialize task
            pb.init(pybullet_data_path=pybullet_data.getDataPath(), task=self.taskId, changeMorphology=self.changeMorphology, changeRate=self.changeRate, noise=self.paramNoise, fixedSize=self.fixedSize, fixedRange=self.fixedRange, fixedVolume=self.fixedVolume, penalty=self.penalty, penaltyValue=self.penaltyValue, motorCoeff=self.motorCoeff, spring=self.spring, warmup=self.warmup, energy=self.energy, energyCoeff=self.energyCoeff, energyMode=self.energyMode, nullTorque=self.nullTorque, limitMotors=self.limitMotors)
            # Initialize the seed
            pb.seed(seed=self.seed)
            # Start new run
            pb.start()
            # Allocate pointers for pybullet.c
            self.obp = ob
            self.acp = ac
            # Copy pointers to pybullet.c
            pb.copyObs(self.obp)
            pb.copyActs(self.acp)
            pb.copyParams(params=self.params[self.netnparams:], nparams=self.naddparams) # We copy only the parameters affecting morphology
            # Copy pointers to evonet
            self.nn.copyInput(self.obp)
            self.nn.copyOutput(self.acp)
            # Initialize additional parameters
            for i in range(self.naddparams):
                self.params[self.netnparams + i] = 0.0
            # Print POLICY params
            print("POLICY: task %d, changeMorphology %d, changeRate %.2f, warmup %d, motorCoeff %.2f, maxEnergy %.2f, spring %d --> nparams %d, naddparams %d" % (self.taskId, self.changeMorphology, self.changeRate, self.warmup, self.motorCoeff, self.energy, self.spring, self.nparams, self.naddparams))

    def reset(self):
        self.nn.seed(self.seed)
        self.nn.initWeights()
        if (self.normalize == 1):
            self.nn.resetNormalizationVectors()
        if mode == 1 and "Bullet" in self.environment:
            import pybullet as pb
            pb.seed(seed=self.seed)
            pb.start()

    # virtual function, implemented in sub-classes
    def rollout(self, render=False, timestep_limit=None, seed=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self.params = np.copy(x)
        self.nn.copyGenotype(self.params)
        if mode == 1 and "Bullet" in self.environment:
            import pybullet as pb
            if self.changeMorphology == 1:
                pb.copyParams(params=self.params[self.netnparams:], nparams=self.naddparams)

    def get_trainable_flat(self):
        return self.params

    def readConfig(self, filename):
        # parse the [POLICY] section of the configuration file
        config = configparser.ConfigParser()
        config.read(filename)
        options = config.options("POLICY")
        for o in options:
          found = 0
          if (o == "ntrials"):
              self.ntrials = config.getint("POLICY","ntrials")
              found = 1
          if (o == "nttrials"):
              self.nttrials = config.getint("POLICY","nttrials")
              found = 1
          if (o == "maxsteps"):
              self.maxsteps = config.getint("POLICY","maxsteps")
              found = 1
          if (o == "nhiddens"):
              self.nhiddens = config.getint("POLICY","nhiddens")
              found = 1
          if (o == "nlayers"):
              self.nlayers = config.getint("POLICY","nlayers")
              found = 1
          if (o == "bias"):
              self.bias = config.getint("POLICY","bias")
              found = 1
          if (o == "out_type"):
              self.out_type = config.getint("POLICY","out_type")
              found = 1
          if (o == "nbins"):
              self.nbins = config.getint("POLICY","nbins")
              found = 1
          if (o == "afunction"):
              self.afunction = config.getint("POLICY","afunction")
              found = 1
          if (o == "architecture"):
              self.architecture = config.getint("POLICY","architecture")
              found = 1
          if (o == "winit"):
              self.winit = config.getint("POLICY","winit")
              found = 1
          if (o == "action_noise"):
              self.action_noise = config.getint("POLICY","action_noise")
              found = 1
          if (o == "normalize"):
              self.normalize = config.getint("POLICY","normalize")
              found = 1
          if (o == "clip"):
              self.clip = config.getint("POLICY","clip")
              found = 1
          if (o == "wrange"):
              self.wrange = config.getint("POLICY","wrange")
              found = 1
          if (o == "taskid"):
              self.taskId = config.getint("POLICY","taskid")
              found = 1
          if (o == "penalty"):
              self.penalty = config.getint("POLICY","penalty")
              found = 1
          if (o == "penaltyvalue"):
              self.penaltyValue = config.getfloat("POLICY","penaltyvalue")
              found = 1
          if (o == "warmup"):
              self.warmup = config.getint("POLICY","warmup")
              found = 1
          if (o == "warmuplength"):
              self.warmupLength = config.getfloat("POLICY","warmuplength")
              found = 1
          if (o == "changemorphology"):
              self.changeMorphology = config.getint("POLICY","changemorphology")
              found = 1
          if (o == "changerate"):
              self.changeRate = config.getfloat("POLICY","changerate")
              found = 1
          if (o == "noise"):
              self.paramNoise = config.getfloat("POLICY","noise")
              found = 1
          if (o == "fixedsize"):
              self.fixedSize = config.getint("POLICY","fixedsize")
              found = 1
          if (o == "fixedrange"):
              self.fixedRange = config.getint("POLICY","fixedrange")
              found = 1
          if (o == "fixedvolume"):
              self.fixedVolume = config.getint("POLICY","fixedvolume")
              found = 1
          if (o == "motorcoeff"):
              self.motorCoeff = config.getfloat("POLICY","motorcoeff")
              found = 1
          if (o == "spring"):
              self.spring = config.getint("POLICY","spring")
              found = 1
          if (o == "energy"):
              self.energy = config.getfloat("POLICY","energy")
              found = 1
          if (o == "energycoeff"):
              self.energyCoeff = config.getfloat("POLICY","energycoeff")
              found = 1
          if (o == "energymode"):
              self.energyMode = config.getint("POLICY","energymode")
              found = 1
          if (o == "nulltorque"):
              self.nullTorque = config.getint("POLICY","nulltorque")
              found = 1
          if (o == "limitmotors"):
              self.limitMotors = config.getint("POLICY","limitmotors")
              found = 1
          if (found == 0):
              print("\033[1mOption %s in section [POLICY] of %s file is unknown\033[0m" % (o, filename))
              sys.exit()

    def loadParams(self, filename):
        # Load parameters from file
        if mode == 1 and "Bullet" in self.environment:
            import pybullet as pb
            # Load the genotype
            geno = np.load(filename)
            # Extract morphology parameters
            params = geno[self.nparams:] # We get additional parameters + normalization data
            # Additional parameters
            naddparams = 0
            if self.taskId == 2:
                # Halfcheetah
                naddparams = 34
            if self.taskId == 3:
                # Walker2D
                naddparams = 14
            if self.taskId == 6:
                # Embryo
                naddparams = 32
            if self.taskId == 7:
                # Embryo_hard
                naddparams = 62
            if self.taskId == 8:
                # Embryo_challenging
                naddparams = 122
            params = params[0:naddparams]
            nparams = len(params)
            if nparams == 0:
                print("FATAL ERROR: attempting to load parameters from file %s without additional parameters (i.e., fixed morphology)... Stop now!!!" % filename)
            # Evolution with parameters loaded from file
            pb.runWithParams(params=params, nparams=nparams)

    @property
    def get_seed(self):
        return self.seed

class GymPolicy(Policy):
    def __init__(self, environment, env, ninputs, noutputs, low, high, ob, ac, filename, seed):
        Policy.__init__(self, environment, env, ninputs, noutputs, low, high, ob, ac, filename, seed)
        # we allocate the vector containing the objects to be rendered by the 2D-Renderer
        self.objs = np.arange(10, dtype=np.float64) # DEBUG SIZE TO BE FIXED
        self.objs[0] = -1                             # to indicate that as default the list contains no objects
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = 1000
        timestep_limit = env_timestep_limit if timestep_limit is None else timestep_limit#min(timestep_limit, env_timestep_limit)
        rews = 0.0
        steps = 0
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            if mode == 0:
                self.ob = self.env.reset()
            if mode == 1 and "Bullet" in self.environment:
                import pybullet as pb
                pb.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < timestep_limit:
                # Copy the input in the network
                if mode == 0:
                    self.nn.copyInput(self.ob)
                # Activate network
                self.nn.updateNet()
                # Perform a step
                if mode == 0:
                    self.ob, rew, done, _ = self.env.step(self.ac)
                if mode == 1 and "Bullet" in self.environment:
                    rew = pb.step()
                    done = pb.isDone()
                
                # Append the reward
                crew += rew
                t += 1
                if render:
                    if (self.displayneurons == 0):
                        if mode == 0:
                            self.env.render(mode="human")
                        if mode == 1 and "Bullet" in self.environment:
                            pb.render(mode="human")
                        time.sleep(0.05)
                    else:
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, crew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact[self.ninputs:len(self.nact)-self.noutputs])
                    if done or t == timestep_limit:
                        print("Trial %d Fit %.2f Steps %d " % (trial, crew, t)) 
                if done:
                    break
            # In case of verbosity, print some information at the end of trial
            if mode == 1 and "Bullet" in self.environment:
                pb.postTrial(timesteps=t)
            rews += crew
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps

class ErPolicy(Policy):
    def __init__(self, environment, env, ninputs, noutputs, low, high, ob, ac, done, filename, seed):
        Policy.__init__(self, environment, env, ninputs, noutputs, low, high, ob, ac, filename, seed)
        self.done = done
        # we allocate the vector containing the objects to be rendered by the 2D-Renderer
        self.objs = np.arange(1000, dtype=np.float64) # DEBUG SIZE TO BE FIXED
        self.objs[0] = -1                             # to indicate that as default the list contains no objects
        self.env.copyDobj(self.objs)
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        rews = 0.0
        steps = 0
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < self.maxsteps:
                # Activate network
                self.nn.updateNet()
                # Perform a step
                rew = self.env.step()
                # Append the reward
                rews += rew
                t += 1
                if render:
                    self.env.render()
                    info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                    renderWorld.update(self.objs, info, self.ob, self.ac, self.nact[self.ninputs:len(self.nact)-self.noutputs])
                    if self.done or t == self.maxsteps:
                        print("Trial %d Fit %d Steps %d " % (trial, rews, t)) 
                if self.done:
                    break
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps
        

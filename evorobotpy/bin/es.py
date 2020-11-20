#!/usr/bin/python

"""

 Run evolutionary experimemts
 run it without parameters for help on usage
 
"""

import numpy as np
import configparser
import sys
import os


# global variables
scriptdirname = os.path.dirname(os.path.realpath(__file__))  # Directory of the script .py
#sys.path.insert(0, scriptdirname) # add the diretcory to the path
#cwd = os.getcwd() # directoy from which the script has been lanched
#sys.path.insert(0, cwd) add the directory to the path
filedir = None                          # Directory used to save files
center = None                           # the solution center
sample = None                           # the solution samples
environment = None                      # the problem 
stepsize = 0.01                         # the learning stepsize
noiseStdDev = 0.02                      # the perturbation noise
sampleSize = 20                         # number of samples
wdecay = 0                              # wether we usse weight decay
sameenvcond = 0                         # whether population individuals experience the same conditions
maxsteps = 1000000                      # total max number of steps
evalCenter = 1                          # whether we evaluate the solution center
saveeach = 60                           # number of seconds after which we save data


# Parse the [ADAPT] section of the configuration file
def parseConfigFile(filename):
    global maxsteps
    global envChangeEvery
    global environment
    global fullyRandom
    global stepsize
    global noiseStdDev
    global sampleSize
    global wdecay
    global sameenvcond
    global evalCenter
    global saveeach

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section EVAL
        options = config.options("ADAPT")
        for o in options:
            found = 0
            if o == "maxmsteps":
                maxsteps = config.getint("ADAPT","maxmsteps") * 1000000
                found = 1
            if o == "environment":
                environment = config.get("ADAPT","environment")
                found = 1
            if o == "stepsize":
                stepsize = config.getfloat("ADAPT","stepsize")
                found = 1
            if o == "noisestddev":
                noiseStdDev = config.getfloat("ADAPT","noiseStdDev")
                found = 1
            if o == "samplesize":
                sampleSize = config.getint("ADAPT","sampleSize")
                found = 1
            if o == "wdecay":
                wdecay = config.getint("ADAPT","wdecay")
                found = 1
            if o == "sameenvcond":
                sameenvcond = config.getint("ADAPT","sameenvcond")
                found = 1
            if o == "evalcenter":
                evalCenter = config.getint("ADAPT","evalcenter")
                found = 1
            if o == "saveeach":
                saveeach = config.getint("ADAPT","saveeach")
                found = 1
              
            if found == 0:
                print("\033[1mOption %s in section [ADAPT] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()
    else:
        print("\033[1mERROR: configuration file %s does not exist\033[0m" % (filename))
        sys.exit()

def helper():
    print("Main()")
    print("Program Arguments: ")
    print("-f [filename]             : the file containing the parameters shown below (mandatory)")
    print("-s [integer]              : the number used to initialize the seed")
    print("-n [integer]              : the number of replications to be run")
    print("-a [algorithm]            : the algorithm: CMAES, Salimans, xNES, sNES, or SSS (default Salimans)")
    print("-t [filename]             : the .npy file containing the policy to be tested")
    print("-T [filename]             : the .npy file containing the policy to be tested, display neurons")    
    print("-d [directory]            : the directory where all output files are stored (default current dir)")
    print("-tf                       : use tensorflow policy (valid only for gym and pybullet")
    print("")
    print("The .ini file contains the following [ADAPT] and [POLICY] parameters:")
    print("[ADAPT]")
    print("environment [string]      : environment (default 'CartPole-v0'")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("sampleSize [integer]      : number of samples (default 20)")
    print("stepsize [float]          : learning stepsize (default 0.01)")
    print("noiseStdDev [float]       : samples noise (default 0.02)")
    print("wdecay [0/1]              : weight decay (defaul 0)")
    print("sameenvcond [0/1]         : samples experience the same environmental conditions")
    print("evalCenter [0/1]          : whether or not centroid is evaluated (default 1)")
    print("saveeach [integer]        : save data each n minutes (default 60)")
    print("[POLICY]")
    print("ntrials [integer]         : number of evaluation episodes (default 1)")
    print("nttrials [integer]        : number of post-evaluation episodes (default 0)")
    print("maxsteps [integer]        : number of evaluation steps [for EREnvs only] (default 1000)")
    print("nhiddens [integer]        : number of hidden x layer (default 50)")
    print("nlayers [integer]         : number of hidden layers (default 1)")
    print("bias [0/1]                : whether we have biases (default 0)")
    print("out_type [integer]        : type of output: 1=logistic, 2=tanh, 3=linear, 4=binary (default 2)")
    print("nbins [integer]           : number of bins 1=no-beans (default 1)")
    print("architecture [0/1/2]      : network architecture 0=feedforward 1=recurrent 2=fullrecurrent (default 0)")
    print("afunction [1/2]           : the activation function of neurons 1=logistic 2=tanh (default 2)")
    print("winit [0/1/2]             : weight initialization 0=xavier 1=norm incoming 2=uniform (default 0)")
    print("action_noise [0/1]        : noise applied to actions (default 1)")
    print("normalized [0/1]          : whether or not the input observations are normalized (default 1)")
    print("clip [0/1]                : whether we clip observation in [-5,5] (default 0)")
    print("")
    sys.exit()


def main(argv):
    global maxsteps
    global environment
    global filedir
    global saveeach

    argc = len(argv)

    # if called without parameters display help information
    if (argc == 1):
        helper()
        sys.exit(-1)

    # Default parameters:
    filename = None         # configuration file
    cseed = 1               # seed
    nreplications = 1       # nreplications
    algorithm = "Salimans"  # algorithm 
    filedir = './'          # directory
    testfile = None         # file containing the policy to be tested
    test = False            # whether we want to test a policy
    testMode = None         # test mode
    displayneurons = 0      # whether we want to display the activation state of the neurons
    useTf = False           # whether we want to use tensorflow to implement the policy
    paramsfile = None       # parameters file (to run evolution with evolved parameters)
    
    i = 1
    while (i < argc):
        if (argv[i] == "-f"):
            i += 1
            if (i < argc):
                filename = argv[i]
                i += 1
        elif (argv[i] == "-s"):
            i += 1
            if (i < argc):
                cseed = int(argv[i])
                i += 1
        elif (argv[i] == "-n"):
            i += 1
            if (i < argc):
                nreplications = int(argv[i])
                i += 1
        elif (argv[i] == "-a"):
            i += 1
            if (i < argc):
                algorithm = argv[i]
                i += 1
        elif (argv[i] == "-t"):
            i += 1
            test = True
            if (i < argc):
                testfile = argv[i]
                i += 1
        elif (argv[i] == "-T"):
            i += 1
            test = True
            displayneurons = 1
            if (i < argc):
                testfile = argv[i]
                i += 1   
        elif (argv[i] == "-d"):
            i += 1
            if (i < argc):
                filedir = argv[i]
                i += 1
        elif (argv[i] == "-tf"):
            i += 1
            useTf = True
        elif (argv[i] == "-m"):
            i += 1
            if (i < argc):
                testMode = argv[i]
                i += 1
        elif (argv[i] == "-F"):
            i += 1
            if (i < argc):
                paramsfile = argv[i]
                i += 1
        else:
            # We simply ignore the argument
            print("\033[1mWARNING: unrecognized argument %s \033[0m" % argv[i])
            i += 1

    # load the .ini file
    if filename is not None:
        parseConfigFile(filename)
    else:
        print("\033[1mERROR: You need to specify an .ini file\033[0m" % filename)
        sys.exit(-1)
    # if a directory is not specified, we use the current directory
    if filedir is None:
        filedir = scriptdirname

    # check whether the user specified a valid algorithm
    availableAlgos = ('CMAES','Salimans','xNES', 'sNES','SSS')
    if algorithm not in availableAlgos:
        print("\033[1mAlgorithm %s is unknown\033[0m" % algorithm)
        print("Please use one of the following algorithms:")
        for a in availableAlgos:
            print("%s" % a)
        sys.exit(-1)

    print("Environment %s nreplications %d maxmsteps %dm " % (environment, nreplications, maxsteps / 1000000))
    env = None
    policy = None
    
    # Evorobot Environments 
    if "Er" in environment:
        ErProblem = __import__(environment)
        env = ErProblem.PyErProblem()
        # Create a new doublepole object
        #action_space = spaces.Box(-1., 1., shape=(env.noutputs,), dtype='float32')
        #observation_space = spaces.Box(-np.inf, np.inf, shape=(env.ninputs,), dtype='float32')
        ob = np.arange(env.ninputs, dtype=np.float32)
        ac = np.arange(env.noutputs, dtype=np.float32)
        done = np.arange(1, dtype=np.float64)
        env.copyObs(ob)
        env.copyAct(ac)
        env.copyDone(done)
        if useTf:
            # tensorflow policy
            from policyt import ErPolicyTf, GymPolicyTf, make_session
            if algorithm == "Salimans":
                size = sampleSize * 2
            else:
                size = sampleSize
            session = make_session(single_threaded=True)
            policy = ErPolicyTf(env, env.ninputs, env.noutputs, env.low, env.high, size, ob, ac, done, filename, cseed)
            policy.initTfVars() 
            if policy.normalize == 1:
                policy.initStat()
        else:       
            # evonet policy
            from policy import ErPolicy
            policy = ErPolicy(environment, env, env.ninputs, env.noutputs, env.low, env.high, ob, ac, done, filename, cseed)
    else:
        # Gym (or pybullet) environment
        import gym
        from gym import spaces
        if "Bullet" in environment:
            import pybullet
            import pybullet_envs
        env = gym.make(environment)
        if useTf:
            from policyt import ErPolicyTf, GymPolicyTf, make_session
            if algorithm == "Salimans":
                size = sampleSize * 2
            else:
                size = sampleSize
            # Create a new session
            session = make_session(single_threaded=True)
            # Use policy with Tensorflow
            policy = GymPolicyTf(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], size, filename, cseed)
            # Initialize tensorflow variables
            policy.initTfVars()
            # Initialize stat
            if policy.normalize == 1:
                policy.initStat()
        else:
            from policy import GymPolicy
            # Define the objects required (they depend on the environment)
            ob = np.arange(env.observation_space.shape[0], dtype=np.float32)
            ac = np.arange(env.action_space.shape[0], dtype=np.float32)
            # Define the policy
            policy = GymPolicy(environment, env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed)

    policy.environment = environment
    policy.saveeach = saveeach
    
    # Create the algorithm class
    if (algorithm == 'CMAES'):
        from cmaes import CMAES
        algo = CMAES(env, policy, cseed, filedir)
    elif (algorithm =='Salimans'):
        from salimans import Salimans
        algo = Salimans(env, policy, cseed, filedir)
    elif (algorithm == 'xNES'):
        from xnes import xNES
        algo = xNES(env, policy, cseed, filedir)
    elif (algorithm == 'sNES'):
        from snes import sNES
        algo = sNES(env, policy, cseed, filedir)
    elif (algorithm == 'SSS'):
        from sss import SSS
        algo = SSS(env, policy, cseed, filedir)
    # Set evolutionary variables
    algo.setEvoVars(sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter)

    if paramsfile is not None:
        policy.loadParams(paramsfile)

    if (test):
        # test a policy
        print("Run Test: Environment %s testfile %s" % (environment, testfile))
        policy.displayneurons = displayneurons
        # Set testMode if any
        if testMode is not None:
            algo.setTestMode(testMode)
        algo.test(testfile)
    else:
        # run evolution
        if (cseed != 0):
            print("Run Evolve: Environment %s Seed %d Nreplications %d" % (environment, cseed, nreplications))
            for r in range(nreplications):
                algo.run(maxsteps)
                algo.seed += 1
                policy.seed += 1
                algo.reset()
                policy.reset()
        else:
            print("\033[1mPlease indicate the seed to run evolution\033[0m")

if __name__ == "__main__":
    main(sys.argv)

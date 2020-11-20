//stefano, modificato print info, eliminata la funzione che legge il file .ini, aggiunta funzione set seed
// stefano. temporary modified range uniform weights from 0.01 to 0.1
// stefano: added pointer to activation vector and copy function
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "evonet.h"
#include "utilities.h"

#define MAX_BLOCKS 20
#define MAXN 10000
#define CLIP_VALUE 5.0

// Local random number generator
RandomGenerator* netRng;

// Pointers to genotype, observation, action, neuron activation, normalization vectors
double *cgenotype = NULL;
float *cobservation = NULL;
float *caction = NULL;
double *neuronact = NULL;
double *cnormalization = NULL;

/*
 * standard logistic
 */
double logistic(double f)
{
	return ((double) (1.0 / (1.0 + exp(-f))));
}

/*
 * hyperbolic tangent
 */
double tanh(double f)
{
	if (f > 10.0)
		return 1.0;
	  else if (f < - 10.0)
		 return -1.0;
	    else
         return ((double) ((1.0 - exp(-2.0 * f)) / (1.0 + exp(-2.0 * f))));
	
}



// constructor
Evonet::Evonet()
{
	m_ninputs = 0;
	m_nhiddens = 0;
	m_noutputs = 0;
	m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
	m_nlayers = 1;
	m_bias = 0;
	m_netType = 0;
	m_actFunct = 2;
	m_outType = 0;
	m_wInit = 0;
	m_clip = 0;
	m_normalize = 0;
	m_randAct = 0;
    m_wrange = 1.0;
    m_nbins = 1;
    m_low = -1.0;
    m_high = 1.0;
	
	netRng = new RandomGenerator(time(NULL));
	//m_act = new double[MAXN];
	m_netinput = new double[MAXN];
	m_netblock = new int[MAX_BLOCKS * 5];
	m_nblocks = 0;
	m_neurontype = new int[MAXN];

}

/*
 * set the seed
 */
void Evonet::seed(int s)
{
    netRng->setSeed(s);
}
	
Evonet::Evonet(int ninputs, int nhiddens, int noutputs, int nlayers, int bias, int netType, int actFunct, int outType, int wInit, int clip, int normalize, int randAct, double wrange, int nbins, double low, double high)
{

	m_nbins = nbins;
	if (m_nbins < 1 || m_nbins > 20) // ensures that m_bins is in an appropriate rate
        m_nbins = 1;
    // set variables
	m_ninputs = ninputs;
	m_nhiddens = nhiddens;
	m_noutputs = noutputs;
	if (m_nbins > 1)
	{
	  m_noutputs = noutputs * m_nbins; // we several outputs for each actuator
	  m_noutputsbins = noutputs;        // we store the number of actuators
	}
	m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
	m_nlayers = nlayers;
	m_bias = bias;
	m_netType = netType;
	if (m_netType > 0)
		// Only feed-forward network can have more than one hidden layer
		m_nlayers = 1;
	m_actFunct = actFunct;
	m_outType = outType;
	m_wInit = wInit;
	m_clip = clip;
	m_normalize = normalize;
	m_randAct = randAct;
    m_low = low;
    m_high = high;
    m_wrange = wrange;

	netRng = new RandomGenerator(time(NULL));

    // display info and check parameters are in range
    printf("Network %d->", m_ninputs);
    int l;
    for(l=0; l < nlayers; l++)
        printf("%d->", m_nhiddens / m_nlayers);
    printf("%d ", m_noutputs);
    if (m_netType < 0 || m_netType > 2)
        m_netType = 0;
    if (m_netType == 0)
        printf("feedforward ");
    else if (m_netType == 1)
        printf("recurrent ");
    else if (m_netType == 2)
        printf("fully recurrent ");
    if (m_bias)
        printf("with bias ");
    if (m_actFunct < 1 || m_actFunct > 2)
        m_netType = 2;
    if (m_actFunct == 1)
        printf("logistic ");
      else
        printf("tanh ");
    if (m_outType < 1 || m_outType > 4)
        m_outType = 2; // ensure it has a proper value
    switch (m_outType)
    {
        case 1:
            printf("output:logistic ");
            break;
        case 2:
            printf("output:tanh ");
            break;
        case 3:
            printf("output:linear ");
            break;
        case 4:
            printf("output:binary ");
            break;
    }
	if (m_nbins > 1)
        printf("bins: %d", m_nbins);
    if (m_wInit < 0 || m_wInit > 2) // ensure it is in the proper range
        m_wInit = 0;
    if (m_wInit == 0)
        printf("init:xavier ");
    else if (m_wInit == 1)
        printf("init:norm-incoming ");
    else if (m_wInit == 2)
        printf("init:uniform ");
    if (m_normalize < 0 || m_normalize > 1)
        m_normalize = 0;
    if (m_normalize == 1)
        printf("input-normalization ");
    if (m_clip < 0 || m_clip > 1)
        m_clip = 0;
    if (m_clip == 1)
        printf("clip ");
    if (m_randAct < 0 || m_randAct > 1)
        m_randAct = 0;
    if (m_randAct == 1)
        printf("action-noise ");
    printf("\n");
    
	// allocate variables
    m_nblocks = 0;
	//m_act = new double[m_nneurons];
	m_netinput = new double[m_nneurons];
	m_netblock = new int[MAX_BLOCKS * 5];
	m_neurontype = new int[m_nneurons];
	// Initialize network architecture
	initNetArchitecture();
	
	// allocate vector for input normalization
	if (normalize == 1)
	  {
	  	m_mean = new double[m_ninputs];  // mean used for normalize
	    m_std = new double[m_ninputs];   // standard deviation used for normalize
	    m_sum = new double[m_ninputs];   // sum of input data used for calculating normalization vectors
	    m_sumsq = new double[m_ninputs]; // squared sum of input data used for calculating normalization vectors
	    m_count = 0.01;                  // to avoid division by 0
	    int i;
		for (i = 0; i < m_ninputs; i++)
	      {
		   m_sum[i] = 0.0;
		   m_sumsq[i] = 0.01;           // eps
		   m_mean[i] = 0.0;
		   m_std[i] = 1.0;
	      }
	  }
}
	
Evonet::~Evonet()
{
}


void Evonet::initNet(char* filename)
{
    
/*
	char *s;
    	char buff[MAX_STR_LEN];
   	char name[MAX_STR_LEN];
    	char value[MAX_STR_LEN];
    	char *ptr;
	char netTypeStr[MAX_STR_LEN];
	char wInitStr[MAX_STR_LEN];
	
	// read parameters from .ini file
	FILE* fp = fopen(filename, "r");
    	if (fp != NULL)
    	{
        	// Read lines
        	while (fgets(buff, MAX_STR_LEN, fp) != NULL)
        	{
		
            		//Skip blank lines and comments
            		if (buff[0] == '\n' || buff[0] == '#' || buff[0] == '['  || buff[0] == '/')
            			continue;
            
            		//Parse name/value pair from line
            		s = strtok(buff, " = ");
            		if (s == NULL)
            			continue;
            		else
            			copyandclear(s, name);
            
            		s = strtok(NULL, " = ");
            		if (s == NULL)
            			continue;
            		else
            			copyandclear(s, value);
            
        // Copy into correct entry in parameters struct
        if (strcmp(name, "ninputs")==0)
            m_ninputs = (int)strtol(value, &ptr, 10);
            		else if (strcmp(name, "nhiddens")==0)
            			m_nhiddens = (int)strtol(value, &ptr, 10);
            		else if (strcmp(name, "noutputs")==0)
            			m_noutputs = (int)strtol(value, &ptr, 10);
			        else if (strcmp(name, "nlayers")==0)
            			m_nlayers = (int)strtol(value, &ptr, 10);
            		else if (strcmp(name, "bias")==0)
            			m_bias = (int)strtol(value, &ptr, 10);
            		else if (strcmp(name, "netType")==0)
            			m_netType = (int)strtol(value, &ptr, 10);
            		else if (strcmp(name, "actFunct")==0)
            			m_actFunct = (int)strtol(value, &ptr, 10);
			else if (strcmp(name, "linOut")==0)
				m_linOut = (int)strtol(value, &ptr, 10);
			else if (strcmp(name, "wInit")==0)
				m_wInit = (int)strtol(value, &ptr, 10);
			else if (strcmp(name, "clip")==0)
				m_clip = (int)strtol(value, &ptr, 10);
			else if (strcmp(name, "normalize")==0)
				m_normalize = (int)strtol(value, &ptr, 10);
			else if (strcmp(name, "randAct")==0)
				m_randAct = (int)strtol(value, &ptr, 10);
			else
				printf("Unknown parameter %s, ignore it!\n", name);
      
        	}
        	fclose (fp);
		// Set the number of neurons
		m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
		if (m_nneurons > 0)
		{
			// Initialize variables
			if (m_act == NULL)
				m_act = new double[m_nneurons];
			if (m_netinput == NULL)
				m_netinput = new double[m_nneurons];
			if (m_neurontype == NULL)
				m_neurontype = new int[m_nneurons];
			// Initialize network architecture
			initNetArchitecture();
			if (m_sumInputs == NULL)
				m_sumInputs = new double[m_ninputs];
			if (m_sqSumInputs == NULL)
				m_sqSumInputs = new double[m_ninputs];
			// Initialize stat
			if (m_stat == NULL)
				m_stat = new Stat(m_ninputs, 0.01);
			if (m_netType == 0)
				sprintf(netTypeStr, "ff");
			else if (m_netType == 1)
				sprintf(netTypeStr, "rec");
			else if (m_netType == 2)
				sprintf(netTypeStr, "f-r");
			else
			{
				printf("Unknown net_type %d\n", m_netType);
				m_netType = 0;
				sprintf(netTypeStr, "ff");
			}
			if (m_wInit == 0)
				sprintf(wInitStr, "xavier");
			else if (m_wInit == 1)
				sprintf(wInitStr, "normc");
			else if (m_wInit == 2)
				sprintf(wInitStr, "uniform");
			else
			{
				printf("Unknown wInit %d\n", m_wInit);
				m_netType = 0;
				sprintf(wInitStr, "xavier");
			}
			printf("ninputs %d nhiddens %d noutputs %d nneurons %d nlayers %d actFunct %s netType %s bias %d weightInit %s linOut %d clip %d normalize %d randAct %d\n", m_ninputs, m_nhiddens, m_noutputs, m_nneurons, m_nlayers, ((m_actFunct == 2) ? "tanh" : "logistic"), netTypeStr, m_bias, wInitStr, m_linOut, m_clip, m_normalize, m_randAct);
		}
		else
			printf("ERROR: parsing file %s returns nneurons = %d\n", filename, m_nneurons);
    	}
    	else
    		printf("ERROR: unable to open file %s\n", filename);
 */
 
}

// reset net
void Evonet::resetNet()
{
	int i;
	for (i = 0; i < m_nneurons; i++)
	{
		neuronact[i] = 0.0;
		m_netinput[i] = 0.0;
	}
}

// store pointer to weights vector
void Evonet::copyGenotype(double* genotype)
{
	cgenotype = genotype;
}

// store pointer to observation vector
void Evonet::copyInput(float* input)
{
	cobservation = input;
}

// store pointer to update vector
void Evonet::copyOutput(float* output)
{
	caction = output;
}

// store pointer to neuron activation vector
void Evonet::copyNeuronact(double* na)
{
    neuronact = na;
}

// store pointer to neuron activation vector
void Evonet::copyNormalization(double* no)
{
    cnormalization = no;
}

// update net
void Evonet::updateNet()
{
	double* p;
	double* a;
	double* ni;
	int i;
    int t;
    int b;
	int* nbl;
	int* nt;
	int j;

	p = cgenotype;
	
	// Collect the input for updatig the normalization
	// We do that before eventually clipping (not clear whether this is the best choice)
	if (m_normalize  == 1 && m_normphase == 1)
		collectNormalizationData();
	
	// Copy inputs to neuronacts and eventually normalize inputs
	if (m_normalize == 1)
	{
		for (j = 0; j < m_ninputs; j++)
			neuronact[j] = (cobservation[j] - m_mean[j]) / m_std[j];
	}
	else
	{
		for (j = 0; j < m_ninputs; j++)
			neuronact[j] = cobservation[j];
	}
	
	// Clip input values
	if (m_clip == 1)
	{
		for (j = 0; j < m_ninputs; j++)
		{
			if (neuronact[j] < -CLIP_VALUE)
				neuronact[j] = -CLIP_VALUE;
			if (neuronact[j] > CLIP_VALUE)
				neuronact[j] = CLIP_VALUE;
			//printf("%.1f ", neuronact[j]);
		}
		//printf("\n");
	}


	// compute biases
	if (m_bias == 1)
	{
		// Only non-input neurons have bias
		for(i = 0, ni = m_netinput; i < m_nneurons; i++, ni++)
		{
			if (i >= m_ninputs)
			 {
               *ni = *p;
               p++;
			 }
			else
			 {
               *ni = 0.0;
			 }
		}
	}

	// blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
	{
        // connection block
        if (*nbl == 0)
		{
            for(t = 0, ni = (m_netinput + *(nbl + 1)); t < *(nbl + 2); t++, ni++)
			{
                for(i = 0, a = (neuronact + *(nbl + 3)); i < *(nbl + 4);i++, a++)
                    {
                        *ni += *a * *p;
                        p++;
                    }
           		}
        	}

        	// update block
        	if (*nbl == 1)
           	{
                for(t = *(nbl + 1), a = (neuronact + *(nbl + 1)), ni = (m_netinput + *(nbl + 1)), nt = (m_neurontype + *(nbl + 1)); t < (*(nbl + 1) + *(nbl + 2)); t++, a++, ni++, nt++)
                 	{
                     switch (*nt)
                      {
                        /*case 0:
                          // input neurons are simple rely units
                          *a = *(cobservation + t);
                        break;*/
                        case 1:
                          // Logistic
                          *a = logistic(*ni);
                          break;
                        case 2:
                          // Tanh
                          *a = tanh(*ni);
                        break;
                        case 3:
                          // linear
                          *a = *ni;
                        break;
                        case 4:
                          // Binary
                          if (*ni >= 0.5)
                            *a = 1.0;
                          else
                            *a = -1.0;
                        break;
                 	   }
			        }
        	}
        	nbl = (nbl + 5);
    	}
	// Store the action
    getOutput(caction);

}

// copy the output and eventually add noise
void Evonet::getOutput(float* output)
{
	
    // standard without bins
    if (m_nbins == 1)
    {
     int i;
	 for (i = 0; i < m_noutputs; i++)
	  {
		if (m_randAct == 1)
          output[i] = neuronact[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * 0.01);
        else
          output[i] = neuronact[m_ninputs + m_nhiddens + i];
	  }
    }
    else // with bins
    {
     int i = 0;
     int j = 0;
     double cact;
     int cidx;
	 // For each output, we look for the bin with the highest activation
     for (i = 0; i < m_noutputsbins; i++)
     {
        // Current highest activation
        cact = -9999.0;
        // Index of the current highest activation
        cidx = -1;
        for (j = 0; j < m_nbins; j++)
        {
            if (m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)] > cact)
            {
                cact = m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)];
                cidx = j;
            }
        }
        output[i] = 1.0 / ((double)m_nbins - 1.0) * (double)cidx * (m_high - m_low) + m_low;
		if (m_randAct == 1)
		     output[i] += (netRng->getGaussian(1.0, 0.0) * 0.01);
     }
     }

}


// compute the number of required parameters
int Evonet::computeParameters()
{
	int nparams;
	int i;
	int t;
	int b;
	int* nbl;

	nparams = 0;
	
	// biases
	if (m_bias)
		nparams += (m_nhiddens + m_noutputs);
    
    // blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    	{
		  // connection block
		  if (*nbl == 0)
		  {
		    	for(t = 0; t < *(nbl + 2); t++)
		    	{
		        	for(i = 0; i < *(nbl + 4); i++)
		        	{
		            		nparams++;
		        	}
		    	}
		  }
		nbl = (nbl + 5);
	}
	m_nparams = nparams;
	return nparams;
}

// initialize the architecture description
void Evonet::initNetArchitecture()
{
	int* nbl;
	int* nt;
	int n;
    
	m_nblocks = 0;
	nbl = m_netblock;

	// neurons' type
	for (n = 0, nt = m_neurontype; n < m_nneurons; n++, nt++)
      	{
          	if (n < m_ninputs)
                *nt = 0; // Inputs correspond to neuron type 0
          	else
              {
			    if (n < (m_ninputs + m_nhiddens))
                   *nt = m_actFunct; // activation function
			     else
			      *nt = m_outType;  // output activation function
			  }
      	}
	
	// input update block
	*nbl = 1; nbl++;
	*nbl = 0; nbl++;
	*nbl = m_ninputs; nbl++;
	*nbl = 0; nbl++;
	*nbl = 0; nbl++;
	m_nblocks++;
	
    // Fully-recurrent network
	if (m_netType == 2)
	{
	  	// hiddens and outputs receive connections from input, hiddens and outputs
	  	*nbl = 0; nbl++;
	  	*nbl = m_ninputs; nbl++;
	  	*nbl = m_nhiddens + m_noutputs; nbl++;
	  	*nbl = 0; nbl++;
	  	*nbl = m_ninputs + m_nhiddens + m_noutputs; nbl++;
	  	m_nblocks++;
		
	  	// hidden update block
	  	*nbl = 1; nbl++;
	  	*nbl = m_ninputs; nbl++;
	  	*nbl = m_nhiddens + m_noutputs; nbl++;
	  	*nbl = 0; nbl++;
	  	*nbl = 0; nbl++;
	  	m_nblocks++;
	}
    // recurrent network with 1 layer
	else if (m_netType == 1)
	{
        // input-hidden connections
        *nbl = 0; nbl++;
		*nbl = m_ninputs; nbl++;
		*nbl = m_nhiddens; nbl++;
		*nbl = 0; nbl++;
		*nbl = m_ninputs; nbl++;
		m_nblocks++;
    
        // hidden-hidden connections
		*nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
    
		// hidden update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;

		// hidden-output connections
		*nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
      
		// output-output connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        m_nblocks++;
    
        // output update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
	}
	else
	{
		// Feed-forward network
		if (m_nhiddens == 0)
		{
			// Sensory-motor network
			*nbl = 0; nbl++;
		  	*nbl = m_ninputs; nbl++;
		  	*nbl = m_noutputs; nbl++;
		  	*nbl = 0; nbl++;
		  	*nbl = m_ninputs; nbl++;
		  	m_nblocks++;

			// output update block
            *nbl = 1; nbl++;
            *nbl = m_ninputs; nbl++;
		  	*nbl = m_noutputs; nbl++;
            *nbl = 0; nbl++;
            *nbl = 0; nbl++;
            m_nblocks++;
		}
		else
		{
			// input-hidden connections
			if (m_nlayers == 1)
			{
                *nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				*nbl = m_nhiddens; nbl++;
				*nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				m_nblocks++;
				
				// hidden update block
                *nbl = 1; nbl++;
			  	*nbl = m_ninputs; nbl++;
			  	*nbl = m_nhiddens; nbl++;
			  	*nbl = 0; nbl++;
			  	*nbl = 0; nbl++;
			  	m_nblocks++;

				// hidden-output connections
				*nbl = 0; nbl++;
			  	*nbl = m_ninputs + m_nhiddens; nbl++;
			  	*nbl = m_noutputs; nbl++;
			  	*nbl = m_ninputs; nbl++;
			  	*nbl = m_nhiddens; nbl++;
			  	m_nblocks++;

				// output update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
			  	*nbl = m_noutputs; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
			}
			else
			{
				int nhiddenperlayer;
				int start;
				int end;
				int prev;
				int i;
				if ((m_nhiddens % m_nlayers) != 0)
				{
					printf("WARNING: invalid combination for number of hiddens %d and number of layers %d --> division has remainder %d... We set m_nlayers to 1\n", m_nhiddens, m_nlayers, (m_nhiddens % m_nlayers));
					m_nlayers = 1;
				}
				nhiddenperlayer = m_nhiddens / m_nlayers;
				*nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				*nbl = nhiddenperlayer; nbl++;
				*nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				m_nblocks++;
				// hidden update block
                *nbl = 1; nbl++;
			  	*nbl = m_ninputs; nbl++;
			  	*nbl = nhiddenperlayer; nbl++;
			  	*nbl = 0; nbl++;
			  	*nbl = 0; nbl++;
			  	m_nblocks++;
				start = m_ninputs + nhiddenperlayer;
				end = nhiddenperlayer;
				prev = m_ninputs;
				i = 1;
				while (i < m_nlayers)
				{
					*nbl = 0; nbl++;
					*nbl = start; nbl++;
					*nbl = end; nbl++;
					*nbl = prev; nbl++;
					*nbl = end; nbl++;
					m_nblocks++;
					// hidden update block
                    *nbl = 1; nbl++;
				  	*nbl = start; nbl++;
				  	*nbl = end; nbl++;
				  	*nbl = 0; nbl++;
				  	*nbl = 0; nbl++;
				  	m_nblocks++;
					i++;
					prev = start;
					start += nhiddenperlayer;
				}

				// hidden-output connections
				*nbl = 0; nbl++;
			  	*nbl = start; nbl++;
			  	*nbl = m_noutputs; nbl++;
			  	*nbl = prev; nbl++;
			  	*nbl = nhiddenperlayer; nbl++;
			  	m_nblocks++;

				// output update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
			  	*nbl = m_noutputs; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
			}
		}
	}
}

// initialize weights
void Evonet::initWeights()
{
    int i;
    int j;
    int t;
    int b;
    int* nbl;
    double range;
    
    
    // cparameter
    j = 0;
    // Initialize biases to 0.0
    if (m_bias)
    {
        // Bias are initialized to 0.0
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
        {
            cgenotype[j] = 0.0;
            j++;
        }
    }
    // Initialize weights of connection blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
        // connection block
        if (*nbl == 0)
        {
            switch (m_wInit)
            {
                // xavier initialization
                // gaussian distribution with range = (radq(2.0 / (ninputs + noutputs))
                case 0:
                    int nin;
                    int nout;
                    // ninput and noutput of the current block
                    nin = *(nbl + 4);
                    nout = *(nbl + 2);
                    // if previous and/or next block include the same receiving neurons we increase ninputs accordingly
                    // connection block are always preceded by update block, so we can check previous values
                    if ((*(nbl + 5) == 0) && ((*(nbl + 1) == *(nbl + 6)) && (*(nbl + 2) == *(nbl + 7))))
                        nin += *(nbl + 9);
                    if ((*(nbl - 5) == 0) && ((*(nbl - 4) == *(nbl + 1)) && (*(nbl - 3) == *(nbl + 2))))
                        nin += *(nbl - 1);
                    // compute xavier range
                    range = sqrt(2.0 / (nin + nout));
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            cgenotype[j] = netRng->getGaussian(range, 0.0);
                            j++;
                        }
                    }
                break;
                // normalization of incoming weights as in salimans et al. (2017)
                // in case of linear output, use a smaller range for the last layer
                // we assume that the last layer corresponds to the last connection block followed by the last update block
                // compute the squared sum of gaussian numbers in order to scale the weights
                // equivalent to the following python code for tensorflow:
                // out = np.random.randn(*shape).astype(np.double32)
                // out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                // where randn extract samples from Gaussian distribution with mean 0.0 and std 1.0
                // std is either 1.0 or 0.01 depending on the layer
                // np.square(out).sum(axis=0, keepdims=True) computes the squared sum of the elements in out
                case 1:
                   {
                    double *wSqSum = new double[*(nbl + 2)];
                    int k;
                    int cnt;
                       range = 1.0; // std
                    if (m_outType == 3 && b == (m_nblocks - 2))
                        range = 0.01; // std for layers followed by linear outputs
                    for (t = 0; t < *(nbl + 2); t++)
                        wSqSum[t] = 0.0;
                    // Index storing the genotype block to be normalized (i.e., the starting index)
                    k = j;
                    // Counter of weights
                    cnt = 0;
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            // Extract weights from Gaussian distribution with mean 0.0 and std 1.0
                            cgenotype[j] = netRng->getGaussian(1.0, 0.0);
                            // Update square sum of weights
                            wSqSum[t] += (cgenotype[j] * cgenotype[j]);
                            j++;
                            // Update counter of weights
                            cnt++;
                        }
                    }
                    // Normalize weights
                    j = k;
                    t = 0;
                    i = 0;
                    while (j < (k + cnt))
                    {
                        cgenotype[j] *= (range / sqrt(wSqSum[t])); // Normalization factor
                        j++;
                        i++;
                        if (i % *(nbl + 4) == 0)
                            // Move to next sum
                            t++;
                    }
		    // We delete the pointer
                    delete[] wSqSum;
                   }
                break;
                // normal gaussian distribution with range netWrange
                case 2:
                    // the range is specified manually and is the same for all layers
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            cgenotype[j] = netRng->getDouble(-m_wrange, m_wrange);
                            j++;
                        }
                    }
                break;
                default:
                    // unrecognized initialization mode
                    printf("ERROR: unrecognized initialization mode: %d \n", m_wInit);
                break;
            }
        }
        nbl = (nbl + 5);
    }
    /* print sum of absolute incoming weight
    j = 0;
    if (m_bias)
    {
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
            j++;
    }
    double sum;
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
     printf("block %d type %d\n", b, *nbl);
     if (*nbl == 0)
     {
      for(t = 0; t < *(nbl + 2); t++)
        {
          sum = 0;
          for(i = 0; i < *(nbl + 4); i++)
          {
            sum += fabs(cgenotype[j]);
            j++;
          }
          printf("block %d neuron %d sum-abs incoming weights %f\n", b, t, sum);
        }
     }
      nbl = (nbl + 5);
    }
    */
}


// set the normalization phase (0 = do nothing, 1 = collect data to update normalization vectors)
void Evonet::normphase(int phase)
{
   m_normphase = phase;
}

// collect data for normalization
void Evonet::collectNormalizationData()
{
	int i;
	for (i = 0; i < m_ninputs; i++)
       //printf("%.2f ", cobservation[i]);
	
	for (i = 0; i < m_ninputs; i++)
	{
		m_sum[i] += cobservation[i];
		m_sumsq[i] += (cobservation[i] * cobservation[i]);
		//printf("%.2f ", m_sum[i]);
	}
	//printf("\n");
	// Update counter
	m_count++;
}

// compute normalization vectors
void Evonet::updateNormalizationVectors()
{
	int i;
	int ii;
	double cStd;
	
	//printf("%.2f ]", m_count);
	for (i = 0; i < m_ninputs; i++)
	{
		m_mean[i] = m_sum[i] / m_count;
		//printf("%.2f ", m_mean[i]);
		cStd = (m_sumsq[i] / m_count - (m_mean[i] * m_mean[i]));
		if (cStd < 0.01)
			cStd = 0.01;
		m_std[i] = sqrt(cStd);
		//printf("%.2f  ", m_std[i]);
	}
	//printf("\n");
	// copy nornalization vectors on the cnormalization vector that is used for saving data
	ii = 0;
	for (i = 0; i < m_ninputs; i++)
	  {
	    cnormalization[ii] = m_mean[i];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	  {
	    cnormalization[ii] = m_std[i];
	    ii++;
	  }
}

// restore a loaded normalization vector
void Evonet::setNormalizationVectors()
{

  int i;
  int ii;
	
  if (m_normalize == 1)
  {
	ii = 0;
	for (i = 0; i < m_ninputs; i++)
	  {
	    m_mean[i] = cnormalization[ii];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	  {
	    m_std[i] = cnormalization[ii];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	   {
		 //printf("%.2f %.2f  ", m_mean[i], m_std[i]);
 	   }
  }
}

// reset normalization vector
void Evonet::resetNormalizationVectors()
{

  if (m_normalize == 1)
   {
    m_count = 0.01;                  // to avoid division by 0
    int i;
    for (i = 0; i < m_ninputs; i++)
    {
        m_sum[i] = 0.0;
        m_sumsq[i] = 0.01;           // eps
        m_mean[i] = 0.0;
        m_std[i] = 1.0;
    }
   }
}

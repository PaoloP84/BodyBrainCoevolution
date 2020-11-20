

/*
 * TO BE FIXED ***********************************************************
 * rob->motorwheelsid: non conosciamo qui il numero di hidden per settarlo correttamente
 **************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "discrim.h"
#include "utilities.h"
#include "robot-env.h"


// Pointer to the observations
float* cobservation;
// Pointer to the actions
float* caction;
// Pointer to termination flag
double* cdone;
// Pointer to world objects to be rendered
double* dobjects;

#define MAX_STR_LEN 1024


// read task parameters from the .ini file
bool readTaskConfig(const char* filename);

// custem variables required for this problem
double *robotsdist;              // matrix containing robot distances;
int *robotsbydistance;           // matrix containing the id of the robots ordered by the inverse of the distance
int robottype = MarXBot;         // the robot's type

// custom sensors defined in this file
void updateRobotDistances();
int initCameraSensorRFB(struct robot *cro, int nsectors);
void updateCameraAddBlob(double *cb, int *nb, double color, double dist, double brangel, double branger);
void updateCameraSensorRFB(struct robot *cro, int *rbd);
int initGroundSensor(struct robot *cro);
void updateGroundSensor(struct robot *cro);
float getled(int robot, int led);

/*
 * env constructor
 */
Problem::Problem()
{

    struct robot *ro;
    int r;

    // read task parameters
	nrobots = 1;  // number of robots (default value)
    readTaskConfig("ErForaging.ini");
	// create the environment
	initEnvironment();
    // creates the list of robot structures that contain the robot data
    // the list include a single element when we have a single robot
    // the robot structure and associated variables are define in the file robot-env.h
    rob = (struct robot *) malloc(nrobots * sizeof(struct robot));
    //
    // the initRobot function is defined in the robot-env.cpp file
	for (r=0, ro=rob; r < nrobots; r++, ro++)
    {
		
	   initRobot(ro, r, MarXBot);             // initilize robot variable (e.g. the size of the robots' radius, ext).
	   ro->maxSpeed = 500.0;                  // set robots max speed
		
	   ninputs = 0;
       ninputs += initInfraredSensor(ro);     // infrared sensor (8 units required to store the value of the corresponding 8 IF sensors)
	   ninputs += initCameraSensorRFB(ro, 2); // color camera sensor, two colors (red and bluee), two sectors
	   ninputs += initGroundSensor(ro);		  // ground sensors
	   ninputs += initEnergySensor(ro);       // energy sensor
	   ro->sensorinfraredid = 0; 			  // the id of the first infrared sensors (used for graphic purpose only to visually siaplay infrared activity)

	   initRobotSensors(rob, ninputs);        // allocate and initialize the robot->sensor vector that contain net->ninputs values and is passed to the function that update the state of the robots' network
		
       ro->motorwheels = 2;                   // define the motors used and set the number of motor neurons
       ro->motorleds = 2;                     // motorleds can be set to 1,2, or 3
	   ro->motorwheelstype = 1;               // the two motor neurons encode desired speed and rotation
       noutputs = ro->motorwheels + ro->motorleds;
		
       ro->motorwheelsid = ninputs + 10; // +net->nhiddens;
       ro->motorledsid = 0;
	}
	
	rng = new RandomGenerator(time(NULL));
	
	// allocate distance matrices and initialize the dyagonal distances to 0
    robotsdist = (double *) malloc((nrobots*nrobots) * sizeof(double));
    robotsbydistance = (int *) malloc((nrobots*nrobots) * sizeof(int));
}


Problem::~Problem()
{
}


/*
 * set the seed
 */
void Problem::seed(int s)
{
    	rng->setSeed(s);
}

/*
 * reset the initial condition randomly
 * when seed is different from 0, reset the seed
 */
void Problem::reset()
{

    int attempts;
    double dx, dy;
    struct robot *ro1;
    struct robot *ro2;
    int r1, r2;
    double cdist, mindist;
    double fcx, fcy;
    double distfromborder;
    double distfromareacenter;

	distfromborder = 1000.0;
	distfromareacenter = 500.0;
	// nest position
	fcx = envobjs[4].x = rng->getDouble(distfromborder, worldx - distfromborder);
	fcy = envobjs[4].y = rng->getDouble(distfromborder, worldy - distfromborder);
	// initial positions and orientations of the robots
	for (r1=0, ro1=rob; r1 < nrobots; r1++, ro1++)
	{
		mindist = 0.0; attempts = 0;
		while(mindist < (ro1->radius*2+5) && attempts < 100)
		{
			ro1->dir = rng->getDouble(0.0, PI2);
			ro1->x = fcx + rng->getDouble(-distfromareacenter, distfromareacenter);
			ro1->y = fcy + rng->getDouble(-distfromareacenter, distfromareacenter);
			mindist = 99999;
			for (r2=0, ro2=rob; r2 < r1; r2++, ro2++)
			  {
				dx = (ro1->x - ro2->x);
				dy = (ro1->y - ro2->y);
				cdist = sqrt(dx*dx+dy*dy);
				if (cdist < mindist)
					mindist = cdist;
			  }
			attempts++;
		}
	}

	// Get observations
	getObs();
	
}


void Problem::copyObs(float* observation)
{
	cobservation = observation;
}

void Problem::copyAct(float* action)
{
	caction = action;
}

void Problem::copyDone(double* done)
{
	cdone = done;
}

void Problem::copyDobj(double* objs)
{
	dobjects = objs;
}


/*
 * perform the action, update the state of the environment, update observations, return the reward
 */
double Problem::step()
{

    double dx, dy;
    double dist;
	
    *cdone = 0;
	if (updateRobot(rob, caction) != 0)
		*cdone = 1;
	
	getObs();
	
	//printf("xyd %.1f %.1f %.1f  o %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f  a %.1f %.1f done %.1f\n",
	//   rob->x, rob->y, rob->dir, cobservation[0],cobservation[1],cobservation[2],cobservation[3],
	//   cobservation[4],cobservation[5],cobservation[6],cobservation[7],caction[0], caction[1], *cdone);
    
    /*
     * return 1.0 when the robot is near the cylinder
     * and is not colliding
     */
    if (*cdone == 1)
      {
        return(0);
      }
      else
      {
        dx = envobjs[0].x - rob->x;
        dy = envobjs[0].y - rob->y;
        dist = sqrt((dx*dx)+(dy*dy));
        if (dist < (rob->radius + envobjs[0].r + 60))
           return(1.0);
         else
           return(0.0);
      }

}

double Problem::isDone()
{
	return *cdone;
}

void Problem::close()
{
    	//printf("close() not implemented\n");
}

/*
 * create the list of robots and environmental objects to be rendered graphically
 */
void Problem::render()
{
    
    int i;
    int c;
    struct robot *ro;
    
    c=0;
    // robots
    for (i=0, ro = rob; i < nrobots; i++, ro++)
    {
        dobjects[c] = 1.0;
        dobjects[c+1] = ro->x;
        dobjects[c+2] = ro->y;
        dobjects[c+3] = ro->radius;
        dobjects[c+4] = 0.0;
        dobjects[c+5] = ro->rgbcolor[0];
        dobjects[c+6] = ro->rgbcolor[1];
        dobjects[c+7] = ro->rgbcolor[2];
        dobjects[c+8] = ro->x + xvect(ro->dir, ro->radius);
        dobjects[c+9] = ro->y + yvect(ro->dir, ro->radius);
        c += 10;
    }
    for(i=0; i < nenvobjs; i++)
    {
        switch(envobjs[i].type)
        {
            case SAMPLEDSCYLINDER:
                dobjects[c] = 3.0;
                dobjects[c+3] = envobjs[i].r;
                dobjects[c+4] = 0.0;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
            case WALL:
                dobjects[c] = 2.0;
                dobjects[c+3] = envobjs[i].x2;
                dobjects[c+4] = envobjs[i].y2;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
        }
        dobjects[c+1] = envobjs[i].x;
        dobjects[c+2] = envobjs[i].y;
        dobjects[c+5] = envobjs[i].color[0];
        dobjects[c+6] = envobjs[i].color[1];
        dobjects[c+7] = envobjs[i].color[2];
        c += 10;
    }
    dobjects[c] = 0.0;
    
}

/*
 * update observation vector
 * ROB->SENSORS SHOULD POINT DIRECTLY TO COBSERVATION, WE DO NOT NEED THE FOR IN THAT CASE
 */
void Problem::getObs()
{

	// during each step reset the pointer of the input pattern that is later updated by the sensor function
	rob->csensors = rob->sensors;
	// update the input pattern by calling the sensor function (in this case we are using sigle sensor fuction)
	// the sensor function used here should be initialized in the initialize() function
	updateInfraredSensor(rob);

	// add noise to observations
	int i;
	for(i=0, rob->csensors = rob->sensors; i < ninputs; i++, rob->csensors++)
	  cobservation[i] = *rob->csensors += rng->getGaussian(0.03, 0.0);
	
}



/*
 * Initialize the environment
 * Environment are costituted by a rectangular area of size worldx*worldy and by a list of objects (walls, cylinders, coloured portions of the ground ext).
 * which are stored in a list of envojects structures (see the file robot-env.h for the definition of the structure)
 * Consequently thus function:
 * set the size of the arena, allocate the objects, and set the relevant characteristics of the objects
 */
void
Problem::initEnvironment()

{

	int cobj=0;
	
	// set the size of the arena
	worldx = 5000.0;		// world x dimension in mm
	worldy = 5000.0;		// world y dimension in mm
	
	// allocate the objects
	nenvobjs = 5;		// number of objects (4 wall objects and 1 nest)
	initEnvObjects(nenvobjs); // allocate and intilialize the environmental objects
	
    // top wall
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = 0.0;
    cobj++;
    // left wall
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = 0.0;
    envobjs[cobj].y2 = worldy;
    cobj++;
    // bottom wall
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = worldx;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = worldy;
    cobj++;
    // right wall
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = worldy;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = worldy;
    cobj++;
    // nest
    envobjs[cobj].type = STARGETAREA;
    envobjs[cobj].x = 400.0;
    envobjs[cobj].y = 400.0;
    envobjs[cobj].r = 400.0;
    envobjs[cobj].color[0] = 0.5;
    cobj++;
	
    if (cobj > nenvobjs)
    {
        printf("ERROR: you should allocate more space for environmental objects");
        fflush(stdout);
    }
	
}
/*
 * return the led-output of a given robot
 */
float getled(int robot, int led)
{
   return *(caction + (robot * 4) + led);
}

/*
 * this is an utility function of the RFB-Camera sensor
 * calculate the current distance among robots
 * and update the matrix of nearest robots
 */
void updateRobotDistances()

{
    
    struct robot *ro1;
    struct robot *ro2;
    int r1, r2, r3;
    double *cdist;
    double *ccdist;
    double smallerdist;
    int smallerid;
    int *rbd;
    double *rbddist;
    int remaining;
    
    
    // update the matrix of distances
    cdist = robotsdist;
    for (r1=0, ro1=rob; r1 < nrobots; r1++, ro1++)
    {
        for (r2=0, ro2=rob; r2 < r1; r2++, ro2++)
        {
            *cdist = ((ro1->x - ro2->x)*(ro1->x - ro2->x)) + ((ro1->y - ro2->y)*(ro1->y - ro2->y));
            *(robotsdist + (r1 + (r2 * nrobots))) = *cdist;
            cdist++;
        }
        *cdist = 9999999999.0;; // distance from itself is set to a large number to exclude itself from the nearest
        cdist = (cdist + (nrobots - r1));
    }
    
    // update the matrix of nearest robots
    rbd = robotsbydistance;
    for (r1=0, cdist = robotsdist; r1 < nrobots; r1++, cdist = (cdist + nrobots))
    {
        remaining = (nrobots - 1);
        for (r2=0; r2 < (nrobots - 1); r2++)
        {
            for (r3=0, ccdist = cdist, smallerid=0, smallerdist = *ccdist, rbddist = ccdist; r3 < nrobots; r3++, ccdist++)
            {
                if (*ccdist <= smallerdist)
                {
                    smallerdist = *ccdist;
                    smallerid = r3;
                    rbddist = ccdist;
                }
            }
            if (smallerdist < 1000000)  // we ignore robots located at a power dstance greater that 750*750
            {
                *rbd = smallerid;
                rbd++;
                *rbddist = 9999999999.0;
                remaining--;
            }
            else
            {
                *rbd = -1;
                rbd = (rbd + remaining);
                break;
            }
        }
        *rbd = -1;  // we use -1 to indicate the end of the list
        rbd++;
        
    }
    
}

/*
 * INITIALIZE THE CAMERA-RFB SENSORS (4 sensors)
 * This is an omnidirectional camera the compute the fraction of red and blue pixels
 * detected on the left and rigth side of the view field
 * The perceiced blobs correspond to the frontal and rear LEDs of the other robots
 * that can be turned on or off by them
 */

int initCameraSensorRFB(struct robot *cro, int nsectors)
{
    
    int i;
    double  **camb;
    int *nb;
    
    cro->camnsectors = nsectors;
    
    if (cro->idn == 0)
        printf("Sensor[%d]: camera2, %d sectors, %d colors \n", cro->camnsectors * 2, cro->camnsectors, 2);
    
    cro->camblobsn = (int *) malloc(cro->camnsectors * sizeof(double));
    // allocate space and initialize
    cro->camblobs = (double **) malloc(cro->camnsectors * sizeof(double *));
    for (i=0, camb = cro->camblobs, nb = cro->camblobsn; i < cro->camnsectors; i++, camb++, nb++)
    {
        *camb = (double *)malloc(nrobots*4*8 * sizeof(double));
        *nb = 0;
    }
    return(cro->camnsectors * 2);
}



/*
 * utility function used by the updateCameraSensorRFB function
 * add a new blob to the blob list of a sector
 * providing that it does not overlap with previously stored blobs
 * it receive as input the pointer to the first blob of the sector-list, the number of existing blobs, the blob color, and the start and ending angle
 * assume that the starting and ending angles are in the range [0, PI2]
 * blobs smaller than the resolution of the camera (0.1 degrees, 0.00174 radiants) are filtered out
 */
void updateCameraAddBlob(double *cb, int *nb, double color, double dist, double brangel, double branger)

{
    
    int b;
    
    // we ignore small blobs with a negative intervals since they are spurious
    if ((branger - brangel) < 0.00174)
    {
        return;
    }
    
    // check whether this blob overlap with preexisting ones
    for (b=0; b < *nb; b++)
    {
        cb++;
        cb++;
        // if fully overlap with previous blobs we simply filter it out
        if (anginrange(brangel, *cb, *(cb + 1)) && anginrange(branger, *cb, *(cb + 1)))
        {
            return;
        }
        else
        {
            // if start inside an existimg blob but ends after the end of the existing blobs we trim the first part of the blob
            if (anginrange(brangel, *cb, *(cb + 1)))
            {
                brangel = *(cb + 1);
            }
            else
            {
                // if end inside an existing blob but starts outside the existing blob we trim the last part of the blob
                if (anginrange(branger, *cb, *(cb + 1)))
                {
                    branger = *cb;
                }
            }
        }
        cb++;
        cb++;
    }
    
    // we ignore small blobs with a negative intervals since they are spurious
    // the blob could had become too small after being trimmed
    if ((branger - brangel) < 0.00174)
    {
        return;
    }
    
    *cb = color; cb++;
    *cb = dist; cb++;
    *cb = brangel; cb++;
    *cb = branger; cb++;
    *nb += 1;
    
}

/*
 * UPDATE THE CAMERA-RFB SENSORS (4 sensors)
 */
void updateCameraSensorRFB(struct robot *cro, int *rbd)

{
    
    
    int s;                        // sector
    struct robot *ro;            // pointer to robot list
    int r;                        // robot id
    double v1, v2;                // visible arc of the robot (from angle v1 to angle v2)
    double x1,x2,x3,y1,y2,y3;   // the coordinate of the initial and final points of the two blobs
    double a1,a2,a3;            // angle of the initial and final points of the two adjacent blobs
    double ab1, ab2;            // the angular boundaries between the frontal and rear side
    double ab;                    // the angular boundary located within the visible arc
    double d1, d2;                // distance of the two frontal/rear borders
    double ab1x, ab1y, ab2x,ab2y; // coordinates of the two borders
    int ac;                        // selected front/rear border
    double ang;                    // the angle of the perceiving robot from the perceived robot
    double dist2;               // the power distance between the two robots
    double rangel, ranger;        // initial and final angle of the current sector
    double cblob[3][2];            // color blobs ([Red, Blue, Green][ang-start, ang-end])
    double **camb;              // pointer to blob matrix
    double *cb;                    // pointer to blobs of a sectors
    int *nb;                    // pointer to the number of blobs x sectors
    double act[128];            // activation of the current visual sector (0=red, 1=blue)
    double secta;                // amplitude of visual sectors
    int c, color;                // color
    double bcolor;              // blob color
    double buf;
    int b;
    int ne;
    
    
    secta = M_PI / 3.0; //PI2 / (double) cro->camnsectors; // / 3.0;        // angular amplitude of the camera sectors
    for(s=0, nb = cro->camblobsn; s < cro->camnsectors; s++, nb++)
        *nb = 0;
    // we extract a red or blue color blob for each perceived robot
    // we stored the visible blobs divided by visual sectors and color
    // finally we compute the fraction of pixels for each sector and each color
    for (r=0; r < nrobots; r++)
    {
        if (rbd[r] < 0)     // if the list of nearby robots ended we exit from the for
            break;
        ro=(rob + rbd[r]);  // the r nearest perceived robot
        ne=rbd[r];          // the id of the r neareast perceived robot
        
        if (1 > 0 /* cro->idn != ro->idn*/)
        {
            // angle from perceived to perceiving robot
            ang = angv(cro->x, cro->y, ro->x, ro->y);
            // compute the visibile and coloured angular intervals
            v1 = ang - (M_PI / 2.0);
            v2 = ang + (M_PI / 2.0);
            ab1 = ro->dir - (M_PI / 2.0);
            ab2 = ro->dir + (M_PI / 2.0);
            // identify the relevant boundary (the boundary that is visible from the point of view of the perceiving robot)
            // we do that by checking the point that is nearer from the perceiving robot
            ab1x = ro->x + xvect(ab1, ro->radius);
            ab1y = ro->y + yvect(ab1, ro->radius);
            ab2x = ro->x + xvect(ab2, ro->radius);
            ab2y = ro->y + yvect(ab2, ro->radius);
            d1 =((ab1x - cro->x)*(ab1x - cro->x) + (ab1y - cro->y)*(ab1y - cro->y));
            d2 =((ab2x - cro->x)*(ab2x - cro->x) + (ab2y - cro->y)*(ab2y - cro->y));
            // the left and right border are followed and preceeded by different colors
            if (d1 <= d2)
            {
                ab = ab1;
                ac = 0;
            }
            else
            {
                ab = ab2;
                ac = 1;
            }
            // calculate the xy coordibate of the three points located on the borders of the perceived robot
            x1 = ro->x + xvect(v2, ro->radius);
            y1 = ro->y + yvect(v2, ro->radius);
            x2 = ro->x + xvect(ab, ro->radius);
            y2 = ro->y + yvect(ab, ro->radius);
            x3 = ro->x + xvect(v1, ro->radius);
            y3 = ro->y + yvect(v1, ro->radius);
            // calculate the correspoding three angle from the point of view of the perceiving robot
            a1 = angv(x1, y1, cro->x, cro->y);
            a2 = angv(x2, y2, cro->x, cro->y);
            a3 = angv(x3, y3, cro->x, cro->y);
            // extract the angular intervals of the red and blue subsections
            if (ac == 0)
            {
                cblob[0][0] = a1;
                cblob[0][1] = a1 + angdelta(a1, a2);
                cblob[1][0] = a3 - angdelta(a2, a3);
                cblob[1][1] = a3;
            }
            else
            {
                cblob[1][0] = a1;
                cblob[1][1] = a1 + angdelta(a1, a2);
                cblob[0][0] = a3 - angdelta(a2, a3);
                cblob[0][1] = a3;
            }
            
            // angles sanity checks
            for (c=0; c < 2; c++)
            {
                // if the first angle is negative the blog is over the border
                // we make both angle positive
                // it will the be divided in two blobs below because the ending angle will exceed PI2
                if (cblob[c][0] < 0)
                {
                    cblob[c][0] += PI2;
                    cblob[c][1] += PI2;
                }
                // if the second angle is smaller than the first and the interval is small, we invert them
                // apparently this is due to limited precision of angle calculation
                if ((cblob[c][1] - cblob[c][0]) < 0)
                {
                    buf = cblob[c][0];
                    cblob[c][0] = cblob[c][1];
                    cblob[c][1] = buf;
                }
            }
            
            /*
             for (c=0; c < 2; c++)
             {
             if ((cblob[c][1] - cblob[c][0]) < 0)
             {
             printf("negative %.4f %.4f   %.4f %d ", cblob[c][0], cblob[c][1], cblob[c][1] - cblob[c][0], ac);
             if (ac == 0 && c == 0) printf("red  (%.4f %.4f %.4f) a1 a2 %.4f %.4f a1 + a1_a2 %.4f \n", a1, a2, a3, a1, a2, angdelta(a1, a2));
             if (ac == 0 && c == 1) printf("blue (%.4f %.4f %.4f) a3 a2 %.4f %.4f a3 - a2_a3 %.4f\n", a1, a2, a3, a3, a2, angdelta(a2, a3));
             if (ac == 1 && c == 1) printf("blue (%.4f %.4f %.4f) a1 a2 %.4f %.4f a1 + a1_a2 %.4f \n", a1, a2, a3, a1, a2, angdelta(a1, a2));
             if (ac == 1 && c == 0) printf("red  (%.4f %.4f %.4f) a3 a2 %.4f %.4f a3 - a2_a3 %.4f \n", a1, a2, a3, a3, a2, angdelta(a2, a3));
             }
             
             if ((cblob[c][1] - cblob[c][0]) > 0.8)
             printf("large %.4f %.4f   %.4f %d\n", cblob[c][0], cblob[c][1], cblob[c][1] - cblob[c][0], ac);
             }
             */
            
            // we store the two blobs
            // blobs extending over PI2 are divided in two
            dist2 =((ro->x - cro->x)*(ro->x - cro->x) + (ro->y - cro->y)*(ro->y - cro->y));
            camb = cro->camblobs;
            nb = cro->camblobsn;
            cb = *camb;
            // we check whether frontal red leds are turned on or not
            if (ro->motorleds == 0 || getled(ne, 0) > 0.5)
                bcolor = 1.0;
            else
                bcolor = 0.0;
            if (cblob[0][1] < PI2)
            {
                updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[0][0], cblob[0][1]);
            }
            else
            {
                updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[0][0], PI2);
                updateCameraAddBlob(cb, nb, bcolor, dist2, 0.0, cblob[0][1] - PI2);
            }
            // we check whether rear blue leds are turned on or not
            if (ro->motorleds == 0 || getled(ne, 1) > 0.5)
                bcolor = 2.0;
            else
                bcolor = 0.0;
            if (cblob[1][1] < PI2)
            {
                updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[1][0], cblob[1][1]);
            }
            else
            {
                updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[1][0], PI2);
                updateCameraAddBlob(cb, nb, bcolor, dist2, 0.0, cblob[1][1] - PI2);
            }
            
        }  // end if (cro->idn != ro->idn)
    }  // end for nrobots
    
    
    
    // sum the angular contribution of each relevant blob to each color sector
    double inrange;
    double addsrangel;  // additional sector rangel
    double addsranger;  // additional sector ranger
    int addsid;         // sector to which the additial sector belong
    int ss;             // id of the sector, usually = s, but differ for the additional sector
    double *cbadd;      // pointer to blob list used to add a new sub-blob
    
    // initialize to 0 neurons actiovation
    for(b=0; b < cro->camnsectors * 2; b++)
        act[b] = 0.0;
    
    camb = cro->camblobs;
    cb = *camb;
    nb = cro->camblobsn;
    b = 0;
    while (b < *nb)
    {
        inrange=false;
        addsid = -1;  // the id of the additional sensors is initialized to a negative number
        if (*cb == 0.0) color = -1; // black
        if (*cb == 1.0) color = 0; // red
        if (*cb == 2.0) color = 1; // blue
        //if (cro->idn == 0) printf("b %d) %.2f %.2f %.2f %.2f (%.2f) \n", b, *cb, *(cb + 1), *(cb + 2), *(cb + 3), *(cb + 3) - *(cb + 2));
        for(s=0, rangel = cro->dir - secta, ranger = rangel + secta; s < (cro->camnsectors + 1); s++, rangel += secta, ranger += secta)
        {
            
            if (s < cro->camnsectors)
            {
                ss = s;
                //if (cro->idn == 0) printf("sector %d (ss %d) %.2f %.2f  \n", s, ss, rangel, ranger);
                // we normalize the angles of the sector in the range [0, PI2+sectora]
                if (rangel < 0.0)
                {
                    rangel += PI2;
                    ranger += PI2;
                }
                // if the current sector extend over PI2 we trim it to PI2 and we initialize the additional sector
                if (ranger > PI2)
                {
                    addsrangel = 0.0;
                    addsranger = ranger - PI2;
                    addsid=s;
                }
            }
            else
            {
                // if an additional sensor has been defined we process is otherwise we exit from the sector for
                if (addsid >= 0)
                {
                    ss = addsid;
                    // if (cro->idn == 1) printf(" Additional sector s %d ss %d addsid %d range %.2f %.2f\n", s, ss, addsid, addsrangel, addsranger);
                    rangel = addsrangel;
                    ranger = addsranger;
                }
                else
                {
                    break;
                }
            }
            //if (cro->idn == 0) printf("sector %d (ss %d) %.2f %.2f  \n", s, ss, rangel, ranger);
            if (color >= 0)
            {
                if ((*(cb + 2) >= rangel) && (*(cb + 2) < ranger) && (*(cb + 3) >= rangel) && (*(cb + 3) < ranger) ) // coloured blob fully contained in the sector
                {
                    act[ss * cro->camnsectors + color] += *(cb + 3) - *(cb + 2);
                    //if (cro->idn == 1) printf("fullin rodir %.2f sector %d %.2f %.2f blobcolor %.2f blobang %.2f %.2f (%.2f)\n", cro->dir, s, rangel, ranger, *cb, *(cb + 2), *(cb + 3), *(cb + 3) - *(cb + 2));
                    inrange=true;
                }
                else
                {
                    if ((*(cb + 2) >= rangel) && (*(cb + 2) < ranger) && (*(cb + 3) >= rangel))  // non-black blob staring inside and ending outside, inside the next sector
                    {
                        act[ss * cro->camnsectors + color] += ranger - *(cb + 2);
                        // we use the exceeding part to create a new blob added at the end of the blob list
                        camb = cro->camblobs;
                        cbadd = *camb;
                        cbadd = (cbadd + (*nb * 4));
                        *cbadd = *cb; cbadd++;
                        *cbadd = *(cb + 1); cbadd++;
                        *cbadd = ranger; cbadd++; // the new blob start from the end of the current sector
                        *cbadd = *(cb + 3); cbadd++;
                        *nb = *nb + 1;
                        //printf("added blob %d %.2f %.2f %.6f %.6f  range %.6f %.6f \n", *nb, *cb, *(cb + 1), ranger, *(cb + 3), rangel, ranger);
                        //if (cro->idn == 1) printf("startin rodir %.2f sector %d %.2f %.2f blobcolor %.2f blobang %.2f %.2f (%.2f)\n", cro->dir, s, rangel, ranger, *cb, *(cb + 2), ranger, ranger - *(cb + 2));
                        inrange=true;
                    }
                    
                }
            }
        }
        if (!inrange)   // blobs outsiode the view range of all sectors are turned in black for graphical purpose
            *cb = 0.0;
        
        cb = (cb + 4);
        b++;
    }
    //if (cro->idn == 0) printf("\n\n");
    
    
    // we finally store the value in the input units
    //printf("activation s1 red blue s2 red blue robot %d ", cro->idn);
    for(b=0; b < cro->camnsectors * 2; b++, cro->csensors++)
    {
        *cro->csensors = act[b];
        //printf("%.2f ", act[b]);
    }
    //printf("\n");
    
    
}



/*
 * INIT GROUND SENSOR
 * the first unit is activated when the robot is on a target area with a color < 0.25
 * the second unit is activated when the robot is on a target are with a color > 0.25 and < 0.75
 */
int initGroundSensor(struct robot *cro)
{
    if (cro->idn == 0) printf("Sensor[%d]: ground color \n", 2);
    return(2);
}

/*
 * UPDATE GROUND SENSOR
 * the first unit is activated when the robot is on a target area with a color < 0.25
 * the second unit is activated when the robot is on a target are with a color > 0.25 and < 0.75
 */
void updateGroundSensor(struct robot *cro)
{
    
    int o;
    double dx, dy, cdist;
    double act[2];
    
    act[0] = act[1] = 0.0;
    
    for (o=0; o < nenvobjs; o++)
    {
        if (envobjs[o].type == STARGETAREA)
        {
            dx = cro->x - envobjs[o].x;
            dy = cro->y - envobjs[o].y;
            cdist = sqrt((dx*dx)+(dy*dy));
            if (cdist < envobjs[o].r)
            {
                if (envobjs[o].color[0] < 0.25)
                    act[0] = 1.0;
                else
                    if (envobjs[o].color[0] < 0.75)
                        act[1] = 1.0;
            }
        }
    }
    *cro->csensors = act[0];
    cro->csensors++;
    *cro->csensors = act[1];
    cro->csensors++;
}

/*
 * read task parameters from the configuration file
 */
bool readTaskConfig(const char* filename)
{
    char *s;
    char buff[1024];
    char name[1024];
    char value[1024];
    char *ptr;
    int section;  // 0=before the section 1=in the section 2= after the section

    section = 0;

    FILE* fp = fopen(filename, "r");
    if (fp != NULL)
    {
        // Read lines
        while (fgets(buff, 1024, fp) != NULL)
        {

            // check whether the relevant section start or end
            if (buff[0] == '[')
            {
              if (section == 1)
                {
                    section = 2;
                    continue;
                }
              if ((section == 0) && (strncmp(buff, "[TASK]",5)==0))
              {
                section = 1;
                continue;
              }
            }

            if (section == 1)
            {

            //Skip blank lines and comments
            if (buff[0] == '\n' || buff[0] == '#' || buff[0] == '/')
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
            if (strcmp(name, "nrobots")==0)
            nrobots = (int)strtol(value, &ptr, 10);
            else printf("WARNING: Unknown parameter %s in section [TASK] of file %s \n", name, filename);
         }
        }
        fclose (fp);
        if (section == 0)
           printf("WARNING: Missing section [TASK] in file %s \n", filename);
		
        return(true);
    }
    else
    {
        printf("ERROR: unable to open file %s\n", filename);
        fflush(stdout);
        return(false);
    }
}

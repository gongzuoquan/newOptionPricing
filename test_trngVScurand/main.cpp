#include<iostream>
#include"testRNG.h"
#include"helper_timer.h"
using namespace std;
#define Real float
unsigned int numSim=1048576;
unsigned int numTimeStep=200;
unsigned int threadBlockSize=128;
unsigned int seed=123344;
int main(int argc,char **argv)
{	
	if(argc>=2)
	{
		numSim=atoi(argv[1]);
	}

	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	testRNG<Real> rng(numSim,numTimeStep,threadBlockSize,seed);

	sdkStartTimer(&timer);
	rng(); //
	sdkStopTimer(&timer);

	float elapsedTime=sdkGetAverageTimerValue(&timer)/1000.0f;

	//六、结果输出
	cout<<"        run time : "<<elapsedTime<<endl;
	sdkDeleteTimer(&timer);
	timer=NULL;

	return 0;
}




#include"testRNG.h"
#include<iostream>
#include<curand_kernel.h>
#include"helper_timer.h"
#include<trng/yarn5s.hpp>
#include<trng/yarn2.hpp>
#include<trng/normal_dist.hpp>

__global__ void initRNG(curandState *const rngStates,const unsigned int seed)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}
__global__ void initRNG(curandStatePhilox4_32_10_t *const rngStates,
					    const unsigned int seed)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}

__global__ void initRNG(curandStateMRG32k3a *const rngStates,
					    const unsigned int seed)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}

__global__ void generator(curandState *const  rngStates,
						  const unsigned int numSims,
						  const unsigned int numTimeSteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目

	curandState localState=rngStates[tid];
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		for(unsigned int t=0;t<numTimeSteps;t++)
		{
			curand_normal(&localState);
		}
	}
	return ;
}
__global__ void generator(curandStatePhilox4_32_10_t *const  rngStates,
						  const unsigned int numSims,
						  const unsigned int numTimeSteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目

	curandStatePhilox4_32_10_t localState=rngStates[tid];
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		for(unsigned int t=0;t<numTimeSteps;t++)
		{
			curand_normal(&localState);
		}
	}
	return ;
}
__global__ void generator(curandStateMRG32k3a *const  rngStates,
						  const unsigned int numSims,
						  const unsigned int numTimeSteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目

	curandStateMRG32k3a localState=rngStates[tid];
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		for(unsigned int t=0;t<numTimeSteps;t++)
		{
			curand_normal(&localState);
		}
	}
	return ;
}

__global__ void generator(trng::yarn5s rngState,
						  const unsigned int numSims,
						  const unsigned int numTimeSteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目
	rngState.jump(tid);
	trng::normal_dist<double> trng_normal(0.0,1.0);

	for(unsigned int i=tid;i<numSims;i+=step)
	{
		for(unsigned int t=0;t<numTimeSteps;t++)
		{
			trng_normal(rngState);
		}
	}
	return ;
}

__global__ void generatorLeap(trng::yarn2 *rngStates,
						  const unsigned int numSims,
						  const unsigned int numTimeSteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目
	trng::normal_dist<double> trng_normal(0.0,1.0);

	for(unsigned int i=tid;i<numSims;i+=step)
	{
		for(unsigned int t=0;t<numTimeSteps;t++)
		{
			trng_normal(rngStates[tid]);
		}
	}
	return ;
}

template<typename Real>
testRNG<Real>::testRNG(unsigned int numSims, 
					   unsigned int numTimeSteps,
					   unsigned int threadBlockSize,
					   unsigned int seed
					  )
					 :m_numSims(numSims),
					  m_numTimeSteps(numTimeSteps),
					  m_threadBlockSize(threadBlockSize),
					  m_seed(seed)
{
	return ;
}

template <typename Real>
void testRNG<Real>::operator()()
{
	cudaError_t cudaResult=cudaSuccess; //cuda函数运行结果标识
	curandState *d_rngStates;
	//curandState *d_rngStates=(curandState*)malloc(sizeof(curandState)*m_numSims);
	curandStatePhilox4_32_10_t *d_rngStatesP=(curandStatePhilox4_32_10_t*)malloc(sizeof(curandStatePhilox4_32_10_t)*m_numSims);
	curandStateMRG32k3a *d_rngStatesM=(curandStateMRG32k3a*)malloc(sizeof(curandStateMRG32k3a)*m_numSims);

	dim3 block,grid;
	block.x=m_threadBlockSize;
	block.y=1;
	block.z=1;

	grid.x=m_numSims/m_threadBlockSize;
	grid.y=1;
	grid.z=1;

	cudaResult=cudaMalloc((void **)&d_rngStates,m_numSims*sizeof(curandState));
	cudaResult=cudaMalloc((void **)&d_rngStatesP,m_numSims*sizeof(curandStatePhilox4_32_10_t));
	cudaResult=cudaMalloc((void **)&d_rngStatesM,m_numSims*sizeof(curandStateMRG32k3a));
	//std::cout<<"ok"<<std::endl;
	cudaDeviceSynchronize();

	initRNG<<<grid,block>>>(d_rngStates,m_seed);
	generator<<<grid,block>>>(d_rngStates,m_numSims,m_numTimeSteps);


	initRNG<<<grid,block>>>(d_rngStatesP,m_seed);
	generator<<<grid,block>>>(d_rngStatesP,m_numSims,m_numTimeSteps);


	initRNG<<<grid,block>>>(d_rngStatesM,m_seed);
	generator<<<grid,block>>>(d_rngStatesM,m_numSims,m_numTimeSteps);


	trng::yarn5s d_rngStateT;
	generator<<<grid,block>>>(d_rngStateT,m_numSims,m_numTimeSteps);


	cudaDeviceSynchronize();

	//StopWatchInterface *timer=NULL;
	//sdkCreateTimer(&timer);

	//sdkStartTimer(&timer);
	trng::yarn2 *rx=new trng::yarn2[m_numSims];
	for(int i=0;i<m_numSims;i++)
	{
		rx[i].split(m_numSims,i);
	}
	//sdkStopTimer(&timer);

	trng::yarn2 *rx_device;
	cudaMalloc(&rx_device,m_numSims*sizeof(*rx_device));
	cudaMemcpy(rx_device, rx, m_numSims*sizeof(*rx),cudaMemcpyHostToDevice);
	generatorLeap<<<grid,block>>>(rx_device,m_numSims,m_numTimeSteps);

	//float elapsedTime=sdkGetAverageTimerValue(&timer)/1000.0f;
	//std::cout<<"        run time : "<<elapsedTime<<std::endl;
	//sdkDeleteTimer(&timer);
	//timer=NULL;


	//std::cout<<"ok"<<std::endl;
	cudaDeviceSynchronize();

	cudaFree(d_rngStates);
	cudaFree(d_rngStatesP);
	cudaFree(d_rngStatesM);


	//std::cout<<"ok"<<std::endl;
	cudaDeviceSynchronize();

}
template class testRNG<float>;
template class testRNG<double>;

//2022.7.14
//测试gpu上的variance计算方法
//使用单浮点float类型数据
#include<iostream>
#include<curand.h>
#include<cuda_runtime.h>
#include"cudasharedmem.h"
#include"stdexcept"
#include<string>
#include<cooperative_groups.h> //cg
#include<cstdio>
using namespace std;
namespace cg=cooperative_groups;
//#define N 128
//#define N 1048576
#define N 33554432
#define seed 123456
__device__ float reduce_sum(float in, 
						   cg::thread_block cta
						   )
{
	SharedMemory<float> sdata;
	unsigned int ltid=threadIdx.x;
	sdata[ltid]=in; //加载数据
	cg::sync(cta); //<<同步

	//交错归约
	for(unsigned int s=blockDim.x/2;s>0;s>>=1)
	{
		if(ltid<s)
		{
			sdata[ltid]+=sdata[ltid+s]; //把数据归约到当前数组元素上
		}
		cg::sync(cta); //<<同步
	}
	return sdata[0]; //索引为0的元素存储了归约结果
}

__global__ void computeVariance(float *const sums,
							 float *const sumSquares,
							 float *const rands)
{
	//声明在同一个cg中
	cg::thread_block cta=cg::this_thread_block();

	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	//unsigned int step=gridDim.x*blockDim.x; //thread数目

	float rand=rands[tid];
	float randSquare=rands[tid]*rands[tid];
	//printf("%d: %f %f\n",threadIdx.x,rand,randSquare);

	//块内的所有payoff求个均值
	rand=reduce_sum(rand,cta);
	randSquare=reduce_sum(randSquare,cta);

	//将所有块内均值结果存在values中，等待再次归约（在主机端）
	if(threadIdx.x==0)
	{
		sums[bid]=rand;
		sumSquares[bid]=randSquare;
		//printf("tid=0: %f %f\n",rand,randSquare);
	}
}
int main()
{
	//1.先使用标准正态分布随机数，存于global memory
	//2.调用规约方法进行variance计算
	//3.验证返回值的正确性

	//1.调用curand 生成随机数
	//先给随机数分配存储空间
	float *rands=(float*)malloc(sizeof(float)*N);
	float *d_rands;
	cudaError_t cudaResult=cudaSuccess;
	cudaResult=cudaMalloc((void**)&d_rands,sizeof(float)*N);
	if (cudaResult!=cudaSuccess)
	{
		string msg("Could not cudaMalloc: ");
		msg += cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}

	// Generate random points in unit square
	curandStatus_t curandResult;
	curandGenerator_t prng;
	curandResult = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
        string msg("Could not create pseudo-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
	}
         
	curandResult = curandSetPseudoRandomGeneratorSeed(prng, seed);
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
        string msg("Could not set seed for pseudo-random number generator: ");
        msg += curandResult;
	    throw std::runtime_error(msg);
	}

	curandResult = curandGenerateNormal(prng, (float *)d_rands, N,0.0,1.0);
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
        string msg("Could not generate pseudo-random number: ");
        msg += curandResult;
	    throw std::runtime_error(msg);
	}
	curandResult = curandDestroyGenerator(prng);
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
        string msg("Could not destroy pseudo-random number generator: ");
        msg += curandResult;
	    throw std::runtime_error(msg);
	}

	/*
	cudaResult=cudaMemcpy(rands, d_rands, N*sizeof(float), cudaMemcpyDeviceToHost); 
	if (cudaResult!=cudaSuccess)
	{
		string msg("Could not memcpy: ");
		msg += cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}
	*/

	dim3 grid, block;
	block.x=128;
	grid.x=(N+block.x-1)/block.x;
	float *sums=(float*)malloc(sizeof(float)*grid.x);
	float *sumSquares=(float*)malloc(sizeof(float)*grid.x);

	float *d_sums,*d_sumSquares;
	cudaResult=cudaMalloc((void**)&d_sums,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumSquares,sizeof(float)*grid.x);

	//2.调用规约方法进行variance计算
	computeVariance<<<grid,block,block.x*sizeof(float),0>>>(d_sums,d_sumSquares,d_rands);
	cudaDeviceSynchronize();

	cudaResult=cudaMemcpy(sums, d_sums, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumSquares, d_sumSquares, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 

	cudaDeviceSynchronize();
	//cout<<"grid.x: "<<grid.x<<endl;

	float S1=0.0,S2;
	float sum=0.0;
	for(int i=0;i<grid.x;i++)
	{
		S1+=sumSquares[i];
		sum+=sums[i];
	}
	//cout<<"sum: "<<sum<<endl;
	//cout<<"S1: "<<S1<<endl;
	S1/=N;
	float avg=sum/N;
	S2=avg*avg;
	//cout<<"S1: "<<S1<<"  S2: "<<S2<<endl;

	//3.验证返回值的正确性
	float variance=S1-S2;
	cout<<"gold: 1.0"<<endl;
	cout<<"var : "<<variance<<endl;
	/*
	for(int i=0;i<100;i++)
	{
		cout<<rands[i]<<" ";
	}
	*/
	cout<<endl;

	cudaFree(d_rands);
	cudaFree(d_sums);
	cudaFree(d_sumSquares);
	free(rands);
	free(sums);
	free(sumSquares);

	return 0;
}


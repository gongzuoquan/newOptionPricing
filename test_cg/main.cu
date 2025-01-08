//2022.7.14
//测试gpu上的variance计算方法
//使用单浮点float类型数据
#include<iostream>
#include<cuda.h>
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
#define range 100
#define b1 100.890
#define b2 10.2340
#define b3 120.5670

__global__ void generateSamples(float *const X,
								float *const Y)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	X[tid]=X[tid]*range;
	X[tid+N]=X[tid+N]*range;
	Y[tid]=b1+b2*X[tid]+b3*X[tid+N]+Y[tid];
}
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
__device__ int reduce_sum(int in, 
						   cg::thread_block cta
						   )
{
	SharedMemory<int> sdata;
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
__global__ void computeWeights(float *const sumX2s,float *const sumX2Squares,
							   float *const sumX3s,float *const sumX3Squares,
							   float *const sumYs,
							   float *const sumX2X3s,
							   float *const sumX2Ys,float *const sumX3Ys,
							   float *const X,float *const Y,
							   int *const count)
{
	//声明在同一个cg中
	cg::thread_block cta=cg::this_thread_block();

	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	//unsigned int step=gridDim.x*blockDim.x; //thread数目

	float X2=reduce_sum(X[tid],cta);
	float X2Square=reduce_sum(X[tid]*X[tid],cta);
	float X3=reduce_sum(X[tid+N],cta);
	float X3Square=reduce_sum(X[tid+N]*X[tid+N],cta);
	float Y_=reduce_sum(Y[tid],cta);
	float X2X3=reduce_sum(X[tid]*X[tid+N],cta);
	float X2Y=reduce_sum(X[tid]*Y[tid],cta);
	float X3Y=reduce_sum(X[tid+N]*Y[tid],cta);
	if (X[tid]>50)
		count[tid]=1;
	else
		count[tid]=0;
	int lcount=reduce_sum(count[tid],cta);

	//将所有块内均值结果存在values中，等待再次归约（在主机端）
	if(threadIdx.x==0)
	{
		sumX2s[bid]=X2;
		sumX2Squares[bid]=X2Square;
		sumX3s[bid]=X3;
		sumX3Squares[bid]=X3Square;
		sumYs[bid]=Y_;
		sumX2X3s[bid]=X2X3;
		sumX2Ys[bid]=X2Y;
		sumX3Ys[bid]=X3Y;
		count[bid]=lcount;
	}
	if(tid==0)
	{
		float sumX2=0,sumX2Square=0,sumX3=0,sumX3Square=0,sumY=0,sumX2X3=0,sumX2Y=0,sumX3Y=0;
		int totalCount=0;

		for(int i=0;i<gridDim.x;i++)
		{
			sumX2+=sumX2s[i];
			sumX2Square+=sumX2Squares[i];
			sumX3+=sumX3s[i];
			sumX3Square+=sumX3Squares[i];

			sumY+=sumYs[i];
			sumX2X3+=sumX2X3s[i];
			sumX2Y+=sumX2Ys[i];
			sumX3Y+=sumX3Ys[i];
			totalCount+=count[i];
		}

		float sumx2x3=sumX2X3-1/static_cast<float>(N)*sumX3*sumX2;
		float sumx2Square=sumX2Square-1/static_cast<float>(N)*sumX2*sumX2;
		float sumx3Square=sumX3Square-1/static_cast<float>(N)*sumX3*sumX3;
		float sumx2y=sumX2Y-1/static_cast<float>(N)*sumY*sumX2;
		float sumx3y=sumX3Y-1/static_cast<float>(N)*sumY*sumX3;

		float beta2=(sumx2y*sumx3Square-sumx3y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
		float beta3=(sumx3y*sumx2Square-sumx2y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
		float beta1=(sumY-beta2*sumX2-beta3*sumX3)/static_cast<float>(N);
		printf("%f %f %f %d\n",beta1,beta2,beta3,totalCount);
		printf("%d %d \n",cta.size(),cta.thread_rank());
	}
}
int main()
{
	//1.确定标准模型参数：b1=1,b2=2,b3=3
	//2.随机生成样本集合：
	//  使用均匀分布随机生成X样本(x1,x2)，x范围定为0-100；
	//  利用标准模型求出Y；
	//  在Y的基础上加入标准正态分布噪音
	//3.调用规约方法计算求解所需变量
	//4.验证返回值的正确性

	//1.确定标准模型参数：b1=1,b2=2,b3=3

	//2.随机生成样本集合：
	//  使用均匀分布随机生成X样本(x1,x2)，x范围定为0-100；
	//  利用标准模型求出Y；
	//  在Y的基础上加入标准正态分布噪音

	//先给随机数分配存储空间
	float *uniform_rands=(float*)malloc(sizeof(float)*N*2); //由于需要X1，X2两个值，因此乘2
	float *normal_rands=(float*)malloc(sizeof(float)*N);

	float *d_uniform_rands;
	float *d_normal_rands;

	cudaError_t cudaResult=cudaSuccess;
	cudaResult=cudaMalloc((void**)&d_uniform_rands,sizeof(float)*N*2);
	cudaResult=cudaMalloc((void**)&d_normal_rands,sizeof(float)*N);
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

	curandResult = curandGenerateUniform(prng, (float *)d_uniform_rands, N*2);
	curandResult = curandGenerateNormal(prng, (float *)d_normal_rands, N,0.0,1.0);
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

	dim3 grid, block;
	block.x=128;
	grid.x=(N+block.x-1)/block.x;

	generateSamples<<<grid,block>>>(d_uniform_rands,d_normal_rands);
	cudaResult=cudaMemcpy(uniform_rands, d_uniform_rands, N*sizeof(float)*2, cudaMemcpyDeviceToHost); 
	if (cudaResult!=cudaSuccess)
	{
		string msg("Could not memcpy: ");
		msg += cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}
	cudaResult=cudaMemcpy(normal_rands, d_normal_rands, N*sizeof(float), cudaMemcpyDeviceToHost); 
	if (cudaResult!=cudaSuccess)
	{
		string msg("Could not memcpy: ");
		msg += cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}
	float *sumX2s=(float*)malloc(sizeof(float)*grid.x);
	float *sumX2Squares=(float*)malloc(sizeof(float)*grid.x);

	float *sumX3s=(float*)malloc(sizeof(float)*grid.x);
	float *sumX3Squares=(float*)malloc(sizeof(float)*grid.x);

	float *sumYs=(float*)malloc(sizeof(float)*grid.x);

	float *sumX2X3s=(float*)malloc(sizeof(float)*grid.x);
	float *sumX2Ys=(float*)malloc(sizeof(float)*grid.x);
	float *sumX3Ys=(float*)malloc(sizeof(float)*grid.x);

	float *d_sumX2s,*d_sumX2Squares;
	float *d_sumX3s,*d_sumX3Squares;
	float *d_sumYs;
	float *d_sumX2X3s,*d_sumX2Ys,*d_sumX3Ys;
	int *d_count;

	cudaResult=cudaMalloc((void**)&d_sumX2s,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX2Squares,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX3s,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX3Squares,sizeof(float)*grid.x);

	cudaResult=cudaMalloc((void**)&d_sumYs,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX2X3s,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX2Ys,sizeof(float)*grid.x);
	cudaResult=cudaMalloc((void**)&d_sumX3Ys,sizeof(float)*grid.x);

	cudaResult=cudaMalloc((void**)&d_count,sizeof(int)*N);

	//2.调用规约方法进行variance计算
	computeWeights<<<grid,block,block.x*sizeof(float),0>>>(d_sumX2s,d_sumX2Squares,d_sumX3s,d_sumX3Squares,d_sumYs,d_sumX2X3s,d_sumX2Ys,d_sumX3Ys,d_uniform_rands,d_normal_rands,d_count);
	cudaDeviceSynchronize();

	cudaResult=cudaMemcpy(sumX2s, d_sumX2s, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX2Squares, d_sumX2Squares, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX3s, d_sumX3s, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX3Squares, d_sumX3Squares, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 

	cudaResult=cudaMemcpy(sumYs, d_sumYs, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX2X3s, d_sumX2X3s, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX2Ys, d_sumX2Ys, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaResult=cudaMemcpy(sumX3Ys, d_sumX3Ys, grid.x*sizeof(float), cudaMemcpyDeviceToHost); 

	cudaDeviceSynchronize();
	//cout<<"grid.x: "<<grid.x<<endl;

	float beta1,beta2,beta3;
	float sumX2=0,sumX2Square=0,sumX3=0,sumX3Square=0,sumY=0,sumX2X3=0,sumX2Y=0,sumX3Y=0;

	for(int i=0;i<grid.x;i++)
	{
		sumX2+=sumX2s[i];
		sumX2Square+=sumX2Squares[i];
		sumX3+=sumX3s[i];
		sumX3Square+=sumX3Squares[i];

		sumY+=sumYs[i];
		sumX2X3+=sumX2X3s[i];
		sumX2Y+=sumX2Ys[i];
		sumX3Y+=sumX3Ys[i];
	}

	//3.验证返回值的正确性

	float sumx2x3=sumX2X3-1/static_cast<float>(N)*sumX3*sumX2;
	float sumx2Square=sumX2Square-1/static_cast<float>(N)*sumX2*sumX2;
	float sumx3Square=sumX3Square-1/static_cast<float>(N)*sumX3*sumX3;
	float sumx2y=sumX2Y-1/static_cast<float>(N)*sumY*sumX2;
	float sumx3y=sumX3Y-1/static_cast<float>(N)*sumY*sumX3;
	//cout<<"$$ "<<1/N*sumY*sumX2<<endl;

	beta2=(sumx2y*sumx3Square-sumx3y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
	beta3=(sumx3y*sumx2Square-sumx2y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
	beta1=(sumY-beta2*sumX2-beta3*sumX3)/static_cast<float>(N);

	//cout<<"sumX2: "<<sumX2<<endl;
	//cout<<"sumX2Square: "<<sumX2Square<<endl;
	//cout<<"sumX3: "<<sumX3<<endl;
	//cout<<"sumX3Square: "<<sumX3Square<<endl;
	//cout<<"sumY: "<<sumY<<endl;
	//cout<<"sumX2X3: "<<sumX2X3<<endl;
	//cout<<"sumX2Y: "<<sumX2Y<<endl;
	//cout<<"sumX3Y: "<<sumX3Y<<endl;

	//cout<<endl;
	//cout<<"sumx2Square: "<<sumx2Square<<endl;
	//cout<<"sumx3Square: "<<sumx3Square<<endl;
	//cout<<"sumx2x3: "<<sumx2x3<<endl;
	//cout<<"sumx2y: "<<sumx2y<<endl;
	//cout<<"sumx3y: "<<sumx3y<<endl;

	cout<<endl;
	cout<<"gold   : "<<b1<<" "<<b2<<" "<<b3<<endl;
	cout<<"result : "<<beta1<<" "<<beta2<<" "<<beta3<<endl;

	/*
	for(int i=0;i<100;i++)
	{
		cout<<"X: "<<uniform_rands[i]<<" "<<uniform_rands[i+N]<<"  Y: "<<normal_rands[i]<<endl;
	}
	*/
	cout<<endl;

	cudaFree(d_uniform_rands);
	cudaFree(d_normal_rands);

	cudaFree(d_sumX2s);
	cudaFree(d_sumX2Squares);
	cudaFree(d_sumX3s);
	cudaFree(d_sumX3Squares);

	cudaFree(d_sumYs);
	cudaFree(d_sumX2X3s);
	cudaFree(d_sumX2Ys);
	cudaFree(d_sumX3Ys);
	cudaFree(d_count);
	
	free(uniform_rands);
	free(normal_rands);
	free(sumX2s);
	free(sumX2Squares);
	free(sumX3s);
	free(sumX3Squares);

	free(sumYs);
	free(sumX2X3s);
	free(sumX2Ys);
	free(sumX3Ys);
	int pi=0;
	//CUdevice dev;
	//cuDeviceGet(&dev,0);
	cuDeviceGetAttribute(&pi,CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,0);

	return 0;
}


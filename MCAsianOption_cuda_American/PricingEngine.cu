//2022.7.17--2022.7.27
//cuda美式期权MC
#include"PricingEngine.h"
#include<ctime>
#include<iostream>
#include<cstdio>

#include<cuda_runtime.h>
#include<cooperative_groups.h> //cg
#include<vector> 
#include<string> //进行异常处理时用于抛出异常信息
#include<numeric> //利用accumulate求均值
#include<stdexcept> //使用throw 进行异常处理
#include<typeinfo> //使用typeid函数，用于识别当前数据类型Real是double还是float
namespace cg=cooperative_groups;
#include<curand_kernel.h> //curand，gpu上的随机数生成器
#include"AsianOption.h" //用于期权参数传递的数据结构
#include"cudasharedmem.h" //在reduce_sum函数中使用shared memory
#include"helper_timer.h"
using std::string;
using std::vector;
#define checkCUDAErrors(call,str)do{ \
	const cudaError_t error=call; \
	if(error!=cudaSuccess){ \
		fprintf(stderr,"CUDA runtime error (%s:%d): ",__FILE__,__LINE__); \
		fprintf(stderr,"%s",str); \
		fprintf(stderr," %s \n",cudaGetErrorString(cudaGetLastError())); \
		exit(EXIT_FAILURE); \
	} \
}while(0); \

#define ORDER 3 //多项式阶数固定为3，此版本中多项式阶数不可更改
__device__ int numSample=0;
__device__ int is_locked=0;
__device__ unsigned int d_counter=0;

__global__ void test()
{
	printf("OK");
	return ;
}

template <typename Real>
__device__ Real dabs(Real x)
{
	if(x>0.0)
		return x;
	else
		return -x;
}
//用于初始化随机数状态，每个thread拥有自己的随机数状态
__global__ void initRNG(curandState *const rngStates,
					    const unsigned int seed)
{
	//printf("OK");
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}
__global__ void initRNG_Philox(curandStatePhilox4_32_10_t *const rngStates,
					    const unsigned int seed)
{
	//printf("OK");
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}

//getPathStep 的两个版本：支持 float 和 double
__device__ inline float getPathStep(float &drift,float &diffusion,curandState &state)
{
	return expf(drift+diffusion*curand_normal(&state));
}

__device__ inline double getPathStep(double &drift,double &diffusion,curandState &state)
{
	return exp(drift+diffusion*curand_normal_double(&state));
}

__device__ inline float getPathStep(float &drift,float &diffusion,curandStatePhilox4_32_10_t &state)
{
	return expf(drift+diffusion*curand_normal(&state));
}

__device__ inline double getPathStep(double &drift,double &diffusion,curandStatePhilox4_32_10_t &state)
{
	return exp(drift+diffusion*curand_normal_double(&state));
}
//MC第一步
//重要组成部分：随机路径生成函数
template <typename Real>
__global__ void generatePaths(Real *const paths,
							  curandState *const rngStates,
							  const AsianOption<Real> *const option,
							  const unsigned int numSims,
							  const unsigned int numTimesteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目
	//printf("%d %d \n",tid, step);
	//printf("\n");

	//1.漂移项
	Real drift=(option->r-static_cast<Real>(0.5)*option->sigma*option->sigma)*option->dt;
	//2.扩散项
	Real diffusion = option->sigma * sqrt(option->dt);

	// 引入局部随机数状态（保证各个thread的随机数状态不同）
	// 该thread内产生的随机数（可以是不同路径）都是来自该状态下的随机数生成器
	curandState localState=rngStates[tid];
	//计算分配到自己线程的路径
	//s为资产价格，迭代生成；paths存储资产价格结果
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		Real *output=paths+i;	//paths[i]
		Real s=static_cast<Real>(1); // s = 1
		for(unsigned int t=0;t<numTimesteps;t++,output+=numSims)
		{
			s*=getPathStep(drift,diffusion,localState);
			*output=s;	//paths[i]=s;
		}
	}
}
template <typename Real>
__global__ void generatePaths_Philox(Real *const paths,
//__global__ void generatePaths(Real *const paths,
							  curandStatePhilox4_32_10_t *const rngStates,
							  const AsianOption<Real> *const option,
							  const unsigned int numSims,
							  const unsigned int numTimesteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目
	//1.漂移项
	Real drift=(option->r-static_cast<Real>(0.5)*option->sigma*option->sigma)*option->dt;
	//2.扩散项
	Real diffusion = option->sigma * sqrt(option->dt);

	// 引入局部随机数状态（保证各个thread的随机数状态不同）
	// 该thread内产生的随机数（可以是不同路径）都是来自该状态下的随机数生成器
	curandStatePhilox4_32_10_t localState=rngStates[tid];
	//计算分配到自己线程的路径
	//s为资产价格，迭代生成；paths存储资产价格结果
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		Real *output=paths+i;	//paths[i]
		Real s=static_cast<Real>(1); // s = 1
		for(unsigned int t=0;t<numTimesteps;t++,output+=numSims)
		{
			s*=getPathStep(drift,diffusion,localState);
			*output=s;	//paths[i]=s;
			//@@@
			//if(tid==0)
			//	printf("%f ",*output);
		}
	}
	//if(tid==0)
	//	printf("\n");
}

//在同一个 cooperative group 中归约求和函数
//本质是线程块内部归约
//被下一个函数computeValue调用
template<typename Real>
__device__ Real reduce_sum(Real in, 
						   cg::thread_block cta
						   )
{
	SharedMemory<Real> sdata;
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
//拉盖尔多项式，分为三阶（0，1，2）
//作为基函数用于拟合
template<typename Real>
__device__ inline Real laguerre(Real x, int order)
{
    Real LaguerrePolynome =0;
    if(order==0)
    {
        LaguerrePolynome=1;

    }
    else
    {
        if(order==1)
        {
            LaguerrePolynome=(1-x);
        }
        else
        {
            if(order==2)
                LaguerrePolynome=(1-2*x+0.5*x*x);
        }
    }

    return LaguerrePolynome;
}

//+美式期权：回溯寻优
//首先，我们生成并存储了全部模拟路径在path中
//然后我们从最后一个时间点回溯到最初始的时间点
//每次都取出当前时间点下全部资产价格和payoff进行回归拟合（相当于所有线程协同进行），获得基函数权值后可以计算持有价值
//每一步中比较执行payoff和持有价值的大小，以确认是否更新最佳执行时间点
//将每条路径的最佳时间点记录在数组exec中备用
//注：本函数中必要使用全局的同步函数，因此要用上cooperative_groups
/*
template<typename Real>
__global__ void backtracking(const Real *const paths,
							 const AsianOption<Real> *const option,
							 const unsigned int numSims,
							 const unsigned int numTimesteps,
							 Real *const sampleX, //记录用于拟合的样本X
							 Real *const sampleY, //记录用于拟合的样本Y
							 Real *const sampleFlag, //记录是否为拟合样本
							 Real *const execPayoff, //记录各个期权路径的最佳执行结果
							 Real *const execTime, //记录各个期权路径的最佳执行时间
							 Real *const sumX2s,
							 Real *const sumX2Squares,
							 Real *const sumX3s,
							 Real *const sumX3Squares,
							 Real *const sumYs,
							 Real *const sumX2X3s,
							 Real *const sumX2Ys,
							 Real *const sumX3Ys,
							 Real *const counts,
							 Real *const weights
							 )
{
	//本函数中一个线程对应一整条路径，各个线程需要协调进行拟合函数的计算，因此需要多次（numTimesteps）的同步
	cg::thread_block cta=cg::this_thread_block();
	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	//unsigned int step=gridDim.x*blockDim.x; //thread数目

	const Real *path=paths+numSims*(numTimesteps-1)+tid; //paths[tid]
	Real payoff=0.0;
	Real continuation_value=0.0;

	//求出最终时刻的payoff，作为当前的最优payoff
	payoff=(*path)*option->spot-option->strike; //St-K，即为看涨期权payoff求出方式
	if(!(option->is_call)) //如果目标为看跌期权，进行翻转
	{
		payoff=-payoff;
	}
	//消除payoff的负值，期权payoff必须为非负数
	payoff=max(static_cast<Real>(0),payoff); // payoff=max(0,payoff)
	execPayoff[tid]=payoff;
	execTime[tid]=numTimesteps-1;

	Real X2=0.0;
	Real X2Square=0.0;
	Real X3=0.0;
	Real X3Square=0.0;
	Real Y_=0.0;
	Real X2X3=0.0;
	Real X2Y=0.0;
	Real X3Y=0.0;
	Real count=0.0;

	//if(numTimesteps-2<=0) do what
	//else do
	//开始回溯时间点
	path-=numSims;
	for(unsigned int t=numTimesteps-2;t>0;t--,path-=numSims)
	{
		//@@@
		//if(tid==1024575)
		//	printf("%f ",*path);
		//+第一步:计算当前线程在当前时间步下的payoff
		payoff=(*path)*option->spot-option->strike; //St-K，即为看涨期权payoff求出方式
		if(!(option->is_call)) //如果目标为看跌期权，进行翻转
		{
			payoff=-payoff;
		}

		//消除payoff的负值，期权payoff必须为非负数
		payoff=max(static_cast<Real>(0),payoff); // payoff=max(0,payoff)
		//if(tid==0||tid==11128)
		//printf("%d-%d %f-%f \n",tid,t,(*path)*option->spot,payoff);

		//+第二步：准备好拟合所用的样本，要求是当前时刻的payoff!=0
		if(payoff>0)
		{
			//利用拉盖尔多项式计算各项X
			//sampleX[tid]=laguerre(*path,1);
			//sampleX[tid+numSims]=laguerre(*path,2);
			//sampleY[tid]=execPayoff[tid]*exp(-option->r*option->dt*(execTime[tid]-t)); 
			sampleX[tid]=*path;
			sampleX[tid+numSims]=(*path)*(*path);
			sampleY[tid]=1.0+2.0*sampleX[tid]+3.0*sampleX[tid+numSims]; 
			sampleFlag[tid]=1;
		}
		else
		{
			sampleX[tid]=0; 
			//@@@ bug 没对sampleX[tid+numSims]置零
			sampleX[tid+numSims]=0; 
			sampleY[tid]=0; 
			sampleFlag[tid]=0;
		}

		//等待所有线程完成样本载入
		//cg::sync(cta); //<<同步
		//@@@
		//if(tid==8191&&t==numTimesteps-4)
		//{
		//	for(int i=0;i<numSims;i++)
		//	{
		//		printf("%d %f %f %f %f %f\n",i,max(static_cast<Real>(0),-*(path+i)*option->spot+option->strike),sampleX[i],sampleX[i+numSims],sampleY[i],sampleFlag[i]);
		//	}
		//}


		//+第三步：把准备好的样本进行拟合（path,execPayoff），求出拟合参数，存在weights里
		//这一步中需要tid==0的线程调用拟合程序，而其他tid!=0的程序挂起，实现方式为设置cg同步

		//读取所有样本进行拟合
		//计算用于求解最小二乘的八个统计量
		X2=reduce_sum(sampleX[tid],cta);
		X2Square=reduce_sum(sampleX[tid]*sampleX[tid],cta);
		X3=reduce_sum(sampleX[tid+numSims],cta);
		X3Square=reduce_sum(sampleX[tid+numSims]*sampleX[tid+numSims],cta);
		Y_=reduce_sum(sampleY[tid],cta);
		X2X3=reduce_sum(sampleX[tid]*sampleX[tid+numSims],cta);
		X2Y=reduce_sum(sampleX[tid]*sampleY[tid],cta);
		X3Y=reduce_sum(sampleX[tid+numSims]*sampleY[tid],cta);

		count=reduce_sum(sampleFlag[tid],cta);
		//printf("%f,%d\n",count,numSims);
		//count=numSims;
		cg::sync(cta); //<<同步

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
			counts[bid]=count;
		}	

		//cg::sync(cta); //<<同步
		if(tid==0)
		{
			Real sumX2=0,sumX2Square=0,sumX3=0,sumX3Square=0,sumY=0,sumX2X3=0,sumX2Y=0,sumX3Y=0;
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
				totalCount+=counts[i];
			}


			if(totalCount==0)
				continue;
			Real sumx2x3=sumX2X3-1/static_cast<Real>(totalCount)*sumX3*sumX2;
			Real sumx2Square=sumX2Square-1/static_cast<Real>(totalCount)*sumX2*sumX2;
			Real sumx3Square=sumX3Square-1/static_cast<Real>(totalCount)*sumX3*sumX3;
			Real sumx2y=sumX2Y-1/static_cast<Real>(totalCount)*sumY*sumX2;
			Real sumx3y=sumX3Y-1/static_cast<Real>(totalCount)*sumY*sumX3;

			weights[1]=(sumx2y*sumx3Square-sumx3y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
			weights[2]=(sumx3y*sumx2Square-sumx2y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
			weights[0]=(sumY-weights[1]*sumX2-weights[2]*sumX3)/static_cast<Real>(totalCount);
			//@@@
			//if(t==numTimesteps-2)
			//if(t==122)
			//printf("%d %f %f %f %f\n",t,weights[0],weights[1],weights[2],totalCount);
			//if(dabs(weights[0]-1.0)>0.5||dabs(weights[1]-2.0>0.5)||dabs(weights[2]-3.0>0.5))
			//	printf("@@%d %f %f %f %f %f %f %f %f %f\n",t,sumX2/totalCount,sumX2Square/totalCount,sumX3/totalCount,sumX3Square/totalCount,sumY/totalCount,sumX2Y/totalCount,sumX3Y/totalCount,sumX2X3/totalCount,totalCount);
			//else
			//	printf("--%d %f %f %f %f %f %f %f %f %f\n",t,sumX2/totalCount,sumX2Square/totalCount,sumX3/totalCount,sumX3Square/totalCount,sumY/totalCount,sumX2Y/totalCount,sumX3Y/totalCount,sumX2X3/totalCount,totalCount);
			//if(t%100==0)

			//@@@
			//if(tid==0&&t==numTimesteps-4)
			//	printf("%d\n",totalCount);

			printf("--%d %f %f %f %f %f %f %f %f %d\n",t,sumX2/static_cast<Real>(totalCount),sumX2Square/static_cast<Real>(totalCount),sumX3/static_cast<Real>(totalCount),sumX3Square/static_cast<Real>(totalCount),sumY/static_cast<Real>(totalCount),sumX2Y/static_cast<Real>(totalCount),sumX3Y/static_cast<Real>(totalCount),sumX2X3/static_cast<Real>(totalCount),totalCount);
		}

		//拟合结果存在weights中
		//cg::sync(cta); //<<同步

		//+第四步：各个线程独立使用拟合好的式子（读取weights）和当前资产价格，计算各自的continuation value
		//对于当前执行payoff不为零的路径，比较continuation value和当前时刻执行的payoff，决定是否更新
		if(payoff>0)
		{
			continuation_value=weights[0]+weights[1]*(*path)+weights[2]*(*path)*(*path);
			//continuation_value=weights[0]+weights[1]*laguerre(*path,1)+weights[2]*laguerre(*path,2);
			if(payoff>continuation_value)
			{
				//printf("tid=%d in %d:  %f instead %f\n",tid,t,payoff,continuation_value);
				execPayoff[tid]=payoff;
				execTime[tid]=t;
			}	
		}
		
		//cg::sync(cta); //<<同步
	}
	//printf("%f\n",execPayoff[tid]);
	return;
}
*/

//一、计算payoff，填充样本
template<typename Real>
__global__ void fillSamples(Real *const paths,
							 const AsianOption<Real> *const option,
							 const unsigned int numSims,
							 const unsigned int numTimesteps,
							 Real *const sampleX, //记录用于拟合的样本X
							 Real *const sampleY, //记录用于拟合的样本Y
							 Real *const sampleFlag, //记录是否为拟合样本
							 Real *const execPayoff, //记录各个期权路径的最佳执行结果
							 Real *const execTime, //记录各个期权路径的最佳执行时间
							 const int t //当前回溯时间点
							 )
{
	//cg::thread_block cta=cg::this_thread_block();
	//unsigned int bid=blockIdx.x; //块索引
	//unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	//unsigned int step=gridDim.x*blockDim.x; //thread数目

	Real *path=paths+numSims*t+tid; //paths[tid]
	Real payoff=0.0;

	//求出最终时刻的payoff，作为当前的最优payoff
	payoff=(*path)*option->spot-option->strike; //St-K，即为看涨期权payoff求出方式
	if(!(option->is_call)) //如果目标为看跌期权，进行翻转
	{
		payoff=-payoff;
	}
	//消除payoff的负值，期权payoff必须为非负数
	payoff=max(static_cast<Real>(0),payoff); // payoff=max(0,payoff)
	//如果是首次回溯则直接写入exec，无需拟合
	//否则进行样本填充，以备后续拟合
	if(t==numTimesteps-1)
	{
		execPayoff[tid]=payoff;
		execTime[tid]=t;
		return ;
	}
	else
	{
		if(payoff>0)
		{
			//利用拉盖尔多项式计算各项X
			sampleX[tid]=laguerre(*path,1);
			sampleX[tid+numSims]=laguerre(*path,2);
			sampleY[tid]=execPayoff[tid]*exp(-option->r*option->dt*(execTime[tid]-t)); 
		}
		else
		{
			sampleX[tid]=0; 
			sampleX[tid+numSims]=0; 
			sampleY[tid]=0; 
		}
		sampleFlag[tid]=payoff;
	}

	/*
	int flag=1;

	int tmp_count;
	if(payoff>0)
	{
		while(flag)
		{
			//is_locked=0时，flag=0；is_locked=1时，flag=1
			//flag获取当前锁的状态，锁为0则当前锁为释放状态，可以获取；
			//锁为1，说明锁为锁住状态，不能获取。
			flag=atomicCAS(&is_locked,0,1); //获取锁的状态，成功获取则返回锁之前的状态（即释放状态，锁为0），flag=0
			if(!flag) //获取到锁，可以操作临界资源
			{
				sampleX[tid]=laguerre(*path,1);
				sampleX[tid+numSims]=laguerre(*path,2);
				sampleY[tid]=execPayoff[tid]*exp(-option->r*option->dt*(execTime[tid]-t)); 

				tmp_count=d_counter++;
				atomicExch(&is_locked,0);	//解锁，将锁置为释放状态
			}
		}
		//printf("%d-- %f : %d\n",tid,payoff,tmp_count);
	}
	*/

	return ;
}
//二、拟合
template<typename Real>
__global__ void fitting(const unsigned int numSims,
						Real *const sampleX, //记录用于拟合的样本X
						 Real *const sampleY, //记录用于拟合的样本Y
						 Real *const sampleFlag, //记录是否为拟合样本
						 Real *const sumX2s,
						 Real *const sumX2Squares,
						 Real *const sumX3s,
						 Real *const sumX3Squares,
						 Real *const sumYs,
						 Real *const sumX2X3s,
						 Real *const sumX2Ys,
						 Real *const sumX3Ys,
						 Real *const counts,
						 Real *const weights
							 )
{
	cg::thread_block cta=cg::this_thread_block();
	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	//+第三步：把准备好的样本进行拟合（path,execPayoff），求出拟合参数，存在weights里
		//这一步中需要tid==0的线程调用拟合程序，而其他tid!=0的程序挂起，实现方式为设置cg同步

	//读取所有样本进行拟合
	//计算用于求解最小二乘的八个统计量
	Real X2=reduce_sum(sampleX[tid],cta);
	Real X2Square=reduce_sum(sampleX[tid]*sampleX[tid],cta);
	Real X3=reduce_sum(sampleX[tid+numSims],cta);
	Real X3Square=reduce_sum(sampleX[tid+numSims]*sampleX[tid+numSims],cta);
	Real Y_=reduce_sum(sampleY[tid],cta);
	Real X2X3=reduce_sum(sampleX[tid]*sampleX[tid+numSims],cta);
	Real X2Y=reduce_sum(sampleX[tid]*sampleY[tid],cta);
	Real X3Y=reduce_sum(sampleX[tid+numSims]*sampleY[tid],cta);

	Real c=sampleFlag[tid]>0?1:0;
	Real count=reduce_sum(c,cta);
	//printf("%f,%d\n",count,numSims);
	//count=numSims;
	//cg::sync(cta); //<<同步

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
		counts[bid]=count;
		//__threadfence();
	}	

	/*
	cg::sync(cta);
	if(threadIdx.x==0)
	{
		atomicAdd(&d_counter,1);
		while(d_counter!=gridDim.x)
		{
			//printf("%d wait: %d\n",tid,d_counter);
		}
	}
	cg::sync(cta);
	*/

	//cg::sync(cta); //<<同步
	if(tid==0)
	{
		/*
		int mc=0;
		for(int i=0;i<numSims;i++)
		{
			if(sampleFlag[i]>0)
			{
				mc++;
			}
		}
		printf("mcount %d\n",mc);
		*/

		Real sumX2=0,sumX2Square=0,sumX3=0,sumX3Square=0,sumY=0,sumX2X3=0,sumX2Y=0,sumX3Y=0;
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
			totalCount+=counts[i];
		}
		//printf("counter: %d\n",d_counter);
		//printf("total Count: %d\n",totalCount);
		if(totalCount==0)
		{
			for(int i=0;i<3;i++)
			{
				weights[i]=0.0;
			}
			return;
		}
		Real sumx2x3=sumX2X3-1/static_cast<Real>(totalCount)*sumX3*sumX2;
		Real sumx2Square=sumX2Square-1/static_cast<Real>(totalCount)*sumX2*sumX2;
		Real sumx3Square=sumX3Square-1/static_cast<Real>(totalCount)*sumX3*sumX3;
		Real sumx2y=sumX2Y-1/static_cast<Real>(totalCount)*sumY*sumX2;
		Real sumx3y=sumX3Y-1/static_cast<Real>(totalCount)*sumY*sumX3;

		weights[1]=(sumx2y*sumx3Square-sumx3y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
		weights[2]=(sumx3y*sumx2Square-sumx2y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
		weights[0]=(sumY-weights[1]*sumX2-weights[2]*sumX3)/static_cast<Real>(totalCount);
	}
}

//三、更新exec
template<typename Real>
__global__ void updateExec(Real *const paths,
						   const unsigned int numSims,
						   Real *const sampleFlag, //记录是否为拟合样本
						   Real *const execPayoff, //记录各个期权路径的最佳执行结果
						   Real *const execTime, //记录各个期权路径的最佳执行时间
						   Real *const weights,
						   const int  t)
{
	//cg::thread_block cta=cg::this_thread_block();
	//unsigned int bid=blockIdx.x; //块索引
	//unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	Real *path=paths+numSims*t+tid; //paths[tid]
	Real continuation_value=0.0;
	/*
	//如果样本数为0则中止程序
	if(weights[0]==0&&weights[1]==0&&weights[2]==0)
		return;
	*/
	if(sampleFlag[tid]>0)
	{
		//continuation_value=weights[0]+weights[1]*(*path)+weights[2]*(*path)*(*path);
		continuation_value=weights[0]+weights[1]*laguerre(*path,1)+weights[2]*laguerre(*path,2);
		if(sampleFlag[tid]>continuation_value)
		{
			//printf("tid=%d in %d:  %f instead %f\n",tid,t,payoff,continuation_value);
			execPayoff[tid]=sampleFlag[tid];
			execTime[tid]=t;
		}	
	}
	
	/*
	cg::sync(cta);
	if(threadIdx.x==0)
	{
		atomicAdd(&d_counter,1);
		while(d_counter!=gridDim.x)
		{
			//printf("%d wait: %d\n",tid,d_counter);
		}
	}
	cg::sync(cta);
	*/


}



//MC第二步
//定价函数 （调用上一个函数reduce_sum）
//计算每条路径payoff并求其期望，以定价
template<typename Real>
__global__ void computeValue(Real *const sum,
							 Real *const sumSquare,
					         const Real *const execPayoff,  //记录最佳payoff
					         const Real *const execTime,    //记录最佳payoff的执行时间
							 const AsianOption<Real> *const option,
							 const unsigned int numSims,
							 const unsigned int numTimesteps)
{
	//声明在同一个cg中
	cg::thread_block cta=cg::this_thread_block();

	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	unsigned int step=gridDim.x*blockDim.x; //thread数目

	Real sumPayoffs=static_cast<Real>(0); //sumPayoffs = 0
	Real sumPayoffSquares=static_cast<Real>(0); //sum(Payoff^2) = 0
	//根据资产价格路径 paths 内的值求payoff
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		Real payoff=execPayoff[tid]*exp(-option->r*option->dt*(execTime[tid]+1)); 
		//printf("%d, %f\n",tid,payoff);
		sumPayoffs+=payoff; //因为最终要求payoff的均值，因此先加和
		sumPayoffSquares+=payoff*payoff; //payoff的平方之和，用于计算方差
	}

	//块内的所有payoff求个均值
	sumPayoffs=reduce_sum<Real>(sumPayoffs,cta);
	sumPayoffSquares=reduce_sum<Real>(sumPayoffSquares,cta);

	//将所有块内均值结果存在values中，等待再次归约（在主机端）
	if(threadIdx.x==0)
	{
		sum[bid]=sumPayoffs;
		sumSquare[bid]=sumPayoffSquares;
	}
}

//PricingEngine的 构造函数
template<typename Real>
PricingEngine<Real>::PricingEngine(unsigned int numSims, 
								   unsigned int device,
								   int numDev,
								   unsigned int threadBlockSize,
								   unsigned int seed,
								   int timerON
								  )
								 :m_numSims(numSims),
								  m_device(device),
								  m_numDev(numDev),
								  m_threadBlockSize(threadBlockSize),
								  m_seed(seed),
								  m_timerON(timerON)
{
	return ;
}

// “()” 符号重载——PricingEngine 类中唯一的一个成员函数
//包含整个定价过程：随机数生成器（RNG）初始化，MC第一步-价格路径生成，MC第二步-payoff计算以及最终产生定价(price)
template <typename Real>
void PricingEngine<Real>::operator()(AsianOption<Real> &option)
{
	//################

	//一、准备步骤
	//0.获取设备信息
	//1.检查精度是否可用（double是否支持）
	//2.将任务分配给某一个GPU
	//3.设定grid和block大小
	//4.获取各个核函数的属性，检测所能支持的最大block大小（initRNG，computeValue）
	//5.分配显存：
	//	(1)给option参数分配显存(d_option);
	//	(2)给paths分配显存(d_paths);
	//	(3)给RNG状态（state）分配显存(d_rngStates);
	//	(4)给value（各个block归约结果）分配显存(d_values);

	//二、正式MC步骤：
	//0.RNG初始化
	//1.生成路径
	//2.计算payoff
	//3.将GPU上各个block归约出的结果返回CPU进行最后归约
	//4.将价格折现

	//三、收尾工作（释放各种内存）
	//1.释放option参数显存;
	//2.释放paths显存;
	//3.释放RNG状态（state）显存;
	//4.释放value（各个block归约结果）显存

	//注：每一次调用cuda相关函数后都要进行异常处理，使用cudaError_t
	//能不能把这些重复的异常处理部分包装成一个函数？
	int nDev=0;
	cudaGetDeviceCount(&nDev);
	if(m_numDev>0&&m_numDev<=nDev)
	{
		nDev=m_numDev;
	}
	else
	{
		if(m_numDev>0)
		{
			std::cout<<"设备数目设置过大"<<std::endl;
			exit(EXIT_FAILURE);
		}
	}
	std::cout<<"设备数目："<<nDev<<std::endl;

	cudaStream_t *stream=(cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
	struct cudaDeviceProp *deviceProperties=(struct cudaDeviceProp*)malloc(sizeof(struct cudaDeviceProp)*nDev); //设备属性

	cudaError_t cudaResult=cudaSuccess; //cuda函数运行结果标识

	unsigned int deviceVersion; //设备版本（计算能力）


	StopWatchInterface *timer=NULL;
	StopWatchInterface *totalTimer=NULL;
	sdkCreateTimer(&timer);
	sdkCreateTimer(&totalTimer);

	if(m_timerON==1)
	{
		sdkStartTimer(&timer);
		sdkStartTimer(&totalTimer);
	}

	// ****************************************************
	// 一、准备步骤
	// ****************************************************

	//0.获取设备信息
	for(int d=0;d<nDev;d++)
	{
		string errorMsg("Could not get device properties: ");
		checkCUDAErrors(cudaGetDeviceProperties(&deviceProperties[d],d),errorMsg.c_str());
		
		//1.检查精度是否可用（double是否支持）
		//在1.3版本之前的GPU设备不支持双精度（double）
		deviceVersion=deviceProperties[d].major*10+deviceProperties[d].minor; //major为主版本号，minor为次版本号
		if(typeid(Real)==typeid(double) && deviceVersion<13)
		{
			//TODO 这里应该标明哪一个设备属性不支持
			throw std::runtime_error("Device does not have double precision support");
		}
	}

	struct cudaFuncAttributes *funcAttributes=(struct cudaFuncAttributes*)malloc(sizeof(struct cudaFuncAttributes)*3); //函数属性
	cudaResult=cudaFuncGetAttributes(&funcAttributes[0],initRNG);
	if(cudaResult!=cudaSuccess)
	{
		string msg("Could not get function attributes:");
		msg+=cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}

	cudaResult=cudaFuncGetAttributes(&funcAttributes[1],generatePaths<Real>);
	if(cudaResult!=cudaSuccess)
	{
		string msg("Could not get function attributes:");
		msg+=cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}

	cudaResult=cudaFuncGetAttributes(&funcAttributes[2],computeValue<Real>);
	if(cudaResult!=cudaSuccess)
	{
		string msg("Could not get function attributes:");
		msg+=cudaGetErrorString(cudaResult);
		throw std::runtime_error(msg);
	}

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<std::endl;
		std::cout<<"-----"<<"预备步骤"<<"-----"<<std::endl;
		std::cout<<"获取设备属性&函数属性耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	int numTimesteps=static_cast<int>(option.tenor/option.dt);
	AsianOption<Real> **d_option=(AsianOption<Real>**)calloc(nDev,sizeof(AsianOption<Real>*));
	Real **d_paths=(Real**)calloc(nDev,sizeof(Real*));
	curandStatePhilox4_32_10_t **d_rngStates=(curandStatePhilox4_32_10_t**)calloc(nDev,sizeof(curandStatePhilox4_32_10_t*));
	//curandState **d_rngStates=(curandState**)calloc(nDev,sizeof(curandState*));

	Real **sumX2s=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX2Squares=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX3s=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX3Squares=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumYs=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX2X3s=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX2Ys=(Real**)calloc(nDev,sizeof(Real*));
	Real **sumX3Ys=(Real**)calloc(nDev,sizeof(Real*));

	Real **counts=(Real**)calloc(nDev,sizeof(Real*));

	Real **d_values=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_valueSquares=(Real**)calloc(nDev,sizeof(Real*));

	Real **d_sampleX=(Real**)calloc(nDev,sizeof(Real*)); //+美式期权特有，用于存储拟合样本的X
	Real **d_sampleY=(Real**)calloc(nDev,sizeof(Real*)); //+美式期权特有，用于存储拟合样本的Y
	Real **d_sampleFlag=(Real**)calloc(nDev,sizeof(Real*)); //+美式期权特有，用于存储拟合样本的标识
	Real **d_execPayoff=(Real**)calloc(nDev,sizeof(Real*)); //+美式期权特有，用于存储当前最优payoff结果
	Real **d_execTime=(Real**)calloc(nDev,sizeof(Real*)); //+美式期权特有，用于存储当前最优payoff的执行时间

	//以下为求最小二乘拟合解所需的八个统计量
	Real **d_sumX2s=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX2Squares=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX3s=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX3Squares=(Real**)calloc(nDev,sizeof(Real*));

	Real **d_sumYs=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX2X3s=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX2Ys=(Real**)calloc(nDev,sizeof(Real*));
	Real **d_sumX3Ys=(Real**)calloc(nDev,sizeof(Real*));

	Real **d_counts=(Real**)calloc(nDev,sizeof(Real*));

	Real **d_weights=(Real**)calloc(nDev,sizeof(Real*));
	Real **weights=(Real**)calloc(nDev,sizeof(Real*));

	dim3 *block=(dim3*)malloc(sizeof(dim3)*nDev);
	dim3 *grid=(dim3*)malloc(sizeof(dim3)*nDev);
	int group=m_numSims/nDev;
	int border=m_numSims%nDev;

	int *numTasks=(int*)malloc(sizeof(int)*nDev);
	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"变量分配内存耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
	}

	for(int d=0;d<nDev;d++)
	{
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);

		int dn=group;
		if(d<border)
		{
			dn=group+1;
		}
		numTasks[d]=dn;
		printf("设备%d上执行的任务数：%d\n",d,numTasks[d]);

		cudaSetDevice(d);
		///
		if(m_timerON==1)
		{
			sdkStopTimer(&timer);
			std::cout<<std::endl;
			std::cout<<"-----设备 "<<d<<" : "<<deviceProperties[d].name<<"-----"<<std::endl;
			std::cout<<"切换设备耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
			sdkResetTimer(&timer);
			sdkStartTimer(&timer);
		}

		cudaStreamCreate(stream+d);
		cudaStreamQuery(stream[d]);

		///
		if(m_timerON==1)
		{
			sdkStopTimer(&timer);
			std::cout<<"创建cuda Stream耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
			sdkResetTimer(&timer);
			sdkStartTimer(&timer);
		}

		//3.设定grid和block大小
		block[d].x=m_threadBlockSize; 
		grid[d].x=(numTasks[d]+m_threadBlockSize-1)/m_threadBlockSize;
		grid[d].y=1;
		block[d].y=1;
		grid[d].z=1;
		block[d].z=1;

		///
		if(m_timerON==1)
		{
			sdkStopTimer(&timer);
			std::cout<<"核函数线程参数设置耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
			sdkResetTimer(&timer);
			sdkStartTimer(&timer);
		}

		//4.获取各个核函数的属性，检测所能支持的最大block大小（initRNG，computeValue）
		//(1)check initRNG
		
		if(block[d].x>(unsigned int)funcAttributes[0].maxThreadsPerBlock)
		{
			throw std::runtime_error("Block X dimension is too large for initRNG kernel");
		}
		//(2)check generatePaths
		
		if(block[d].x>(unsigned int)funcAttributes[1].maxThreadsPerBlock)
		{
			throw std::runtime_error("Block X dimension is too large for generatePaths kernel");
		}

		//(3)check computeValue
		
		if(block[d].x>(unsigned int)funcAttributes[2].maxThreadsPerBlock)
		{
			throw std::runtime_error("Block X dimension is too large for computeValue kernel");
		}
		///
		if(m_timerON==1)
		{
			sdkStopTimer(&timer);
			std::cout<<"函数检测过程耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
			sdkResetTimer(&timer);
			sdkStartTimer(&timer);
		}

		//5.分配显存：
		//	(1)给option参数分配显存(d_option);
		cudaResult=cudaMalloc((void**)&d_option[d],sizeof(AsianOption<Real>));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for option data: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		cudaResult=cudaMemcpy(d_option[d],&option,sizeof(AsianOption<Real>),cudaMemcpyHostToDevice);
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not copy data to device: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//	(2)给paths分配显存(d_paths);
		int numTimesteps=static_cast<int>(option.tenor/option.dt);
		cudaResult=cudaMalloc((void **)&d_paths[d],numTasks[d]*numTimesteps*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for paths: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//	(3)给RNG状态（state）分配显存(d_rngStates);

		cudaResult=cudaMalloc((void **)&d_rngStates[d],grid[d].x*block[d].x*sizeof(curandStatePhilox4_32_10_t));
		//cudaResult=cudaMalloc((void **)&d_rngStates[d],grid[d].x*block[d].x*sizeof(curandState));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for RNG state: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//	(4)给value（各个block归约结果）分配显存(d_values);
		cudaResult=cudaMalloc((void **)&d_values[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_valueSquares[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//+  (5)给美式期权的特有变量分配显存(sampleX,sampleY,execPayoff);
		cudaResult=cudaMalloc((void **)&d_sampleX[d],grid[d].x*block[d].x*sizeof(Real)*2);
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sampleY[d],grid[d].x*block[d].x*sizeof(Real)*2);
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sampleFlag[d],grid[d].x*block[d].x*sizeof(Real)*2);
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_execPayoff[d],grid[d].x*block[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_execTime[d],grid[d].x*block[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//为8个统计量分配空间（大小都为grid.x）
		cudaResult=cudaMalloc((void **)&d_sumX2s[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX2Squares[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX3s[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX3Squares[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		cudaResult=cudaMalloc((void **)&d_sumYs[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX2X3s[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX2Ys[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		cudaResult=cudaMalloc((void **)&d_sumX3Ys[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		cudaResult=cudaMalloc((void **)&d_counts[d],grid[d].x*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		//为拟合函数的权重分配空间（order=3：w1，w2，w3）
		cudaResult=cudaMalloc((void **)&d_weights[d],ORDER*sizeof(Real));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for partial results: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
		counts[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX2s[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX2Squares[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX3s[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX3Squares[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumYs[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX2X3s[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX2Ys[d]=(Real*)malloc(sizeof(Real)*grid[d].x);
		sumX3Ys[d]=(Real*)malloc(sizeof(Real)*grid[d].x);

		weights[d]=(Real*)malloc(ORDER*sizeof(Real));

		///
		if(m_timerON==1)
		{
			sdkStopTimer(&timer);
			std::cout<<"分配缓存耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		}

	}//end numTasks
	//cudaDeviceSynchronize();
	if(m_timerON==1)
	{
		std::cout<<std::endl;
		std::cout<<"-----MC step start-----"<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	// ****************************************************
	// 二、正式MC步骤：
	// ****************************************************
	
	//0.RNG初始化
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		//initRNG<<<grid[d],block[d],0,stream[d]>>>(d_rngStates[d],m_seed+d*3);
		initRNG_Philox<<<grid[d],block[d],0,stream[d]>>>(d_rngStates[d],m_seed+m_numSims*3);
		//cudaDeviceSynchronize();

	}

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-initRNG函数耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	//1.生成路径
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		//generatePaths<Real><<<grid[d],block[d],0,stream[d]>>>(d_paths[d],d_rngStates[d],d_option[d],numTasks[d],numTimesteps);
		generatePaths_Philox<Real><<<grid[d],block[d],0,stream[d]>>>(d_paths[d],d_rngStates[d],d_option[d],numTasks[d],numTimesteps);
		//cudaDeviceSynchronize();
	}

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-generatePaths函数耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	//+美式期权回溯过程
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		/*
		backtracking<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(
				d_paths[d],
				d_option[d],
				numTasks[d],
				numTimesteps,
				d_sampleX[d],
				d_sampleY[d],
				d_sampleFlag[d],
				d_execPayoff[d],
				d_execTime[d],
				d_sumX2s[d],
				d_sumX2Squares[d],
				d_sumX3s[d],
				d_sumX3Squares[d],
				d_sumYs[d],
				d_sumX2X3s[d],
				d_sumX2Ys[d],
				d_sumX3Ys[d],
				d_counts[d],
				d_weights[d]
				);
		*/
		//unsigned int counter=0;
		//for(int t=numTimesteps-1;t>numTimesteps-3;t--)
		for(int t=numTimesteps-1;t>=0;t--)
		{
			fillSamples<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(
					d_paths[d],
					d_option[d],
					numTasks[d],
					numTimesteps,
					d_sampleX[d],
					d_sampleY[d],
					d_sampleFlag[d],
					d_execPayoff[d],
					d_execTime[d],
					t
			);
			
			//cudaDeviceSynchronize();	
			//cudaMemcpyToSymbol(d_counter,&counter,sizeof(unsigned int));
			//std::cout<<"fill d_counter: "<<d_counter<<std::endl;
			/*
			cudaResult=cudaMemcpy(counts[d], d_counts[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			if (cudaResult!=cudaSuccess)
			{
				string msg("Could not memcpy: ");
				msg += cudaGetErrorString(cudaResult);
				throw std::runtime_error(msg);
			}
			*/
			//std::cout<<std::endl;

			if(t==numTimesteps-1)
				continue;

			fitting<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(
					numTasks[d],
					d_sampleX[d],
					d_sampleY[d],
					d_sampleFlag[d],
					d_sumX2s[d],
					d_sumX2Squares[d],
					d_sumX3s[d],
					d_sumX3Squares[d],
					d_sumYs[d],
					d_sumX2X3s[d],
					d_sumX2Ys[d],
					d_sumX3Ys[d],
					d_counts[d],
					d_weights[d]
			);
			cudaDeviceSynchronize();
			cudaResult=cudaMemcpy(sumX2s[d], d_sumX2s[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX2Squares[d], d_sumX2Squares[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX3s[d], d_sumX3s[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX3Squares[d], d_sumX3Squares[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 

			cudaResult=cudaMemcpy(sumYs[d], d_sumYs[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX2X3s[d], d_sumX2X3s[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX2Ys[d], d_sumX2Ys[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(sumX3Ys[d], d_sumX3Ys[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			cudaResult=cudaMemcpy(counts[d], d_counts[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 

	

			Real sumX2=0,sumX2Square=0,sumX3=0,sumX3Square=0,sumY=0,sumX2X3=0,sumX2Y=0,sumX3Y=0;
			int totalCount=0;

			for(int i=0;i<grid[d].x;i++)
			{
				sumX2+=sumX2s[d][i];
				sumX2Square+=sumX2Squares[d][i];
				sumX3+=sumX3s[d][i];
				sumX3Square+=sumX3Squares[d][i];

				sumY+=sumYs[d][i];
				sumX2X3+=sumX2X3s[d][i];
				sumX2Y+=sumX2Ys[d][i];
				sumX3Y+=sumX3Ys[d][i];
				totalCount+=counts[d][i];
			}
			//printf("counter: %d\n",d_counter);
			//printf("total Count: %d\n",totalCount);
			if(totalCount==0)
				continue;
			Real sumx2x3=sumX2X3-1/static_cast<Real>(totalCount)*sumX3*sumX2;
			Real sumx2Square=sumX2Square-1/static_cast<Real>(totalCount)*sumX2*sumX2;
			Real sumx3Square=sumX3Square-1/static_cast<Real>(totalCount)*sumX3*sumX3;
			Real sumx2y=sumX2Y-1/static_cast<Real>(totalCount)*sumY*sumX2;
			Real sumx3y=sumX3Y-1/static_cast<Real>(totalCount)*sumY*sumX3;

			weights[d][1]=(sumx2y*sumx3Square-sumx3y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
			weights[d][2]=(sumx3y*sumx2Square-sumx2y*sumx2x3)/(sumx2Square*sumx3Square-(sumx2x3)*(sumx2x3));
			weights[d][0]=(sumY-weights[d][1]*sumX2-weights[d][2]*sumX3)/static_cast<Real>(totalCount);
			//cudaMemcpyToSymbol(d_counter,&counter,sizeof(unsigned int));
			cudaMemcpy(d_weights[d],weights[d],sizeof(Real)*ORDER,cudaMemcpyHostToDevice);
			/*
			cudaResult=cudaMemcpy(counts[d], d_counts[d], grid[d].x*sizeof(Real), cudaMemcpyDeviceToHost); 
			if (cudaResult!=cudaSuccess)
			{
				string msg("Could not memcpy: ");
				msg += cudaGetErrorString(cudaResult);
				throw std::runtime_error(msg);
			}
			Real sumCounts=0.0;
			for(int i=0;i<grid[d].x;i++)
			{
				sumCounts+=counts[d][i];
			}
			//printf("%f\n",sumCounts);
			*/

			//cudaDeviceSynchronize();	
			//cudaMemcpyToSymbol(d_counter,&counter,sizeof(unsigned int));
			//std::cout<<"fit d_counter: "<<d_counter<<std::endl;

			updateExec<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(
					d_paths[d],
					numTasks[d],
					d_sampleFlag[d],
					d_execPayoff[d],
					d_execTime[d],
					d_weights[d],
					t
			);
			//cudaDeviceSynchronize();	
			//cudaMemcpyToSymbol(d_counter,&counter,sizeof(unsigned int));
			//std::cout<<"update d_counter: "<<d_counter<<std::endl;
		}
		
	}

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-backtracking函数耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	//2.计算payoff
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		computeValue<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(d_values[d],d_valueSquares[d],d_execPayoff[d],d_execTime[d],d_option[d],numTasks[d],numTimesteps);
		//cudaDeviceSynchronize();
	}

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-computeValue函数耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	//3.将GPU上各个block归约出的结果返回CPU进行最后归约
	Real S1=0.0,S2=0.0;
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		vector<Real> values(grid[d].x);
		cudaResult=cudaMemcpy(&values[0],d_values[d],grid[d].x*sizeof(Real),cudaMemcpyDeviceToHost);

		vector<Real> valueSquares(grid[d].x);
		cudaResult=cudaMemcpy(&valueSquares[0],d_valueSquares[d],grid[d].x*sizeof(Real),cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not copy partial results to host: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		Real deviceMean=std::accumulate(values.begin(),values.end(),static_cast<Real>(0));
		Real S1_partial=std::accumulate(valueSquares.begin(),valueSquares.end(),static_cast<Real>(0));

		deviceMean/=static_cast<Real>(numTasks[d]);
		S1_partial/=static_cast<Real>(numTasks [d]);
		option.price+=deviceMean;
		S1+=S1_partial;
	}
	S1/=static_cast<Real>(nDev);
	Real avg=option.price/static_cast<Real>(nDev);
	S2=avg*avg;
	Real variance=S1-S2;
	printf("var: %f\n",variance);
	option.error=sqrt(variance/m_numSims);
	printf("error: %f\n",option.error);
	
	//4.将价格折现
	option.price/=static_cast<Real>(nDev);

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-传输计算结果耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	// ****************************************************
	// 三、收尾工作（释放显存/内存）
	// ****************************************************

	for(int d=0;d<nDev;d++)
	{
		//1.释放option参数显存;
		if(d_option[d])	
		{
			cudaFree(d_option[d]);
		}
		//2.释放paths显存;
		if(d_paths[d])	
		{
			cudaFree(d_paths[d]);
		}
		//3.释放RNG状态（state）显存;
		if(d_rngStates[d])	
		{
			cudaFree(d_rngStates[d]);
		}
		//4.释放value（各个block归约结果）显存
		if(d_values[d])	
		{
			cudaFree(d_values[d]);
		}
		if(d_valueSquares[d])	
		{
			cudaFree(d_valueSquares[d]);
		}
		//+5.释放美式期权特有变量的显存
		//释放X,Y 样本集的显存
		if(d_sampleX[d])	
		{
			cudaFree(d_sampleX[d]);
		}
		if(d_sampleY[d])	
		{
			cudaFree(d_sampleY[d]);
		}
		if(d_sampleFlag[d])	
		{
			cudaFree(d_sampleFlag[d]);
		}

		//释放执行payoff的显存
		if(d_execPayoff[d])	
		{
			cudaFree(d_execPayoff[d]);
		}
		if(d_execTime[d])	
		{
			cudaFree(d_execTime[d]);
		}

		//释放8个统计量的显存
		if(d_sumX2s[d])	
		{
			cudaFree(d_sumX2s[d]);
		}
		if(d_sumX2Squares[d])	
		{
			cudaFree(d_sumX2Squares[d]);
		}
		if(d_sumX3s[d])	
		{
			cudaFree(d_sumX3s[d]);
		}
		if(d_sumX3Squares[d])	
		{
			cudaFree(d_sumX3Squares[d]);
		}

		if(d_sumYs[d])	
		{
			cudaFree(d_sumYs[d]);
		}
		if(d_sumX2X3s[d])	
		{
			cudaFree(d_sumX2X3s[d]);
		}
		if(d_sumX2Ys[d])	
		{
			cudaFree(d_sumX2Ys[d]);
		}
		if(d_sumX3Ys[d])	
		{
			cudaFree(d_sumX3Ys[d]);
		}

		if(d_counts[d])	
		{
			cudaFree(d_counts[d]);
		}

		//释放权重的显存
		if(d_weights[d])	
		{
			cudaFree(d_weights[d]);
		}

		//主机变量释放
		if(sumX2s[d])	
		{
			free(sumX2s[d]);
		}
		if(sumX2Squares[d])	
		{
			free(sumX2Squares[d]);
		}
		if(sumX3s[d])	
		{
			free(sumX3s[d]);
		}
		if(sumX3Squares[d])	
		{
			free(sumX3Squares[d]);
		}

		if(sumYs[d])	
		{
			free(sumYs[d]);
		}
		if(sumX2X3s[d])	
		{
			free(sumX2X3s[d]);
		}
		if(sumX2Ys[d])	
		{
			free(sumX2Ys[d]);
		}
		if(sumX3Ys[d])	
		{
			free(sumX3Ys[d]);
		}

		if(counts[d])	
		{
			free(counts[d]);
		}

		//释放权重的显存
		if(weights[d])	
		{
			free(weights[d]);
		}
	}

	free(d_option);
	free(d_paths);
	free(d_rngStates);
	free(d_values);
	free(d_valueSquares);

	free(d_sampleX);
	free(d_sampleY);
	free(d_sampleFlag);
	free(d_execPayoff);
	free(d_execTime);

	free(d_sumX2s);
	free(d_sumX2Squares);
	free(d_sumX3s);
	free(d_sumX3Squares);
	free(d_sumYs);
	free(d_sumX2X3s);
	free(d_sumX2Ys);
	free(d_sumX3Ys);

	free(d_counts);
	free(d_weights);

	free(sumX2s);
	free(sumX2Squares);
	free(sumX3s);
	free(sumX3Squares);
	free(sumYs);
	free(sumX2X3s);
	free(sumX2Ys);
	free(sumX3Ys);

	free(counts);
	free(weights);

	free(stream);
	free(deviceProperties);
	free(funcAttributes);
	free(numTasks);
	free(block);
	free(grid);
	//cudaDeviceSynchronize();

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"各种收尾工作耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkDeleteTimer(&timer);
	}

	if(m_timerON==1)
	{
		sdkStopTimer(&totalTimer);
		std::cout<<"函数内部总耗时："<<sdkGetAverageTimerValue(&totalTimer)/1000.0f<<std::endl;
		sdkDeleteTimer(&totalTimer);
	}
	cudaDeviceReset();
		
	return;
}
template class PricingEngine<float>;
template class PricingEngine<double>;


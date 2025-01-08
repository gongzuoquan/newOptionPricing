//2022.7.10
//Merton Diffusion-Jump model cuda版
//使用单浮点float类型数据
#include"PricingEngine.h"
#include<ctime>
#include<iostream>
#include<cstdio>

//#include<cuda_runtime.h>
#include<cooperative_groups.h> //cg
#include<vector> 
#include<string> //进行异常处理时用于抛出异常信息
#include<numeric> //利用accumulate求均值
#include<stdexcept> //使用throw 进行异常处理
#include<typeinfo> //使用typeid函数，用于识别当前数据类型Real是double还是float
namespace cg=cooperative_groups;
#include<curand_kernel.h> //curand，gpu上的随机数生成器
//#include"AsianOption.h" //用于期权参数传递的数据结构
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

__global__ void test()
{
	printf("OK");
	return ;
}
//用于初始化随机数状态，每个thread拥有自己的随机数状态
__global__ void initRNG(curandState *const rngStates,
					    const unsigned int seed)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	return ;
}
__global__ void initRNG_Philox(curandStatePhilox4_32_10_t *const rngStates,
					    const unsigned int seed)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
	curand_init(seed,tid,0,&rngStates[tid]);
	//curand_init(seed,tid*2,0,&rngStates[tid*2]);
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
							  const MertonDJAsianOption<Real> *const option,
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
__device__ inline float getPathStep(float &drift,float &diffusion,curandStatePhilox4_32_10_t &state,float &dJ_sum)
{
	return expf(drift+diffusion*curand_normal(&state)+dJ_sum);
}

__device__ inline double getPathStep(double &drift,double &diffusion,curandStatePhilox4_32_10_t &state,double &dJ_sum)
{
	return exp(drift+diffusion*curand_normal_double(&state)+dJ_sum);
}

template <typename Real>
__global__ void generatePaths_Philox(Real *const paths,
//__global__ void generatePaths(Real *const paths,
							  curandStatePhilox4_32_10_t *const rngStates,
							  const MertonDJAsianOption<Real> *const option,
							  const unsigned int numSims,
							  const unsigned int numTimesteps)
{
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //当前thread在全局的索引号
	unsigned int step=gridDim.x*blockDim.x; //当前应用程序中全部thread数目
	//1.漂移项
	Real kappa=exp(option->jump_mu)-1.0;
	Real drift=(option->r-static_cast<Real>(0.5)*option->sigma*option->sigma-option->lambda*kappa)*option->dt;
	//2.扩散项
	Real diffusion = option->sigma * sqrt(option->dt);

	//Real jump_mu=(option->jump_mu-0.5*option->jump_sigma*option->jump_sigma);
	Real jump_mu=option->jump_mu;
	Real jump_sigma=option->jump_sigma;
	Real lambda=option->lambda*option->dt;

	// 引入局部随机数状态（保证各个thread的随机数状态不同）
	// 该thread内产生的随机数（可以是不同路径）都是来自该状态下的随机数生成器
	curandStatePhilox4_32_10_t localState=rngStates[tid];
	curandStatePhilox4_32_10_t poissonState=rngStates[tid];
	//计算分配到自己线程的路径
	//s为资产价格，迭代生成；paths存储资产价格结果
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		Real *output=paths+i;	//paths[i]
		Real s=static_cast<Real>(1); // s = 1
		//Real s=option->spot;
		for(unsigned int t=0;t<numTimesteps;t++,output+=numSims)
		{
			Real dJ_sum=0.0;
			unsigned int poisson_rand=curand_poisson(&poissonState,lambda);
			//生成一个泊松分布随机数poisson_rand
			for(int j=0;j<poisson_rand;j++)
			{
				//dJ_sum+=jump_mu+jump_sigma*curand_normal(&state);
				dJ_sum+=jump_mu-0.5*jump_sigma*jump_sigma+jump_sigma*curand_normal(&localState);
			}
			//s *= expf(drift + diffusion*curand_normal(&localState) + dJ_sum);
			//s*=getPathStep(drift,diffusion,localState);
			s*=getPathStep(drift,diffusion,localState,dJ_sum);
			*output=s;	//paths[i]=s;
			//printf("%f--%f ",s*option->spot,lambda);
		}
	}
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

//MC第二步
//定价函数 （调用上一个函数reduce_sum）
//计算每条路径payoff并求其期望，以定价
template<typename Real>
__global__ void computeValue(Real *const values,
					         const Real *const paths,
							 const MertonDJAsianOption<Real> *const option,
							 const unsigned int numSims,
							 const unsigned int numTimesteps)
{
	//声明在同一个cg中
	cg::thread_block cta=cg::this_thread_block();

	unsigned int bid=blockIdx.x; //块索引
	unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x; //thread索引
	unsigned int step=gridDim.x*blockDim.x; //thread数目

	Real sumPayoffs=static_cast<Real>(0); //sumPayoffs = 0
	//根据资产价格路径 paths 内的值求payoff
	for(unsigned int i=tid;i<numSims;i+=step)
	{
		const Real *path=paths+i; //paths[i]
		//该期权为算术平均 固定价格 亚式期权
		Real avg=static_cast<Real>(0); //avg = 0
		for(unsigned int t=0;t<numTimesteps;t++,path+=numSims)
		{
			avg+=*path;
		}
		avg=avg * option->spot/numTimesteps; //注意，这里乘了一个S0，是因为之前生成路径时起始值为1，而非S0

		Real payoff=avg-option->strike; //avg - K ：固定定价（浮动定价是 St - avg）

		//如果是看跌期权（Put）payoff逆转 （call——avg-K; put——K-avg)
		if(!(option->is_call))
		{
			payoff=-payoff;
		}

		//消除payoff的负值，期权payoff必须为非负数
		payoff=max(static_cast<Real>(0),payoff); // payoff=max(0,payoff)
		sumPayoffs+=payoff; //因为最终要求payoff的均值，因此先加和
	}

	//块内的所有payoff求个均值
	sumPayoffs=reduce_sum<Real>(sumPayoffs,cta);

	//printf("OK");
	//将所有块内均值结果存在values中，等待再次归约（在主机端）
	if(threadIdx.x==0)
	{
		values[bid]=sumPayoffs;
		//printf("%f  ",sumPayoffs);
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
void PricingEngine<Real>::operator()(Option<Real> &option)
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
	MertonDJAsianOption<Real> **d_option=(MertonDJAsianOption<Real>**)calloc(nDev,sizeof(MertonDJAsianOption<Real>*));
	Real **d_paths=(Real**)calloc(nDev,sizeof(Real*));

	curandStatePhilox4_32_10_t **d_rngStates=(curandStatePhilox4_32_10_t**)calloc(nDev,sizeof(curandStatePhilox4_32_10_t*));
	//curandState **d_rngStates=(curandState**)calloc(nDev,sizeof(curandState*));

	Real **d_values=(Real**)calloc(nDev,sizeof(Real*));

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
		//std::cout<<block[d].x<<" "<<grid[d].x<<std::endl;

		/*
		//限制每个SM上的block个数
		unsigned int blocksPerSM=10; //每个SM上分配的block个数
		unsigned int numSMs=deviceProperties[d].multiProcessorCount ; //当前设备上的SM个数
		while(grid[d].x>2*blocksPerSM*numSMs)
		{
			grid[d].x>>=1;
		}
		*/

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
		cudaResult=cudaMalloc((void**)&d_option[d],sizeof(MertonDJAsianOption<Real>));
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not allocate memory on device for option data: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		cudaResult=cudaMemcpy(d_option[d],&option,sizeof(MertonDJAsianOption<Real>),cudaMemcpyHostToDevice);
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

	//2.计算payoff
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		computeValue<<<grid[d],block[d],block[d].x*sizeof(Real),stream[d]>>>(d_values[d],d_paths[d],d_option[d],numTasks[d],numTimesteps);
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
	for(int d=0;d<nDev;d++)
	{
		cudaSetDevice(d);
		vector<Real> values(grid[d].x);
		cudaResult=cudaMemcpy(&values[0],d_values[d],grid[d].x*sizeof(Real),cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not copy partial results to host: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		Real deviceMean=std::accumulate(values.begin(),values.end(),static_cast<Real>(0));
		deviceMean/=numTasks[d];
		option.value+=deviceMean;
	}
	
	//4.将价格折现
	option.value/=nDev;
	option.value*=exp(-option.r*option.tenor); //price=price*exp(-rT)

	if(m_timerON==1)
	{
		sdkStopTimer(&timer);
		std::cout<<"MC-传输计算结果耗时："<<sdkGetAverageTimerValue(&timer)/1000.0f<<std::endl;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}

	// ****************************************************
	// 三、收尾工作（释放各种内存）
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
	}

	free(d_option);
	free(d_paths);
	free(d_rngStates);
	free(d_values);

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
		
	return;
}
template class PricingEngine<float>;
template class PricingEngine<double>;


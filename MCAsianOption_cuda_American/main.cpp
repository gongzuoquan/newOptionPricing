#include<iostream>
#include<fstream>
#include"AsianOption.h"
#include"PricingEngine.h"
#include"Param.h"
#include"helper_timer.h"
using namespace std;
#define Real double
string config_file="configure";  //配置文件的名称
struct Param<Real> params;  //参数存储变量
int main(int argc,char **argv)
{	
	//一、参数加载
	int result=params.load_config(config_file);
	if(!result)
	{
		exit(1);
	}

	//二、设备初始化（可多个设备）
	//注：提前在这里查询一次设备属性，可以起到warmup的作用，在pricer()里面再次查询时速度提高8倍左右
	int nDev=0;
	struct cudaDeviceProp deviceProperties;
	cudaGetDeviceCount(&nDev);
	cudaError_t cudaResult=cudaSuccess;
	for(int d=0;d<nDev;d++)
	{
		cudaResult=cudaGetDeviceProperties(&deviceProperties,d);
		if(cudaResult!=cudaSuccess)
		{
			string msg("Could not get device properties: ");
			msg+=cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}
	}

	//三、命令行参数处理
	if(argc>=2)
	{
		params.num_sim=atoi(argv[1]);
		if(argc>=3)
		{
			params.num_timestep=atoi(argv[2]);
		}
	}

	//四、期权实例化
	AsianOption<Real> option;
	option.spot=static_cast<Real>(params.S0);
	option.strike=static_cast<Real>(params.K);
	option.r=static_cast<Real>(params.R);
	option.sigma=static_cast<Real>(params.V);
	option.tenor=static_cast<Real>(params.T);
	option.dt=static_cast<Real>(params.T/params.num_timestep);
	option.is_call=params.is_call;
	option.price=static_cast<Real>(0.0);
	option.golden=static_cast<Real>(params.golden);
	
	//五、定价模块（计时）
	PricingEngine<Real> pricer(params.num_sim,params.device,params.num_device,params.block_size,params.seed,params.timer_on);

	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	//pricer(option);

	sdkStartTimer(&timer);
	pricer(option); //
	sdkStopTimer(&timer);

	float elapsedTime=sdkGetAverageTimerValue(&timer)/1000.0f;

	//六、结果输出
	cout<<endl;
	cout<<"numSim,numTimeStep: "<<params.num_sim<<" , "<<params.num_timestep<<endl;
	cout<<"MonteCarlo result: "<<option.price<<endl;
	cout<<"    golden result: "<<option.golden<<endl;
	cout<<"MonteCarlo error : "<<option.error<<endl;
	cout<<"95%置信区间为 [ "<< option.price-1.96*option.error<<" ; "<<option.price+1.96*option.error<<" ]"<<endl;

	cout<<"        run time : "<<elapsedTime<<endl;
	sdkDeleteTimer(&timer);
	timer=NULL;

	return 0;
}




#include<iostream>
#include<fstream>
#include<ctime>
//#include"AsianOption.h"
#include"MertonDJAsianOption.h"
#include"PricingEngine.h"
#include"Param.h"
#include"helper_timer.h"
using namespace std;
#define Real float
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
	MertonDJAsianOption<Real> option;
	option.jump_mu=static_cast<Real>(params.jump_mu);
	option.jump_sigma=static_cast<Real>(params.jump_sigma);
	option.lambda=static_cast<Real>(params.lambda);
	/*
	if(params.is_MertonDJ)
	{
		MertonDJAsianOption<Real> option;
		option.jump_mu=static_cast<Real>(params.jump_mu);
		option.jump_sigma=static_cast<Real>(params.jump_sigma);
		option.lambda=static_cast<Real>(params.lambda);
	}
	else
	{
		AsianOption<Real> option;
	}
	*/
	option.spot=static_cast<Real>(params.S0);
	option.strike=static_cast<Real>(params.K);
	option.r=static_cast<Real>(params.R);
	option.sigma=static_cast<Real>(params.V);
	option.tenor=static_cast<Real>(params.T);
	option.dt=static_cast<Real>(params.T/params.num_timestep);
	option.is_call=params.is_call;
	option.value=static_cast<Real>(0.0);
	option.golden=static_cast<Real>(params.golden);

	//cout<<option.jump_mu<<endl;
	//cout<<option.jump_sigma<<endl;
	//cout<<option.lambda<<endl;
	//cout<<option.spot<<endl;
	//cout<<option.strike<<endl;
	//cout<<option.r<<endl;
	//cout<<option.sigma<<endl;
	//cout<<option.tenor<<endl;

	//五、定价模块（计时）
	if(params.seed==0)
	{
		params.seed=time(NULL);
	}
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
	cout<<"MonteCarlo result: "<<option.value<<endl;
	cout<<"    golden result: "<<option.golden<<endl;
	cout<<"        run time : "<<elapsedTime<<endl;
	sdkDeleteTimer(&timer);
	timer=NULL;

	return 0;
}




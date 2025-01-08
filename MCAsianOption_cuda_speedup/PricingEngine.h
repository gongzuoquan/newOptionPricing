#ifndef PRICINGENGINE_H
#define PRICINGENGINE_H

#include"AsianOption.h"
#include<cuda_runtime.h>

template<typename Real>
class PricingEngine
{
	public:
		// 无参数构造函数
		PricingEngine(unsigned int numSims,unsigned int device,int numDev,unsigned int threadBlockSize,unsigned int seed,int timerON);

		//实际执行函数，()符号重载
		void operator()(AsianOption<Real> &option);

	private:
		unsigned int m_seed; //随机数种子
		unsigned int m_numSims; //模拟数目
		unsigned int m_device; //设备号，可有可无
		int m_numDev; //设备号，可有可无
		unsigned int m_threadBlockSize; //线程块大小
		int m_timerON; //是否开启计时器进行测试
};

#endif


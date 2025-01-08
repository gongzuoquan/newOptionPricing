#ifndef TESTRNG_H
#define TESTRNG_H

#include<cuda_runtime.h>

template<typename Real>
class testRNG
{
	public:
		testRNG(unsigned int numSims,unsigned int numTimeSteps,unsigned int threadBlockSize,unsigned int seed);

		//实际执行函数，()符号重载
		void operator()();

	private:
		unsigned int m_seed; //随机数种子
		unsigned int m_numSims; //模拟数目
		unsigned int m_numTimeSteps; //模拟数目
		unsigned int m_threadBlockSize; //线程块大小
};

#endif


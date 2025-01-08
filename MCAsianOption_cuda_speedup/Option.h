//2021.8.23
//设计一个基类，作为所有期权的父类
#ifndef OPTION_H
#define OPTION_H

template <typename Real>
class Option
{
	public:
		//期权参数
		Real spot; //标的资产当前价格，现价
		Real strike; //期权定价，K
		Real r; //无风险利率
		Real sigma; //波动率，volatility
		Real tenor; //期权成熟期，T
		Real dt; //采样区间，delta T

		//期权价格
		Real golden; //真实定价
		Real price; //MC定价结果
		Real error; //MC误差

		//期权种类：看涨 或者 看跌
		bool is_call;
	public:
		Option()
		{
			spot=0.0; //标的资产当前价格，现价
			strike=0.0; //期权定价，K
			r=0.0; //无风险利率
			sigma=0.0; //波动率，volatility
			tenor=1.0; //期权成熟期，T
			dt=0.0; //采样区间，delta T

			//期权价格
			golden=0.0; //真实定价
			price=0.0; //MC定价结果
			error=0.0; //MC误差

			//期权种类：看涨 或者 看跌
			is_call=true;
		}

		Option(Real in_spot,
				    Real in_strike,
					Real in_r,
					Real in_sigma,
					Real in_tenor,
					Real in_dt,
					Real in_golden,
					bool in_is_call
					)
			        :spot(in_spot),
					strike(in_strike),
					r(in_r),
					sigma(in_sigma),
					tenor(in_tenor),
					dt(in_dt),
					golden(in_golden),
					is_call(in_is_call)
		{
			price=0.0;
		}

		~Option(){}
};

#endif

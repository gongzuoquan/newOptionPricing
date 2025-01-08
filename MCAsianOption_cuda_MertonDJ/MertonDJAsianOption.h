#ifndef MERTONDJASIANOPTION_H
#define MERTONDJASIANOPTION_H
#include"Option.h"

//固定定价期权
template <typename Real>
class MertonDJAsianOption : public Option<Real>
{
	public:
		Real jump_mu;
		Real jump_sigma;
		Real lambda;
	public:
		MertonDJAsianOption():Option<Real>(){}

		~MertonDJAsianOption(){}
};

#endif

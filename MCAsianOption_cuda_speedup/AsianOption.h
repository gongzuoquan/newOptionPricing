#ifndef ASIANOPTION_H
#define ASIANOPTION_H
#include"Option.h"

//固定定价期权
template <typename Real>
class AsianOption : public Option<Real>
{
	public:
		AsianOption():Option<Real>(){}

		~AsianOption(){}
};

#endif

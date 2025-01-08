// 2021.9.1
// 参数类
#ifndef PARAM_H
#define PARAM_H
#include<iostream>
#include<boost/algorithm/string.hpp>
#include<cstdlib>
#include<string>
#include<vector>
using namespace std;
using namespace boost;
template<typename Real>
struct Param
{
	public:
		int device;
		int num_device;
		int block_size;
		int seed;

		int num_sim;
		int num_timestep;

		Real S0;
		Real K;
		Real R;
		Real V;
		Real T;
		int is_call;

		int is_MertonDJ;
		Real jump_mu;
		Real jump_sigma;
		Real lambda;

		Real golden;
		int timer_on;
	public:
		void get(string var,string value)
		{
			if(!var.compare("device"))
				device=atoi(value.c_str());
			if(!var.compare("num_device"))
				num_device=atoi(value.c_str());
			if(!var.compare("block_size"))
				block_size=atoi(value.c_str());
			if(!var.compare("seed"))
				seed=atoi(value.c_str());

			if(!var.compare("num_sim"))
				num_sim=atoi(value.c_str());
			if(!var.compare("num_timestep"))
				num_timestep=atoi(value.c_str());

			if(!var.compare("S0"))
				S0=atof(value.c_str());
			if(!var.compare("K"))
				K=atof(value.c_str());
			if(!var.compare("R"))
				R=atof(value.c_str());
			if(!var.compare("V"))
				V=atof(value.c_str());
			if(!var.compare("T"))
				T=atof(value.c_str());
			if(!var.compare("is_call"))
				is_call=atoi(value.c_str());

			if(!var.compare("is_MertonDJ"))
				is_MertonDJ=atoi(value.c_str());
			if(!var.compare("jump_mu"))
				jump_mu=atof(value.c_str());
			if(!var.compare("jump_sigma"))
				jump_sigma=atof(value.c_str());
			if(!var.compare("lambda"))
				lambda=atof(value.c_str());

			if(!var.compare("golden"))
				golden=atof(value.c_str());
			if(!var.compare("timer_on"))
				timer_on=atoi(value.c_str());
			return ;
		}
		int load_config(string fcon)
		{
			ifstream fin(fcon.c_str());
			if(!fin.is_open())
			{
				cerr<<"Can't open configure file : "<<fcon<<"."<<endl;
				return 0;
			}
			string buffer;
			vector<string> split_results;
			while(getline(fin,buffer))
			{
				//cout<<buffer<<endl;;
				//“#”开头的行为注释行
				if(buffer[0]=='#'||!buffer.compare(""))
					continue;
				split(split_results,buffer,is_any_of("="));

				/*
				for(size_t i=0;i<split_results.size();i++)
				{
					cout<<split_results[i]<<" ";
				}
				cout<<endl;
				cout<<split_results.size()<<endl;
				*/

				if(split_results.size()!=2)
				{
					cerr<<"Configure file error."<<endl;
					return 0;
				}

				get(split_results[0],split_results[1]);
				//cout<<split_results[0]<<" "<<split_results[1]<<endl;
			}
			return 1;
		}
};
#endif


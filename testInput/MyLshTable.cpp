#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <limits.h>
#include <map>
#include <random>
#include <flann/flann.hpp>
class MylshTable{
public:
	MylshTable(){
	}

	MylshTable(double w, int d, int kkk, flann::Matrix<double> features):w(w),d(d),kkk(kkk){
		bParams = GenUniform(kkk, w);
		r1 = GenUniform(kkk, Prime);
		r2 = GenUniform(kkk, Prime);
		for (int i = 0; i < kkk; i++){
			aParams.push_back(GenNormal(d));
		}
		for (int i = 0; i < features.rows; i++){
			add(features[i], i);
		}
	}
	std::vector<int> GetNeibours(const double * vec)const{
		auto ans = h(vec);
		std::vector<int> out;
		auto i = Table[ans.first].begin();
		for (; i != Table[ans.first].end(); i++){
			if (ans.second == i->first){
				out.push_back(i->second);
			}
		}
		return out;
	}
	//向哈希表中插入一个点
	void add(const double* vec, const int index){
		auto ans = h(vec);
		Table[ans.first].push_back(std::make_pair(ans.second, index));
	}
	std::pair<int, int> h(const double* vec)const{
		std::vector<int> hash1;
		for (int i = 0; i < kkk; i++){
			double sum = 0;
			for (int d = 0; d < d; d++){
				sum += vec[d] * aParams[i][d];
			}
			int ans = (int)std::ceil((sum + bParams[i]) / w);
			hash1.push_back(ans);
		}
		std::pair<int, int> ans = { 0, 0 };
		for (int i = 0; i < kkk; i++){
			ans.first = (ans.first + (hash1[i] % Prime) * (long long)(r1[i] % Prime)) % Prime;
			ans.second = (ans.second + (hash1[i] % Prime) * (long long)(r2[i] % Prime)) % Prime;
		}
		ans.first %= TableSize;
		return ans;
	}
	//利用normal_distribution生成满足标准正态分布的随机向量
	std::vector<double> GenNormal(int n){
		std::vector<double> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> d(0, 1);
		for (int i = 0; i < n; i++){
			ans.push_back(d(gen));
		}
		return ans;
	}
	//生成均匀分布的随机小数b
	std::vector<double> GenUniform(int n, double w){
		std::vector<double> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(0, w);

		for (int i = 0; i < n; i++){
			ans.push_back(dis(gen));
		}
		return ans;
	}
	std::vector<int> GenUniform(int n, int w){
		std::vector<int> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, w);

		for (int i = 0; i < n; i++){
			ans.push_back(dis(gen));
		}
		return ans;
	}

	const int Prime = 1e9 + 7;

	const static int TableSize = 100005;

	int d; //输入向量的维数

	int kkk; //哈希函数的个数

	std::vector<std::vector<double> > aParams; //哈希函数的随机参数a

	std::vector<double> bParams;			//哈希函数的随机参数b

	std::vector<int> r1;

	std::vector<int> r2;

	std::vector<std::pair<int, int> > Table[TableSize];	//哈希表

	double w; //用于分段的长度 
};
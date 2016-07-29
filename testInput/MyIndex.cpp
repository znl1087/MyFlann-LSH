#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <limits.h>
#include <map>
#include <random>
#include <math.h>
#include <stddef.h>
#include <flann/flann.hpp>
#include "MyLshTable.cpp"

namespace MyFlann{
	class MyIndex{
	public:
		MyIndex(){}
		MyIndex(const flann::Matrix<double>& features){
			setDataset(features);
			table = new MylshTable(1.0, features.rows, 10, features);
		}
		~MyIndex(){
			delete table;
		}
		void setDataset(const flann::Matrix<double>& dataset)
		{
			size_ = dataset.rows;
			veclen_ = dataset.cols;
			last_id_ = 0;

			ids_.clear();

			points_.resize(size_);
			for (size_t i = 0; i < size_; ++i) {
				points_[i] = dataset[i];
			}
		}

		virtual int knnSearch(const flann::Matrix<double>& queries,
			flann::Matrix<size_t>& indices,
			flann::Matrix<double>& dists,
			size_t knn) const
		{
			int count = 0;

			flann::KNNSimpleResultSet<double> resultSet(knn);
			for (int i = 0; i < (int)queries.rows; i++) {
				resultSet.clear();
				findNeighbors(resultSet, queries[i]);
				size_t n = std::min(resultSet.size(), knn);
				resultSet.copy(indices[i], dists[i], n);
				count += n;
			}
			return count;
		}
		void findNeighbors(flann::ResultSet<double>& result, const double* vec)const{
			auto ans = table->GetNeibours(vec);
			flann::L2<double> distance;
			for (auto i = ans.begin(); i != ans.end(); i++){
				double dis = distance(vec, points_[*i], veclen_);
				result.addPoint(dis, *i);
			}
		}
	protected:
		/**
		* Each index point has an associated ID. IDs are assigned sequentially in
		* increasing order. This indicates the ID assigned to the last point added to the
		* index.
		*/
		size_t last_id_;
		/**
		* Number of points in the index (and database)
		*/
		size_t size_;


		/**
		* Size of one point in the index (and database)
		*/
		size_t veclen_;

		/**
		* Array of point IDs, returned by nearest-neighbour operations
		*/
		std::vector<size_t> ids_;

		/**
		* Point data
		*/
		std::vector<double*> points_;

		MylshTable *table;
	};
}
// testInput.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <flann/flann.hpp>
#include <fstream>
#include "MyIndex.cpp"

int _tmain(int argc, char** argv)
{
	std::vector<double> pointInput;
	std::ifstream inFile("Iris90.txt");
	//std::ofstream outFile("outLSH.txt");
	std::ofstream outFile("out.txt");
	std::vector<double> data;
	while (!inFile.eof()){
		double temp;
		inFile >> temp;
		data.push_back(temp);
	}
	int nn = 3;
	/*flann::Matrix<double> points = flann::Matrix<double>(&data[0], 90, 5);
	flann::Matrix<double> query = flann::Matrix<double>(&data[0], 90, 5);

	flann::Matrix<double> dists(new double[query.rows*nn], query.rows, nn);
	flann::Matrix<size_t> indices(new size_t[query.rows*nn], query.rows, nn);

	MyFlann::MyIndex index = MyFlann::MyIndex(points);
	index.knnSearch(query, indices, dists, nn);*/

	flann::Matrix<double> points = flann::Matrix<double>(&data[0], 90, 5);
	flann::Matrix<double> query = flann::Matrix<double>(&data[0], 90, 5);

	flann::Matrix<double> dists(new double[query.rows*nn], query.rows, nn);
	flann::Matrix<size_t> indices(new size_t[query.rows*nn], query.rows, nn);

	flann::Index<flann::L2<double> > index(points, flann::KDTreeIndexParams(4));
	index.buildIndex();
	index.knnSearch(query, indices, dists, nn, flann::SearchParams(128)); 
	
	for (int i = 0; i < 90; i++){
		outFile << indices[i][0] << " " <<dists[i][0] << " " << indices[i][1] << " " << dists[i][1]<< " ";
		outFile << indices[i][2] << " " << dists[i][2] << std::endl;
	}
	outFile.close();
	inFile.close();
	//delete[] points.ptr();
	//delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr(); 
	return 0;
}


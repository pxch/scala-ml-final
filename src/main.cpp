/*
 * main.cpp
 *
 *  Created on: Nov 16, 2014
 *      Author: xun
 */

#include <iostream>
#include <vector>
#include "Lanczos.h"
#include <omp.h>
using namespace std;

int main(int argc, char*argv[]) {
	//srand((long) time(NULL));
	omp_set_num_threads(atoi(argv[1]));
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	int n = 20000;
	int k = 100;
	double sparsity = 0.05;
	tripletList.reserve(n * n * sparsity);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if ((double) rand() / RAND_MAX < sparsity) {
				double v = (double) rand() / RAND_MAX;
				tripletList.push_back(T(i, j, v));
				tripletList.push_back(T(j, i, v));
			}
		}
		tripletList.push_back(T(i, i, (double) rand() / RAND_MAX));
	}
	Eigen::SparseMatrix<double> A(n, n);
	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, k);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	for (int i = 0; i < k; i++) {
		Q(i, i) = 1;
	}
	cout << "Matrix created." << endl;
	double wtime = omp_get_wtime();
	/*
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
	cout << "Eigens: WTime = " << omp_get_wtime() - wtime << endl;
	Eigen::VectorXd values = solver.eigenvalues();
	Mat vectors = solver.eigenvectors();
	int front = 0, back = values.rows() - 1;
	for (int i = 0, idx; i < k; i++) {
		if (abs((double)values(front)) > abs((double)values(back)))
			idx = front++;
		else
			idx = back--;
		cout << values(idx) << endl;
	}
	cout << "=-=-=-=-=-=-=-=" << endl;*/
	wtime = omp_get_wtime();
	Lanczos lanczos(A, k, Q);
	cout << "Lanczos: WTime = " << omp_get_wtime() - wtime << endl;
	cout << lanczos.eigenValues() << endl << endl;
	return 0;
}


/*
 * Lanczos.cpp
 *
 *  Created on: Nov 16, 2014
 *      Author: xun
 */

#include "Lanczos.h"
#include "QR"
#include <iostream>
#include <omp.h>

using namespace std;

static void formT(Mat& T, Mat& B, Mat& Aj) {
	if (T.cols() == 0) {
		T = Aj;
	} else {
		int k = Aj.cols();
		Mat temp = T;
		T = Mat::Zero(temp.rows() + k, temp.cols() + k);
		T.block(0, 0, temp.rows(), temp.cols()) = temp;
		T.block(temp.rows(), temp.cols(), k, k) = Aj;
		T.block(temp.rows(), temp.cols() - k, k, k) = B;
		T.block(temp.rows() - k, temp.cols(), k, k) = B.transpose();
	}
}

static void formQ(Mat& Q, Mat& Qj) {
	Mat temp = Q;
	Q = Mat::Zero(Q.rows(), temp.cols() + Qj.cols());
	Q.block(0, 0, Q.rows(), temp.cols()) = temp;
	Q.block(0, temp.cols(), Q.rows(), Qj.cols()) = Qj;
}

void Lanczos::computeEigenPairs(Mat& T, Mat& Q, int k) {
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
	Eigen::VectorXd values = solver.eigenvalues();
	Mat vectors = solver.eigenvectors();
	int front = 0, back = values.rows() - 1;
	for (int i = 0, idx; i < k; i++) {
		if (abs((double) values(front)) > abs((double) values(back)))
			idx = front++;
		else
			idx = back--;
		_eigenValues(i) = values(idx);
		_eigenVectors.col(i) = Q * vectors.col(idx);
	}
}

Lanczos::Lanczos(SpMat& A, int k, Mat& V0) {
	_eigenValues = Mat::Zero(k, 1);
	_eigenVectors = Mat::Zero((int) A.cols(), k);
	_theta = sqrt(0.0001 * A.rows());
	Mat B = Mat::Zero(k, k);
	Mat Q0 = Mat::Zero((int) A.cols(), k);
	Mat Q1 = V0;
	Mat Q = Q1;
	Mat T;
	int chunkSize = A.outerSize() * k / omp_get_max_threads();
	cout << chunkSize <<endl;
	double total = 0;
	int j = 0;
	for (; j < max_iter; j++) {
		double w2, w1, s;
		s = omp_get_wtime();
		Mat R = Mat::Zero(A.rows(), k);
		w1 = omp_get_wtime();
#pragma omp parallel
		{
#pragma omp for schedule(static) collapse(2)
			for(int r = 0; r < k; r++) {
				for (int c = 0; c < A.outerSize(); c++) {
					double& tar = R.coeffRef(c, r);
					for (SpMat::InnerIterator it(A, c); it; ++it) {
						tar += it.value() * Q1.coeff(it.row(), r);
					}
				}
			}
		}
		w2 = omp_get_wtime();
		R = R - Q0 * B.transpose();
		Mat Aj = Q1.transpose() * R;
		R = R - Q1 * Aj;
		Q0 = Q1;
		Eigen::HouseholderQR<Mat> qr(R);
		Q1 = qr.householderQ() * Mat::Identity(A.rows(), k);
		formT(T, B, Aj);
		computeEigenPairs(T, Q, k);
		if (validPairs(A))
			break;
		formQ(Q, Q1);
		B = Q1.transpose() * R;
		total+=w2-w1;
		cout << j << ": " << w2 - w1 << "/" << omp_get_wtime() - s << endl;
	}
	cout << total / j << endl;
}

bool Lanczos::validPairs(SpMat& A) {
	Eigen::VectorXd y = Eigen::VectorXd::Zero(_eigenValues.rows());
	for (int i = 0; i < _eigenValues.rows(); i++) {
		y = A * _eigenVectors.col(i) - _eigenValues(i) * _eigenVectors.col(i);
		if (sqrt(y.transpose() * y) > _theta)
			return false;
	}
	return true;
}


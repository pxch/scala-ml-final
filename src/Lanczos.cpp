#include "Lanczos.h"
#include "QR"
#include <iostream>
#include <omp.h>
#include <fstream>
//#include <mkl_lapacke.h>
#include <stdlib.h>

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

//    Mat tmp0 = Mat::Zero(T.rows(), k);

	int front = 0, back = values.rows() - 1;
	for (int i = 0, idx; i < k; i++) {
		if (fabs((double) values(front)) > fabs((double) values(back)))
			idx = front++;
		else
			idx = back--;
		_eigenValues(i) = values(idx);
		_eigenVectors.col(i) = Q * vectors.col(idx);
//        tmp0.col(i) = vectors.col(idx);
	}

/*
    Mat tmp1 = tmp0.transpose() * tmp0 - Mat::Identity(k, k);
    Mat tmp2 = _eigenVectors.transpose() * _eigenVectors - Mat::Identity(k, k);

    Mat tmp3 = Q.transpose() * Q - Mat::Identity(Q.cols(), Q.cols());

    cout << "Orthogonal test:\t" << tmp1.minCoeff() << "\t" << tmp1.maxCoeff() << ",\t";
    cout << tmp2.minCoeff() << "\t" << tmp2.maxCoeff() << ",\t";
    cout << tmp3.minCoeff() << "\t" << tmp3.maxCoeff() << endl;
*/
}

Lanczos::Lanczos(SpMat& A, int k, Mat& V0, bool flag) {
	_eigenValues = Mat::Zero(k, 1);
	_eigenVectors = Mat::Zero((int) A.cols(), k);
	_theta = sqrt(0.0001 * A.rows());

	Mat B = Mat::Zero(k, k);
	Mat Q0 = Mat::Zero((int) A.cols(), k);
	Mat Q1 = V0;
	Mat Q = Q1;
	Mat T;

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

        if (j == 0) {
            Mat tmp11 = Q.transpose() * R;
            R = R - Q * tmp11;
        }

		Q0 = Q1;
		Eigen::HouseholderQR<Mat> qr(R);

		Q1 = qr.householderQ() * Mat::Identity(A.rows(), k);
/*
        if (j == 0 && flag == true) {
            ofstream out;
            out.open("R.txt");
            out << R;
            out.close();
            out.open("Aj.txt");
            out << Aj;
            out.close();
            out.open("Q1.txt");
            out << Q1;
            out.close();
            out.open("Q0.txt");
            out << Q0;
            out.close();
        }
*/
/*
        Mat tmp0 = Q1.transpose() * Q1 - Mat::Identity(k, k);
        Mat tmp1 = Q0.transpose() * Q1;
        Mat tmp2 = Q0.transpose() * R;
        cout << "QR Fact test:\t" << tmp0.minCoeff() << "\t" << tmp0.maxCoeff() << ",\t";
        cout << tmp1.minCoeff() << "\t" << tmp1.maxCoeff() << ",\t";
        cout << tmp2.minCoeff() << "\t" << tmp2.maxCoeff() << endl;
*/

		formT(T, B, Aj);
//        if ((j + 1) % 5 == 0) {
            double tt = omp_get_wtime();
            computeEigenPairs(T, Q, k);
            bool valid = validPairs(A);
            cout << "EigenSolver time = " << omp_get_wtime() - tt << endl;
            if (valid) {
                break;
            }
//        }

		formQ(Q, Q1);
		B = Q1.transpose() * R;
/*
        vector<int> piv_col_idx;
        if (j == 0) {
            cout << "Diagonal of B:" << endl;
            for (int i = 0; i < k; ++i) {
                cout << i << ":\t" << B(i,i) << endl;
                if (fabs(B(i,i)) < 1e-10) {
                    piv_col_idx.push_back(i);
                }
            }
            for (int i = 0; i < piv_col_idx.size(); ++i) {
                cout << "Pivoting test of column " << piv_col_idx[i] << endl;
                cout << Q0.transpose() * Q1.col(piv_col_idx[i]) << endl;
                cout << "---------" << endl;
            }

            ofstream out;
            out.open("B.txt");
            out << B;
            out.close();

        }
*/

		total+=w2-w1;
		cout << j << ": " << w2 - w1 << "/" << omp_get_wtime() - s << endl;
	}
//	cout << total / j << endl;
}

bool Lanczos::validPairs(SpMat& A) {
	Eigen::VectorXd y = Eigen::VectorXd::Zero(A.rows());
	for (int i = 0; i < _eigenValues.rows(); i++) {
//		y = A * _eigenVectors.col(i) - _eigenValues(i) * _eigenVectors.col(i);
//		y = A * _eigenVectors.col(i) / _eigenValues(i) - _eigenVectors.col(i);

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int c = 0; c < A.outerSize(); c++) {
                y.coeffRef(c) = 0;
                for (SpMat::InnerIterator it(A, c); it; ++it) {
                    y.coeffRef(c) += it.value() * _eigenVectors.col(i).coeffRef(it.row());
                }
            }
        }

        y = y / _eigenValues(i) - _eigenVectors.col(i);

        if (sqrt(y.transpose() * y) > 0.1)
			return false;
	}
	return true;
}


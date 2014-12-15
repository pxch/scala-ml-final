#ifndef MSEIGS_H_
#define MSEIGS_H_

#include "GraphClustering.h"
#include "Lanczos.h"

#include <algorithm>
#include <tuple>
#include <Eigenvalues>

#define NVTXS_THRES 400
#define C 16

typedef std::tuple<idx_t, idx_t, double> eigval_triple;
bool eig_comparator(const eigval_triple& l, const eigval_triple& r) {
    return std::get<2>(l) > std::get<2>(r);
}

namespace mseigs{
    void load_data (const std::string filename, smat_t& mat) {
        std::cout << "... Loading file - " << filename << std::endl;

        double start_time = omp_get_wtime();

        std::ifstream in(filename.c_str());
        if (!in.is_open()) {
            std::cerr << "Error in opening file " << filename << std::endl;
            exit(-1);
        }

        idx_t row_cnt = 0, col_cnt = 0, nnz_cnt = 0;

        std::string line;
        size_t pos;
        idx_t row_idx, col_idx;
        double val;

        std::vector<triplet_t> tri_list;

        while(std::getline(in, line)) {
            std::stringstream(line) >> row_idx >> col_idx;
            if (row_idx > row_cnt) {
                row_cnt = row_idx;
            }

            tri_list.push_back(triplet_t(row_idx-1, col_idx-1, 1));
            ++nnz_cnt;
        }
        col_cnt = col_idx;

        std::cout << "\trows:\t" << row_cnt << std::endl;
        std::cout << "\tcols:\t" << col_cnt << std::endl;
        std::cout << "\tnnzs:\t" << nnz_cnt << std::endl;

        mat.resize(row_cnt, col_cnt);
        mat.reserve(nnz_cnt);

        mat.setFromTriplets(tri_list.begin(), tri_list.end());

        mat.makeCompressed();

        std::cout << "... Done. Elapsed time = ";
        std::cout << omp_get_wtime() - start_time << "s" << std::endl;
    }

    void eigenvalues(smat_t& A, idx_t k, dmat_t& omega, vec_t& lambda, idx_t lvl, bool flag0) {
        if (A.rows() != A.cols()) {
            std::cout << "Dimension mismatch!" << std::endl;
            exit(-1);
        }
        idx_t nvtxs = A.rows();
        
        omega.resize(nvtxs, k);
        omega.setZero();
        lambda.resize(k, 1);
        lambda.setZero();

        std::stringstream ss;
        ss << "Level " << lvl << " nvtxs = " << nvtxs;
        
        std::string dbg_lvl_info = ss.str();

        if (nvtxs <= NVTXS_THRES) {
//            std::cout << dbg_lvl_info << "\tSolve eigenvalue by standard EigenSolver" << std::endl;
            Eigen::SelfAdjointEigenSolver<dmat_t> es(A);

            vec_t eigvals = es.eigenvalues();
            idx_t start = 0, end = eigvals.size() - 1;
            double eigval;
            idx_t idx;
            for (idx_t i = 0; i < k; ++i) {
                if (fabs(eigvals(start)) >= fabs(eigvals(end))) {
                    eigval = fabs(eigvals(start));
                    idx = start++;
                } else {
                    eigval = fabs(eigvals(end));
                    idx = end--;
                }
                vec_t eigvec = es.eigenvectors().col(idx);
                eigvec *= (double)1.0 / eigvec.norm();
                omega.col(i) = eigvec;
//                omega.block(0, i, nvtxs, 1) = eigvec;
                lambda(i) = eigval;
            }
        } else {
            idx_t k1 = ceil((double)k / C) * 2;
            if (k == 1) {
                k1 = 1;
            }
//            std::cout << dbg_lvl_info << "\tPartition into " << C << " subgraphs" << std::endl;
            GraphClustering gc(A, C);

            veci_t vtx_offset = gc.getVtxOffset();
            veci_t vtx_cnt = gc.getVtxCnt();

            std::vector<dmat_t> omega_vec(C);
            std::vector<vec_t> lambda_vec(C);
            std::vector<eigval_triple> top_eigvals;

            if (lvl == 0) {
                #pragma omp parallel for schedule(dynamic)
                for (idx_t p = 0; p < C; ++p) {
                    std::cout << "DBG\t" << dbg_lvl_info << "\tsubgraph " << p << std::endl;
                    eigenvalues(gc.getSubGraph(p), k1, omega_vec[p], lambda_vec[p], lvl + 1, 1);
                }
            } else {
                for (idx_t p = 0; p < C; ++p) {
                    eigenvalues(gc.getSubGraph(p), k1, omega_vec[p], lambda_vec[p], lvl + 1, 0);
                }
            }

            for (idx_t p = 0; p < C; ++p) {
                idx_t c = 0, cnt = 0;
/*
                if (lvl == 0) {
                    std::cout << "------------" << std::endl;
                    for (idx_t i = 0; i < k1; ++i) {
                        for (idx_t j = i; j < k1; ++j) {
                            std::cout << omega_vec[p].col(i).dot(omega_vec[p].col(j)) << ", ";
                        }
                        std::cout << std::endl;
                    }
                }
*/

                veci_t ortho_set;
                for (idx_t c = 0; c < k1; ++c) {
                    bool flag = true;
                    for (idx_t cc = 0; cc < ortho_set.size(); ++cc) {
                        if (fabs(omega_vec[p].col(c).dot(omega_vec[p].col(ortho_set[cc]))) > 1e-10) {
                            flag = false;
                            break;
                        }
                    }
                    if (flag == true) {
                        top_eigvals.push_back(std::make_tuple<idx_t, idx_t, double>
                            (p, c, fabs(lambda_vec[p][c])));
                        ortho_set.push_back(c);
                    }
                }
            }
/*
            if (lvl == 0) {
                for (idx_t i = 0; i < C; ++i) {
                    std::cout << i << "\t" << vtx_offset[i] << "\t" << vtx_cnt[i] << std::endl;
                }
            }
*/
            std::sort(top_eigvals.begin(), top_eigvals.end(), eig_comparator);

            std::cout << "DBG\t" << dbg_lvl_info << "\t" << top_eigvals.size() << "\t" << k << std::endl;
/*
            if (lvl == 0) {
                for (idx_t i = 0; i < k; ++i) {
                    std::cout << i << "\t" << std::get<0>(top_eigvals[i]) << "\t";
                    std::cout << std::get<1>(top_eigvals[i]) << "\t";
                    std::cout << std::get<2>(top_eigvals[i]) << std::endl;
                }
            }
*/
            for (idx_t i = 0; i < k; ++i) {
                idx_t p_idx = std::get<0>(top_eigvals[i]);
                idx_t c_idx = std::get<1>(top_eigvals[i]);
/*
                if (lvl == 0) {
                    std::cout << i << "\t" << p_idx << "\t" << c_idx << std::endl;
                }
*/
                omega.block(vtx_offset[p_idx], i, vtx_cnt[p_idx], 1) =
                    omega_vec[p_idx].col(c_idx);
            }
/*
            if (lvl == 0) {
                std::cout << "--------" << std::endl;
                for (idx_t i = 0; i < k; ++i) {
                    for (idx_t j = i; j < k; ++j) {
                        std::cout << omega.col(i).dot(omega.col(j)) << ", ";
                    }
                    std::cout << std::endl;
                }
            }
*/
            dmat_t tmp0 = omega.transpose() * omega - dmat_t::Identity(k, k);
            std::cout << dbg_lvl_info << "\t" << tmp0.minCoeff() << "\t" << tmp0.maxCoeff() << std::endl;

//            std::cout << "Level " << lvl << " nvtxs = " << nvtxs << std::endl;
            Lanczos lanczos(gc.getNewMat(), k, omega, flag0);
            lambda = lanczos.eigenValues();

            omega = gc.getTransMat() * lanczos.eigenVectors();
//            omega = gc.getTransMat().transpose() * lanczos.eigenVectors();
        }
    }

}

#endif //MSEIGS_H_


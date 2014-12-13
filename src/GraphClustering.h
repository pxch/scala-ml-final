#ifndef GRAPH_CLUSTERING_H_
#define GRAPH_CLUSTERING_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <sstream>
#include <string>
#include <omp.h>
#include <Sparse>
#include <Eigen>
#include <Eigenvalues>
#include "metis.h"

typedef long idx_t;

typedef std::tuple<idx_t, idx_t, double> eigval_triple;
bool comparator(const eigval_triple& l, const eigval_triple& r) {
    return std::get<2>(l) > std::get<2>(r);
}

typedef Eigen::SparseMatrix<double, 1, idx_t> smat_t;
typedef Eigen::MatrixXd dmat_t;
typedef Eigen::Triplet<double, idx_t> triplet_t;
typedef std::vector<triplet_t> tri_vec;
typedef Eigen::VectorXd vec_t;
typedef std::vector<double> vecd_t;
typedef std::vector<idx_t> veci_t;

class GraphClustering {
public:
    GraphClustering() {}
    ~GraphClustering() {}
 
    void load_data (const std::string filename) {
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

        while(std::getline(in, line)) {
            std::stringstream(line) >> row_idx >> col_idx;
/*
            sprintf(line, "%ld\t%ld", &row_idx, &col_idx);
    
            pos = line.find("\t");
            row_idx = atoi(line.substr(0, pos).c_str());
            col_idx = atoi(line.substr(pos+1, line.length()-1-pos).c_str());
*/
            if (row_idx > row_cnt) {
                row_cnt = row_idx;
            }

            raw_tri_list.push_back(triplet_t(row_idx-1, col_idx-1, 1));
            ++nnz_cnt;
        }
        col_cnt = col_idx;

        std::cout << "\trows:\t" << row_cnt << std::endl;
        std::cout << "\tcols:\t" << col_cnt << std::endl;
        std::cout << "\tnnzs:\t" << nnz_cnt << std::endl;

        this->nnz_cnt_before = nnz_cnt;
        this->nvtxs = row_cnt;

        raw_mat.resize(row_cnt, col_cnt);
        raw_mat.reserve(nnz_cnt);

        raw_mat.setFromTriplets(raw_tri_list.begin(), raw_tri_list.end());

        raw_mat.makeCompressed();

        std::cout << "... Done. Elapsed time = ";
        std::cout << omp_get_wtime() - start_time << "s" << std::endl;
    }

    void partition(idx_t np) {
        std::cout << "Partition into " << np << " clusters." << std::endl;
        this->nparts = np;

        idx_t* innerIndices = raw_mat.innerIndexPtr();
        idx_t* outerStarts = raw_mat.outerIndexPtr();

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
        options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
        options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
        options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
        options[METIS_OPTION_NCUTS] = 1;
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_NITER] = 10;
        options[METIS_OPTION_SEED] = 0;
        options[METIS_OPTION_MINCONN] = 0;
        options[METIS_OPTION_NO2HOP] = 0;
        options[METIS_OPTION_CONTIG] = 0;
        options[METIS_OPTION_UFACTOR] = 30;
        options[METIS_OPTION_DBGLVL] = 0;

        idx_t ncon = 1;

        idx_t objval;

        double start_time = omp_get_wtime();

        idx_t partIndices[nvtxs];

        int flag = METIS_PartGraphKway(&nvtxs, &ncon, outerStarts, innerIndices,
            NULL, NULL, NULL, &nparts, NULL, NULL, options, &objval, partIndices);

        std::cout << "Clustering done in " << omp_get_wtime() - start_time << "s" << std::endl;
        std::cout << "\tedge cut:\t" << objval << std::endl;

//        part.resize(nvtxs);
//        std::cout << part.size() << std::endl;
/*
        for (idx_t i = 0; i < nvtxs; ++i) {
            part[i] = partIndices[i];
        }
*/
        part.assign(partIndices, partIndices + nvtxs);
    }

    void calc_subgraph() {
        part_vtx_cnt.resize(nparts);
        for (idx_t i = 0; i < nparts; ++i) {
            part_vtx_cnt[i] = 0;
        }
        part_idx_map.resize(nvtxs);
        for (idx_t i = 0; i < nvtxs; ++i) {
            part_idx_map[i] = part_vtx_cnt[part[i]]++;
        }

        part_vtx_offset.resize(nparts);
        part_vtx_offset[0] = 0;
        for (idx_t i = 1; i < nparts; ++i) {
            part_vtx_offset[i] = part_vtx_offset[i-1] + part_vtx_cnt[i-1];
//            std::cout << "Offset of cluster " << i << ": " << part_vtx_offset[i] << std::endl;
        }

        new_idx_map.resize(nvtxs);
        for (idx_t i = 0; i < nvtxs; ++i) {
            idx_t idx = part_vtx_offset[part[i]] + part_idx_map[i];
//            new_idx_map[idx] = i;
            new_idx_map[i] = idx;
        }

        std::ofstream out1, out2, out3;
        out1.open("part_idx_map.txt");
        out2.open("new_idx_map.txt");
        out3.open("part_vec.txt");
        for (idx_t i = 0; i < nvtxs; ++i) {
            out1 << part_idx_map[i] << std::endl;
            out2 << new_idx_map[i] << std::endl;
            out3 << part[i] << std::endl;
        }

        part_tri_lists.resize(nparts);

        for (tri_vec::iterator tri = raw_tri_list.begin(); tri != raw_tri_list.end(); ++tri) {
            idx_t row = (*tri).row();
            idx_t col = (*tri).col();

            new_tri_list.push_back(triplet_t(new_idx_map[row], new_idx_map[col], 1));

            if (part[row] == part[col]) {
                idx_t p = part[row];
                part_tri_lists[p].push_back(triplet_t(part_idx_map[row], part_idx_map[col], 1));
            }
        }

        part_nnz_cnt.resize(nparts);
        nnz_cnt_after = 0;

        for (idx_t i = 0; i < nparts; ++i) {
            part_nnz_cnt[i] = part_tri_lists[i].size();
            nnz_cnt_after += part_nnz_cnt[i];
            std::cout << "Partition " << i+1 << ": nvtxs = " << part_vtx_cnt[i];
            std::cout << ", nnz = " << part_nnz_cnt[i] << std::endl;
        }

        std::cout << "Total number of edges after clustering:\t" << nnz_cnt_after << std::endl;

        new_mat.resize(nvtxs, nvtxs);
        new_mat.reserve(nnz_cnt_before);
        new_mat.setFromTriplets(new_tri_list.begin(), new_tri_list.end());

        part_mats.resize(nparts);
        for (idx_t i = 0; i < nparts; ++i) {
            part_mats[i].resize(part_vtx_cnt[i], part_vtx_cnt[i]);
            part_mats[i].reserve(part_nnz_cnt[i]);
            part_mats[i].setFromTriplets(part_tri_lists[i].begin(), part_tri_lists[i].end());
        }

        std::cout << "Done computing subgraphs ..." << std::endl;
    }

    void calc_omega(idx_t k) {
        std::vector<Eigen::SelfAdjointEigenSolver<dmat_t>> es_vec;
        std::vector<eigval_triple> top_eigvals;
        idx_t c = 2 * ceil((double)k / nparts);

        for (idx_t i = 0; i < nparts; ++i) {
            std::cout << "Eigen decomposition of partition " << i << std::endl;

            Eigen::SelfAdjointEigenSolver<dmat_t> es(part_mats[i]);
            es_vec.push_back(es);
            vec_t eigvals = es.eigenvalues();
            idx_t start = 0, end = eigvals.size() - 1;
            for (idx_t j = 0; j < c; ++j) {
                if (abs(eigvals(start)) >= abs(eigvals(end))) {
                    top_eigvals.push_back(std::make_tuple<idx_t, idx_t, double>(i, start, fabs(eigvals(start))));
                    ++start;
                } else {
                    top_eigvals.push_back(std::make_tuple<idx_t, idx_t, double>(i, end, fabs(eigvals(end))));
                    --end;
                }
            }
        }

        omega.resize(nvtxs, k);
        omega.setZero();

        std::sort(top_eigvals.begin(), top_eigvals.end(), comparator);

        for (idx_t i = 0; i < top_eigvals.size(); ++i) {
            std::cout << std::get<2>(top_eigvals[i]) << std::endl;
        }

        for (idx_t i = 0; i < k; ++i) {
            idx_t p_idx = std::get<0>(top_eigvals[i]);
            idx_t col_idx = std::get<1>(top_eigvals[i]);

            std::cout << p_idx << "\t" << col_idx << std::endl;

            vec_t eigvec = es_vec[p_idx].eigenvectors().col(col_idx);
            eigvec *= (double) 1.0 / eigvec.norm();
            omega.block(part_vtx_offset[p_idx], i, part_vtx_cnt[p_idx], 1) = eigvec;
        }

        std::cout << "Done computing Omega ..." << std::endl;
    }

    smat_t& get_new_mat() {
        return new_mat;
    }

    dmat_t& get_omega() {
        return omega;
    }

    idx_t get_nvtxs() {
        return nvtxs;
    }

    idx_t get_nnz() {
        return nnz_cnt_before;
    }

private:
    smat_t raw_mat;
    smat_t new_mat;
    std::vector<smat_t> part_mats;

    dmat_t omega;

    tri_vec raw_tri_list;
    tri_vec new_tri_list;
    std::vector<tri_vec> part_tri_lists;

    idx_t nnz_cnt_before, nnz_cnt_after;
    idx_t nvtxs, nparts;
    veci_t part;

    veci_t part_vtx_cnt, part_vtx_offset, part_nnz_cnt;
    veci_t part_idx_map, new_idx_map;
};


#endif //GRAPH_CLUSTERING_H_


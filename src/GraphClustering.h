#ifndef GRAPH_CLUSTERING_H_
#define GRAPH_CLUSTERING_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <omp.h>
#include <Sparse>
#include <Eigen>
#include <Eigenvalues>
#include "metis.h"

typedef int idx_t;

typedef Eigen::SparseMatrix<double> smat_t;
typedef Eigen::MatrixXd dmat_t;
typedef Eigen::Triplet<double, idx_t> triplet_t;
typedef std::vector<triplet_t> tri_vec;
typedef Eigen::VectorXd vec_t;
typedef std::vector<double> vecd_t;
typedef std::vector<idx_t> veci_t;

class GraphClustering {
public:
    GraphClustering(smat_t& A) : mat(A) {}
    GraphClustering(smat_t& A, idx_t nParts) : mat(A) {
        partition(nParts);
    }

    ~GraphClustering() {}

    void partition(idx_t np) {
        if (mat.rows() != mat.cols()) {
            std::cout << "Dimension mismatch!" << std::endl;
            exit(-1);
        }
        this->nvtxs = mat.rows();

        std::cout << "Partition into " << np << " clusters." << std::endl;
        this->nparts = np;

        idx_t* innerIndices = mat.innerIndexPtr();
        idx_t* outerStarts = mat.outerIndexPtr();

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

        part.assign(partIndices, partIndices + nvtxs);

        transformGraph();
    }

    smat_t& getSubGraph(idx_t i) {
        if (i < 0 || i >= nparts) {
            std::cerr << "Index out of range!" << std::endl;
            exit(-1);
        }
        return this->part_mats[i];
    }

    veci_t& getVtxOffset() {
        return this->vtx_offset;
    }

    veci_t& getVtxCnt() {
        return this->vtx_cnt;
    }

    smat_t& getNewMat() {
        return this->new_mat;
    }

    smat_t& getTransMat() {
        return this->trans_mat;
    }

private:
    void transformGraph() {
        vtx_cnt.resize(nparts);
        for (idx_t i = 0; i < nparts; ++i) {
            vtx_cnt[i] = 0;
        }
        part_idx_map.resize(nvtxs);
        for (idx_t i = 0; i < nvtxs; ++i) {
            part_idx_map[i] = vtx_cnt[part[i]]++;
        }

        vtx_offset.resize(nparts);
        vtx_offset[0] = 0;
        for (idx_t i = 1; i < nparts; ++i) {
            vtx_offset[i] = vtx_offset[i-1] + vtx_cnt[i-1];
        }

        idx_map.resize(nvtxs);
        for (idx_t i = 0; i < nvtxs; ++i) {
            idx_map[i] = vtx_offset[part[i]] + part_idx_map[i];
        }

        trans_mat.resize(nvtxs, nvtxs);
        for (idx_t i = 0; i < nvtxs; ++i) {
            trans_mat.insert(i, idx_map[i]) = 1;
        }
        new_mat = trans_mat.transpose() * mat * trans_mat;
        part_mats.resize(nparts);
        for (idx_t i = 0; i < nparts; ++i) {
            part_mats[i] = new_mat.block(vtx_offset[i], vtx_offset[i], vtx_cnt[i], vtx_cnt[i]);
        }
    }

    smat_t mat, new_mat, trans_mat;
    std::vector<smat_t> part_mats;

    idx_t nvtxs, nparts;

    veci_t part, vtx_cnt, vtx_offset;
    veci_t part_idx_map, idx_map;
};

#endif //GRAPH_CLUSTERING_H_


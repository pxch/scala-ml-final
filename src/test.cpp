#include "GraphClustering.h"
#include "Lanczos.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: ./clustering file_name nr_threads n_parts k" << endl;
        exit(-1);
    }

    string file_name = argv[1];
    int nr_threads = atoi(argv[2]);
    int n_parts = atoi(argv[3]);
    int k = atoi(argv[4]);

	omp_set_num_threads(nr_threads);

    GraphClustering clustering;
    clustering.load_data(file_name);
    clustering.partition(n_parts);
    clustering.calc_subgraph();
    clustering.calc_omega(k);

    idx_t nvtxs = clustering.get_nvtxs();
    idx_t nnz = clustering.get_nnz();


    smat_t A = clustering.get_new_mat();
/*
    for (int i = 0; i < 100; ++i) {
        int x = rand() % nvtxs;
        int y = rand() % nvtxs;
        cout << x << "\t" << y << "\t" << A.coeff(x,y) << "\t" << A.coeff(y,x) << endl;
    }
*/
/*
    nvtxs = 20000;
    Eigen::SparseMatrix<double> A = smat_t(nvtxs, nvtxs);
    tri_vec triplet_list;
    for (idx_t i = 0; i < nvtxs; ++i) {
        for (idx_t j = 0; j < i; ++j) {
            if ((double)rand() / RAND_MAX < 0.05) {
                triplet_list.push_back(triplet_t(i, j, 1));
                triplet_list.push_back(triplet_t(j, i, 1));
            }
        }
    }
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());
*/
/*
    for (idx_t i = 0; i < nnz /2; ++i) {
        idx_t x = rand() % nvtxs;
        idx_t y = rand() % nvtxs;
        A.coeffRef(x, y) = 1;
        A.coeffRef(y, x) = 1;
    }
*/
    cout << A.nonZeros() << endl;
/*
    dmat_t omega = dmat_t::Zero(nvtxs, k);
    for (idx_t i = 0; i < k; ++i) {
        omega(i,i) = 1;
    }
*/
//    omega.block(0, 0, k, k) = dmat_t::Identity(k, k);

    dmat_t omega = clustering.get_omega();
    double start_time = omp_get_wtime();
    Lanczos lanczos(A, k, omega);
    cout << "Lanczos time = " << omp_get_wtime() - start_time << endl;
    cout << lanczos.eigenValues() << endl;

    return 0;
}

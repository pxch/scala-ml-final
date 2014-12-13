#include "GraphClustering.h"
#include "Lanczos.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: ./clustering file_name nr_threads k" << endl;
        exit(-1);
    }

    string file_name = argv[1];
    int nr_threads = atoi(argv[2]);
    int k = atoi(argv[3]);

	omp_set_num_threads(nr_threads);

    GraphClustering clustering;
    clustering.load_data(file_name);
    clustering.partition(16);
    clustering.calc_subgraph();
//    clustering.calc_omega(k);

    idx_t nvtxs = clustering.get_nvtxs();
    idx_t nnz = clustering.get_nnz();

/*
    smat_t A = clustering.get_new_mat();

    for (int i = 0; i < 100; ++i) {
        int x = rand() % nvtxs;
        int y = rand() % nvtxs;
        cout << x << "\t" << y << "\t" << A.coeff(x,y) << "\t" << A.coeff(y,x) << endl;
    }
*/
    smat_t A = smat_t(nvtxs, nvtxs);
    for (idx_t i = 0; i < nnz/2; ++i) {
        idx_t x = rand() % nvtxs;
        idx_t y = rand() % nvtxs;
        A.coeffRef(x, y) = 1;
        A.coeffRef(y, x) = 1;
    }

    dmat_t omega = dmat_t::Zero(nvtxs, k);
    omega.block(0, 0, k, k) = dmat_t::Identity(k, k);

    double start_time = omp_get_wtime();
    Lanczos lanczos(A, k, omega);
    cout << "Lanczos time = " << omp_get_wtime() - start_time << endl;
    cout << lanczos.eigenValues() << endl;

    return 0;
}

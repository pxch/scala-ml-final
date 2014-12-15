#include "GraphClustering.h"
#include "Lanczos.h"
#include "mseigs.h"

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

    smat_t A;
    mseigs::load_data(file_name, A);
    dmat_t omega;
    vec_t lambda;

    double start_time = omp_get_wtime();
    mseigs::eigenvalues(A, k, omega, lambda, 0, 0);
    cout << "Done computing. Elapsed time = " << omp_get_wtime() - start_time << " s" << endl;

    cout << lambda << endl;

    return 0;
}

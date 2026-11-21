// Phase 0 smoke test: proves MPI (processes) and OpenMP (threads) both work.
// Each MPI rank starts a team of OpenMP threads; every thread prints its IDs.
#include <mpi.h>
#include <omp.h>
#include <cstdio>

int main(int argc, char** argv) {
    int provided = 0;
    // FUNNELED = only the main thread will call MPI (the same mode the real program uses).
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #pragma omp parallel
    {
        std::printf("hello from rank %d of %d, thread %d of %d\n",
                    rank, size, omp_get_thread_num(), omp_get_num_threads());
    }
    MPI_Finalize();
    return 0;
}

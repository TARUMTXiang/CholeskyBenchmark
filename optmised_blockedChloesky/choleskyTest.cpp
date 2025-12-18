#include <windows.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <malloc.h>

using namespace std;

// ----------------------------- Helpers -----------------------------
inline size_t idx(size_t n, size_t i, size_t j) { return i * n + j; }

// Function to convert seconds to ms using QueryPerformanceCounter
double now_milliseconds() {
    static LARGE_INTEGER freq = []() {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
        }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double(t.QuadPart) / double(freq.QuadPart)) * 1000.0; // convert seconds to ms
}

// Aligned memory allocation
double* aligned_alloc_double(size_t elems) { return static_cast<double*>(_aligned_malloc(elems * sizeof(double), 64)); }
void aligned_free_double(double* ptr) { _aligned_free(ptr); }

// Function to flush CPU cache
void flush_cpu_cache(size_t bytes = 200ull * 1024 * 1024) {
    static vector<char> dummy;
    dummy.assign(bytes, 1);
    volatile char s = 0;
    for (size_t i = 0; i < dummy.size(); i += 64) s ^= dummy[i];
    (void)s;
}

// Startup OpenMP threads
void omp_warmup() {
#pragma omp parallel
    {
        volatile double x = 0;
        int tid = omp_get_thread_num();
        for (int i = 0; i < 1000; ++i) x += (i + tid) * 1e-6;
    }
}

// ----------------------------- Generate SPD matrix -----------------------------
void generateSPD_flat(double* A, size_t n, double shift = 100.0, uint64_t seed = 123456789ULL) {
    mt19937_64 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);

    double* R = aligned_alloc_double(n * n);
    for (size_t i = 0; i < n * n; ++i) R[i] = dist(rng);

    for (size_t i = 0; i < n * n; ++i) A[i] = 0.0;

    for (size_t i = 0; i < n; ++i)
        for (size_t j = i; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += R[k * n + i] * R[k * n + j];
            A[idx(n, i, j)] = s;
            A[idx(n, j, i)] = s;
        }

    for (size_t i = 0; i < n; ++i) A[idx(n, i, i)] += shift;
    aligned_free_double(R);
}

// ----------------------------- Cholesky algorithms -----------------------------
// Sequential Cholesky decomposition
bool cholesky_seq_flat(const double* A, double* L, size_t n) {
    for (size_t i = 0; i < n * n; ++i) L[i] = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) s += L[idx(n, i, k)] * L[idx(n, j, k)];
            if (i == j) {
                double diag = A[idx(n, i, i)] - s;
                if (diag <= 0.0) return false;
                L[idx(n, i, i)] = sqrt(diag);
            }
            else {
                L[idx(n, i, j)] = (A[idx(n, i, j)] - s) / L[idx(n, j, j)];
            }
        }
    }
    return true;
}

// Parallel Cholesky decomposition
bool cholesky_par_flat(const double* A, double* L, size_t n) {
    for (size_t i = 0; i < n * n; ++i) L[i] = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < i; ++k) s += L[idx(n, i, k)] * L[idx(n, i, k)];
        double diag = A[idx(n, i, i)] - s;
        if (diag <= 0.0) return false;
        double Lii = sqrt(diag);
        L[idx(n, i, i)] = Lii;

		// Parallelize the computation of the i-th column below the diagonal, dynamic scheduling with chunk size 32
#pragma omp parallel for schedule(dynamic, 32)
        for (long jj = (long)i + 1; jj < (long)n; ++jj) {
            size_t j = (size_t)jj;
            double s2 = 0.0;
            for (size_t k = 0; k < i; ++k) s2 += L[idx(n, j, k)] * L[idx(n, i, k)];
            L[idx(n, j, i)] = (A[idx(n, j, i)] - s2) / Lii;
        }
    }
    return true;
}

// Blocked Cholesky decomposition
bool cholesky_blocked_flat(const double* A_in, double* L_out, size_t n, size_t b) {
    double* A = aligned_alloc_double(n * n);
    for (size_t i = 0; i < n * n; ++i) A[i] = A_in[i];
    for (size_t i = 0; i < n * n; ++i) L_out[i] = 0.0;

    for (size_t k = 0; k < n; k += b) {
        size_t kb = min(b, n - k);
        for (size_t ii = 0; ii < kb; ++ii)
            for (size_t jj = 0; jj <= ii; ++jj) {
                double s = A[idx(n, k + ii, k + jj)];
                for (size_t t = 0; t < jj; ++t) s -= L_out[idx(n, k + ii, k + t)] * L_out[idx(n, k + jj, k + t)];
                if (ii == jj) {
                    if (s <= 0.0) { aligned_free_double(A); return false; }
                    L_out[idx(n, k + ii, k + ii)] = sqrt(s);
                }
                else {
                    L_out[idx(n, k + ii, k + jj)] = s / L_out[idx(n, k + jj, k + jj)];
                }
            }
        if (k + kb >= n) break;

        size_t start = k + kb;
        size_t rows = n - start;
		// Update block column below the diagonal
#pragma omp parallel for schedule(dynamic,8)
        for (long ii = 0; ii < (long)rows; ++ii) {
            size_t i = start + (size_t)ii;
            for (size_t j = 0; j < kb; ++j) {
                double s = A[idx(n, i, k + j)];
                for (size_t t = 0; t < j; ++t) s -= L_out[idx(n, i, k + t)] * L_out[idx(n, k + j, k + t)];
                L_out[idx(n, i, k + j)] = s / L_out[idx(n, k + j, k + j)];
            }
        }

		// Update trailing submatrix
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (long ii = (long)start; ii < (long)n; ++ii) {
            for (long jj = ii; jj < (long)n; ++jj) {
                size_t i = (size_t)ii;
                size_t j = (size_t)jj;
                double s = A[idx(n, i, j)];
                double acc = 0.0;
                for (size_t t = 0; t < kb; ++t) acc += L_out[idx(n, i, k + t)] * L_out[idx(n, j, k + t)];
                s -= acc;
                A[idx(n, i, j)] = s;
                A[idx(n, j, i)] = s;
            }
        }
    }

    aligned_free_double(A);
    return true;
}

// ----------------------------- statistics -----------------------------
struct Stats { double mean; double stddev; };
Stats summarize(const vector<double>& samples) {
    double mean = accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double var = 0.0;
    for (double x : samples) var += (x - mean) * (x - mean);
    var /= samples.size();
    return { mean, sqrt(var) };
}

// ----------------------------- benchmark runner -----------------------------
void run_benchmark(size_t n, int max_threads, int trials, size_t block) {
    cout << "=== Matrix size n=" << n << " ===\n";

    double* A = aligned_alloc_double(n * n);
    double* L_seq = aligned_alloc_double(n * n);
    double* L_par = aligned_alloc_double(n * n);
    double* L_blk = aligned_alloc_double(n * n);

    generateSPD_flat(A, n, 100.0, 123456789ULL);

    // --- Serial (single run)
    vector<double> seq_times;
    for (int t = 0; t < trials; ++t) {
        flush_cpu_cache(); omp_warmup();
        double ts = now_milliseconds();
        bool ok = cholesky_seq_flat(A, L_seq, n);
        double te = now_milliseconds();
        if (!ok) { cerr << "Sequential failed\n"; break; }
        seq_times.push_back(te - ts);
    }
    Stats seq_stats = summarize(seq_times);
    cout << fixed << setprecision(4);
    cout << "Sequential avg: " << seq_stats.mean << " ms\n";

    // --- Parallel and Blocked for threads 1..max_threads
    for (int threads = 1; threads <= max_threads; ++threads) {
        omp_set_num_threads(threads); omp_set_dynamic(0);

        vector<double> par_times, blk_times;

        for (int t = 0; t < trials; ++t) {
            flush_cpu_cache(); omp_warmup();
            double ts = now_milliseconds();
            bool ok = cholesky_par_flat(A, L_par, n);
            double te = now_milliseconds();
            if (!ok) { cerr << "Parallel failed\n"; break; }
            par_times.push_back(te - ts);
        }

        for (int t = 0; t < trials; ++t) {
            flush_cpu_cache(); omp_warmup();
            double ts = now_milliseconds();
            bool ok = cholesky_blocked_flat(A, L_blk, n, block);
            double te = now_milliseconds();
            if (!ok) { cerr << "Blocked failed\n"; break; }
            blk_times.push_back(te - ts);
        }

        Stats par_stats = summarize(par_times);
        Stats blk_stats = summarize(blk_times);

        double speedup_par = seq_stats.mean / par_stats.mean;
        double eff_par = speedup_par / threads;
        double speedup_blk = seq_stats.mean / blk_stats.mean;
        double eff_blk = speedup_blk / threads;

        cout << "Threads=" << threads
            << " | Parallel avg=" << par_stats.mean << " ms, speedup=" << speedup_par << ", eff=" << eff_par
            << " | Blocked avg=" << blk_stats.mean << " ms, speedup=" << speedup_blk << ", eff=" << eff_blk << "\n";
    }

    aligned_free_double(A);
    aligned_free_double(L_seq);
    aligned_free_double(L_par);
    aligned_free_double(L_blk);
    cout << "\n";
}

// ----------------------------- main -----------------------------
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    vector<size_t> sizes = { 100, 500, 1000, 1500, 2000 };
    int trials = 5;
    size_t block = 64;
    int max_threads = 4;

    for (size_t n : sizes)
        run_benchmark(n, max_threads, trials, block);

    return 0;
}

// Histogram + threshold segmentation of a CT volume.
//   1. histogram (256 bins) - two versions:
//        global : every thread does atomicAdd on the global histogram (heavy contention)
//        shared : each block builds a private histogram in shared memory, then merges
//   2. host picks thresholds with Otsu's method from the 256 bins (microseconds)
//   3. label kernel: one thread per voxel -> 0 background, 1 soft tissue, 2 bone
//
// Usage: threshold.exe <vol_in.bin> <labels_out.bin> <hist_out.bin> <n_voxels> <t1> <t2> <reps>
//   Volume is float32 in [0,1]. If t1 < 0 the program only computes the histogram
//   (written to hist_out) so Python/host can run Otsu; then it is called again with thresholds.
// Prints one JSON line with timings in milliseconds.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

#define BINS 256

__device__ __forceinline__ int to_bin(float v)
{
    int b = (int)(v * BINS);
    return b < 0 ? 0 : (b >= BINS ? BINS - 1 : b);
}

__global__ void hist_global(const float* __restrict__ vol, unsigned int* hist, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        atomicAdd(&hist[to_bin(vol[i])], 1u);
}

__global__ void hist_shared(const float* __restrict__ vol, unsigned int* hist, size_t n)
{
    __shared__ unsigned int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        atomicAdd(&local[to_bin(vol[i])], 1u);                 // fast on-chip atomics
    __syncthreads();
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        if (local[b]) atomicAdd(&hist[b], local[b]);           // one global add per bin per block
}

__global__ void label(const float* __restrict__ vol, unsigned char* __restrict__ out, size_t n, float t1, float t2)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = vol[i];
        out[i] = v > t2 ? 2 : (v > t1 ? 1 : 0);
    }
}

int main(int argc, char** argv)
{
    if (argc < 8) { fprintf(stderr, "usage: threshold vol labels hist n t1 t2 reps\n"); return 1; }
    size_t n = strtoull(argv[4], nullptr, 10);
    float t1 = (float)atof(argv[5]), t2 = (float)atof(argv[6]);
    int reps = atoi(argv[7]);

    std::vector<float> hVol(n);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(hVol.data(), sizeof(float), n, f) != n) { fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);

    float* dVol; unsigned int* dHist; unsigned char* dLab;
    CHECK(cudaMalloc(&dVol, n * sizeof(float)));
    CHECK(cudaMalloc(&dHist, BINS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&dLab, n));
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);

    int dev; cudaGetDevice(&dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int hBlocks = sms * 8, threads = 256;                        // grid-stride histogram
    int lBlocks = (int)((n + threads - 1) / threads);            // one thread per voxel

    cudaEventRecord(e0);
    CHECK(cudaMemcpy(dVol, hVol.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    CHECK(cudaEventSynchronize(e1));

    // warm-up
    cudaMemset(dHist, 0, BINS * sizeof(unsigned int));
    hist_shared<<<hBlocks, threads>>>(dVol, dHist, n);
    hist_global<<<hBlocks, threads>>>(dVol, dHist, n);
    label<<<lBlocks, threads>>>(dVol, dLab, n, 0.5f, 0.9f);
    CHECK(cudaDeviceSynchronize());

    float tg = 0, ts = 0, tl = 0, ms;
    for (int r = 0; r < reps; ++r) {
        cudaMemset(dHist, 0, BINS * sizeof(unsigned int));
        cudaEventRecord(e2); hist_global<<<hBlocks, threads>>>(dVol, dHist, n); cudaEventRecord(e3);
        cudaEventSynchronize(e3); cudaEventElapsedTime(&ms, e2, e3); tg += ms;
    }
    std::vector<unsigned int> hHistG(BINS), hHist(BINS);
    CHECK(cudaMemcpy(hHistG.data(), dHist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int r = 0; r < reps; ++r) {
        cudaMemset(dHist, 0, BINS * sizeof(unsigned int));
        cudaEventRecord(e2); hist_shared<<<hBlocks, threads>>>(dVol, dHist, n); cudaEventRecord(e3);
        cudaEventSynchronize(e3); cudaEventElapsedTime(&ms, e2, e3); ts += ms;
    }
    CHECK(cudaMemcpy(hHist.data(), dHist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool same = hHist == hHistG;

    f = fopen(argv[3], "wb");
    fwrite(hHist.data(), sizeof(unsigned int), BINS, f);
    fclose(f);

    float d2h = 0;
    if (t1 >= 0) {
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e2); label<<<lBlocks, threads>>>(dVol, dLab, n, t1, t2); cudaEventRecord(e3);
            cudaEventSynchronize(e3); cudaEventElapsedTime(&ms, e2, e3); tl += ms;
        }
        std::vector<unsigned char> hLab(n);
        cudaEventRecord(e4);
        CHECK(cudaMemcpy(hLab.data(), dLab, n, cudaMemcpyDeviceToHost));
        cudaEventRecord(e5);
        CHECK(cudaEventSynchronize(e5));
        cudaEventElapsedTime(&d2h, e4, e5);
        f = fopen(argv[2], "wb");
        fwrite(hLab.data(), 1, n, f);
        fclose(f);
    }
    float h2d; cudaEventElapsedTime(&h2d, e0, e1);
    printf("{\"h2d_ms\": %.4f, \"hist_global_ms\": %.4f, \"hist_shared_ms\": %.4f, \"label_ms\": %.4f, "
           "\"d2h_ms\": %.4f, \"hist_match\": %s, \"hist_blocks\": %d, \"label_blocks\": %d}\n",
           h2d, tg / reps, ts / reps, t1 >= 0 ? tl / reps : 0.0f, d2h, same ? "true" : "false", hBlocks, lBlocks);
    cudaFree(dVol); cudaFree(dHist); cudaFree(dLab);
    return 0;
}

// 3D convolution (5x5x5 Gaussian denoising filter) on a CT volume.
// Two versions of the same computation:
//   naive : every thread reads its 125 neighbours straight from global memory
//   tiled : each 8x8x8 block first copies a 12x12x12 halo tile into fast
//           on-chip shared memory, then all threads read neighbours from there
// Borders use clamp-to-edge (same as scipy.ndimage mode='nearest').
//
// Usage: conv3d.exe <vol_in.bin> <vol_out.bin> <kernel.bin> <D> <H> <W> <reps>
// Prints one JSON line with timings in milliseconds.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

#define R 2                 // kernel radius -> 5x5x5
#define K (2 * R + 1)
#define T 8                 // output tile edge per block
#define S (T + 2 * R)       // shared tile edge with halo

__constant__ float c_w[K * K * K];

__device__ __forceinline__ int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

__global__ void conv_naive(const float* __restrict__ in, float* __restrict__ out, int D, int H, int W)
{
    int x = blockIdx.x * T + threadIdx.x;
    int y = blockIdx.y * T + threadIdx.y;
    int z = blockIdx.z * T + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;

    float acc = 0.0f;
    for (int dz = -R; dz <= R; ++dz)
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                int zz = clampi(z + dz, D - 1), yy = clampi(y + dy, H - 1), xx = clampi(x + dx, W - 1);
                acc += c_w[((dz + R) * K + (dy + R)) * K + (dx + R)] * in[((size_t)zz * H + yy) * W + xx];
            }
    out[((size_t)z * H + y) * W + x] = acc;
}

__global__ void conv_tiled(const float* __restrict__ in, float* __restrict__ out, int D, int H, int W)
{
    __shared__ float tile[S][S][S];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x * T - R, by = blockIdx.y * T - R, bz = blockIdx.z * T - R;

    // cooperative load of the halo tile: 512 threads copy 1728 values
    int tid = (tz * T + ty) * T + tx;
    for (int i = tid; i < S * S * S; i += T * T * T) {
        int lz = i / (S * S), ly = (i / S) % S, lx = i % S;
        int gz = clampi(bz + lz, D - 1), gy = clampi(by + ly, H - 1), gx = clampi(bx + lx, W - 1);
        tile[lz][ly][lx] = in[((size_t)gz * H + gy) * W + gx];
    }
    __syncthreads();

    int x = blockIdx.x * T + tx, y = blockIdx.y * T + ty, z = blockIdx.z * T + tz;
    if (x >= W || y >= H || z >= D) return;

    float acc = 0.0f;
    #pragma unroll
    for (int dz = 0; dz < K; ++dz)
        #pragma unroll
        for (int dy = 0; dy < K; ++dy)
            #pragma unroll
            for (int dx = 0; dx < K; ++dx)
                acc += c_w[(dz * K + dy) * K + dx] * tile[tz + dz][ty + dy][tx + dx];
    out[((size_t)z * H + y) * W + x] = acc;
}

int main(int argc, char** argv)
{
    if (argc < 8) { fprintf(stderr, "usage: conv3d in out kernel D H W reps\n"); return 1; }
    int D = atoi(argv[4]), H = atoi(argv[5]), W = atoi(argv[6]), reps = atoi(argv[7]);
    size_t n = (size_t)D * H * W;

    std::vector<float> hIn(n), hOut(n), hW(K * K * K);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(hIn.data(), sizeof(float), n, f) != n) { fprintf(stderr, "read vol failed\n"); return 1; }
    fclose(f);
    f = fopen(argv[3], "rb");
    if (!f || fread(hW.data(), sizeof(float), K * K * K, f) != K * K * K) { fprintf(stderr, "read kernel failed\n"); return 1; }
    fclose(f);
    CHECK(cudaMemcpyToSymbol(c_w, hW.data(), hW.size() * sizeof(float)));

    float *dIn, *dOut;
    CHECK(cudaMalloc(&dIn, n * sizeof(float)));
    CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);

    dim3 block(T, T, T);
    dim3 grid((W + T - 1) / T, (H + T - 1) / T, (D + T - 1) / T);

    cudaEventRecord(e0);
    CHECK(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    conv_naive<<<grid, block>>>(dIn, dOut, D, H, W);           // warm-up
    conv_tiled<<<grid, block>>>(dIn, dOut, D, H, W);
    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(e1);
    for (int r = 0; r < reps; ++r) conv_naive<<<grid, block>>>(dIn, dOut, D, H, W);
    cudaEventRecord(e2);
    for (int r = 0; r < reps; ++r) conv_tiled<<<grid, block>>>(dIn, dOut, D, H, W);
    cudaEventRecord(e3);
    CHECK(cudaMemcpy(hOut.data(), dOut, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(e4);
    CHECK(cudaEventSynchronize(e4));

    float h2d, naive, tiled, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);   // includes warm-up; re-measured below
    cudaEventElapsedTime(&naive, e1, e2);
    cudaEventElapsedTime(&tiled, e2, e3);
    cudaEventElapsedTime(&d2h, e3, e4);

    // clean H2D measurement
    cudaEventRecord(e0);
    CHECK(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    CHECK(cudaEventSynchronize(e1));
    cudaEventElapsedTime(&h2d, e0, e1);

    f = fopen(argv[2], "wb");
    fwrite(hOut.data(), sizeof(float), n, f);
    fclose(f);

    printf("{\"h2d_ms\": %.4f, \"naive_ms\": %.4f, \"tiled_ms\": %.4f, \"d2h_ms\": %.4f, \"voxels\": %zu}\n",
           h2d, naive / reps, tiled / reps, d2h, n);
    cudaFree(dIn); cudaFree(dOut);
    return 0;
}

// Filtered Back Projection (back-projection step) for a stack of CT slices.
// One CUDA thread reconstructs one pixel of one slice: it walks over every
// projection angle, finds where that pixel lands on the detector and
// linearly interpolates the filtered sinogram value.
//
// Usage: fbp.exe <sino_in.bin> <img_out.bin> <slices> <angles> <detectors> <N> <reps>
// Input : float32 [slices][angles][detectors]  (already ramp-filtered)
//         float32 [angles] theta in radians is NOT needed: angles are 0..180 evenly spaced
// Output: float32 [slices][N][N]
// Prints one JSON line with timings in milliseconds.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__constant__ float c_cos[2048];
__constant__ float c_sin[2048];

__global__ void backproject(const float* __restrict__ sino, float* __restrict__ img,
                            int A, int D, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int s   = blockIdx.z;                      // slice index
    if (col >= N || row >= N) return;

    const float* sv = sino + (size_t)s * A * D;
    float x = (float)(row - N / 2);            // same convention as skimage.iradon
    float y = (float)(col - N / 2);
    float centre = (float)(D / 2);
    float acc = 0.0f;

    for (int a = 0; a < A; ++a) {
        float t = y * c_cos[a] - x * c_sin[a] + centre;
        int   i = (int)floorf(t);
        float w = t - (float)i;
        if (i >= 0 && i + 1 < D) {
            const float* p = sv + a * D;
            acc += (1.0f - w) * __ldg(p + i) + w * __ldg(p + i + 1);
        }
    }
    img[(size_t)s * N * N + (size_t)row * N + col] = acc * 3.14159265f / (2.0f * A);
}

int main(int argc, char** argv)
{
    if (argc < 8) { fprintf(stderr, "usage: fbp in out S A D N reps\n"); return 1; }
    int S = atoi(argv[3]), A = atoi(argv[4]), D = atoi(argv[5]), N = atoi(argv[6]), reps = atoi(argv[7]);
    size_t nSino = (size_t)S * A * D, nImg = (size_t)S * N * N;

    std::vector<float> hSino(nSino), hImg(nImg);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(hSino.data(), sizeof(float), nSino, f) != nSino) { fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);

    std::vector<float> hc(A), hs(A);
    for (int a = 0; a < A; ++a) {
        double th = a * 3.14159265358979 / A;
        hc[a] = (float)cos(th); hs[a] = (float)sin(th);
    }
    CHECK(cudaMemcpyToSymbol(c_cos, hc.data(), A * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_sin, hs.data(), A * sizeof(float)));

    float *dSino, *dImg;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    CHECK(cudaMalloc(&dSino, nSino * sizeof(float)));
    CHECK(cudaMalloc(&dImg, nImg * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16, S);
    backproject<<<grid, block>>>(dSino, dImg, A, D, N);   // warm-up
    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(e0);
    CHECK(cudaMemcpy(dSino, hSino.data(), nSino * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    for (int r = 0; r < reps; ++r) backproject<<<grid, block>>>(dSino, dImg, A, D, N);
    cudaEventRecord(e2);
    CHECK(cudaMemcpy(hImg.data(), dImg, nImg * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(e3);
    CHECK(cudaEventSynchronize(e3));

    float h2d, kern, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);
    cudaEventElapsedTime(&kern, e1, e2);
    cudaEventElapsedTime(&d2h, e2, e3);

    f = fopen(argv[2], "wb");
    fwrite(hImg.data(), sizeof(float), nImg, f);
    fclose(f);

    printf("{\"h2d_ms\": %.4f, \"kernel_ms\": %.4f, \"d2h_ms\": %.4f, \"threads\": %zu}\n",
           h2d, kern / reps, d2h, nImg);
    cudaFree(dSino); cudaFree(dImg);
    return 0;
}

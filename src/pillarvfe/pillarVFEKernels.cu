// PillarVFE plugin kernel: fuses PointPillars point-feature-augmentation (reader)
// + PFN(linear 11->64, relu, maxpool over points) into one kernel.
// Output: pillar features [P, 64] (fp16 or fp32), ready to feed PPScatter.
//
// Verified math (vs ORT) — per pillar p with n=num_points[p] valid points:
//   raw point j: (x,y,z,f3,f4) = features[p,j,0:5]
//   mean_xyz = sum_{j<n}(x,y,z)/n
//   11-ch feature per point j:
//     [0:3] = (x/norm.x, y/norm.y, z/norm.z)
//     [3:5] = (f3, f4)
//     [5:8] = (x,y,z) - mean_xyz
//     [8]   = x - (coors[p,3]*vsize.x + offset.x)
//     [9]   = y - (coors[p,2]*vsize.y + offset.y)
//     [10]  = z - (coors[p,1]*vsize.z + offset.z)
//     * mask(j<n)   (padded points -> 0)
//   pfn: out[c] = max_{j<n} relu( sum_k W[c,k]*f11[j,k] + B[c] ),  c=0..63
//   empty pillar (n==0) -> all zeros.
//
// norm/vsize/offset are passed in (plugin attributes) so one .so serves models
// with different point-cloud ranges. AEB: norm=(183.2,48,5); multitask-od:
// norm=(151.2,48,5). Both: vsize=(0.2,0.2,8.0), offset=(-27.9,-47.9,1.0).

#include <stdio.h>
#include "vfe_kernel.h"

#define VFE_C 11
#define VFE_OUT 64
#define VFE_MAXP 20

// One block per pillar; blockDim.x = 64 threads (one per output channel).
// Each thread computes one output channel's max over the (<=20) valid points.
// TIn: input features dtype (fp32 recommended — feeding fp16 loses accuracy on
//      large raw coords used in cluster/center offsets). TOut: output dtype.
template <typename TIn, typename TOut>
__global__ void pillarVFEKernel(
    const TIn* __restrict__ features,    // [P,maxp,5]
    const int* __restrict__ coors,       // [P,4]
    const int* __restrict__ num_points,  // [P]
    const float* __restrict__ W,         // [64,11]
    const float* __restrict__ Bias,      // [64]
    float3 norm,                         // xyz normalization divisors
    float3 vsize,                        // voxel size xyz
    float3 offset,                       // range-min offset xyz
    int maxp,                            // points per pillar (AEB/od:20, ld:10)
    float clusterZDiv,                   // extra div on cluster-offset z (ld:8, else 1)
    float centerZDiv,                    // extra div on center-offset z  (ld:4, else 1)
    int P,
    TOut* __restrict__ out)              // [P,64]
{
  int p = blockIdx.x;
  if (p >= P) return;
  int c = threadIdx.x;                   // 0..63 output channel
  int n = num_points[p];
  if (n > maxp) n = maxp;
  if (n < 0) n = 0;

  // shared: the 11-ch features for all points + mean (buffer sized for the max maxp=20)
  __shared__ float sf[VFE_MAXP][VFE_C];
  __shared__ float smean[3];

  // compute mean (thread 0..2 handle x,y,z)
  if (c < 3) {
    float s = 0.f;
    for (int j = 0; j < n; ++j) s += (float)features[(p*maxp + j)*5 + c];
    smean[c] = (n > 0) ? s / (float)n : 0.f;
  }
  __syncthreads();

  // build 11-ch features (first maxp threads each build one point's row; we have 64 threads)
  if (c < maxp) {
    int j = c;
    if (j < n) {
      const TIn* pt = features + (p*maxp + j)*5;
      float x=(float)pt[0], y=(float)pt[1], z=(float)pt[2], f3=(float)pt[3], f4=(float)pt[4];
      float* d = sf[j];
      d[0]=x/norm.x; d[1]=y/norm.y; d[2]=z/norm.z;
      d[3]=f3; d[4]=f4;
      d[5]=x-smean[0]; d[6]=y-smean[1]; d[7]=(z-smean[2])/clusterZDiv;
      d[8] = x - ((float)coors[p*4+3]*vsize.x + offset.x);
      d[9] = y - ((float)coors[p*4+2]*vsize.y + offset.y);
      d[10]=(z - ((float)coors[p*4+1]*vsize.z + offset.z))/centerZDiv;
    } else {
      #pragma unroll
      for (int k=0;k<VFE_C;++k) sf[j][k]=0.f;
    }
  }
  __syncthreads();

  // each thread c: linear(11->1) + relu per point, max over valid points.
  // NOTE: original graph masks padding points' 11-ch features to 0, then does
  //   linear(11->64)+Bias+ReLU+MaxPool over ALL maxp slots. A padding slot yields
  //   relu(0*W + B) = relu(B), and MaxPool includes it. So when the pillar has
  //   any padding slot (n < maxp), the channel max must also consider relu(B[c]).
  float acc = -1e30f;
  const float* wc = W + c*VFE_C;
  float b = Bias[c];
  for (int j = 0; j < n; ++j) {
    float v = b;
    #pragma unroll
    for (int k=0;k<VFE_C;++k) v += wc[k]*sf[j][k];
    v = v > 0.f ? v : 0.f;             // relu
    acc = v > acc ? v : acc;           // max
  }
  if (n < maxp) {                      // padding slots contribute relu(B)
    float rb = b > 0.f ? b : 0.f;
    acc = rb > acc ? rb : acc;
  }
  if (n == 0) acc = 0.f;              // empty pillar -> all zeros (matches graph: whole pillar masked)
  out[p*VFE_OUT + c] = (TOut)acc;
}

template <typename TIn, typename TOut>
int pillarVFELaunch(const TIn* features, const int* coors, const int* num_points,
                    const float* W, const float* Bias,
                    float3 norm, float3 vsize, float3 offset,
                    int maxp, float clusterZDiv, float centerZDiv,
                    int P, TOut* out, cudaStream_t stream) {
  dim3 blocks(P);
  dim3 threads(VFE_OUT);   // 64
  pillarVFEKernel<TIn, TOut><<<blocks, threads, 0, stream>>>(
      features, coors, num_points, W, Bias, norm, vsize, offset, maxp, clusterZDiv, centerZDiv, P, out);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { fprintf(stderr, "pillarVFELaunch failed: %s\n", cudaGetErrorString(err)); return -1; }
  return 0;
}

// Instantiate the combinations we use:
//   fp32 in -> fp16 out (recommended: full-precision reader math, fp16 to feed PPScatter)
//   fp32 in -> fp32 out ; fp16 in -> fp16 out (back-compat)
template int pillarVFELaunch<float, __half>(const float*, const int*, const int*, const float*, const float*, float3, float3, float3, int, float, float, int, __half*, cudaStream_t);
template int pillarVFELaunch<float, float>(const float*, const int*, const int*, const float*, const float*, float3, float3, float3, int, float, float, int, float*, cudaStream_t);
template int pillarVFELaunch<__half, __half>(const __half*, const int*, const int*, const float*, const float*, float3, float3, float3, int, float, float, int, __half*, cudaStream_t);

#include "conv2dfused.cuh"
#include <mma.h>

// block_nums((output_width*output_height) / image_block, output_channels/16, batch_size)
// block_dim(WARP_SIZE, num_warps)
// shmem => num_warps * (image_block * (ic_per_warp * KW * KH)*sizeof(half) + image_block*16*sizeof(float))

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    16, 16, 16, half, nvcuda::wmma::row_major> frag_kernel;
typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    16, 16, 16, half, nvcuda::wmma::col_major> frag_im2col;
typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>                         frag_output;
#endif

template<int image_block, int ic_per_warp>
static __global__ void im2gemm(
        const half* kernel,
        const float* image,
        float* dest,
        int IW, int IH, int IC,
        int OW, int OH, int OC,
        int KW, int kernel_size, int image_size, int im2col_overhead, int p0, int p1, int s0, int s1, int d0, int d1) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
        const int lane_idx = threadIdx.x;
        const int warp_idx = threadIdx.y;
        const int num_warps = blockDim.y;

        extern __shared__ char im2mem[];

        const uint32_t tile_cols = ic_per_warp * kernel_size;
        const uint32_t im2col_tile = image_block * tile_cols;
        const uint32_t warp_data_size = im2col_tile*sizeof(half) + image_block*16*sizeof(float);
        const uint32_t oc_offset = blockIdx.y*16;

        // image_block and ic_per_warp must be multiple of 16
        half* warp_data =    (half*)(im2mem + warp_idx * warp_data_size);
        float* warp_output = (float*)(warp_data + im2col_tile);

        // im2c [image_block, ic_per_warp * KW * KH]
        // kernel [oc, ic_per_warp * KW * KH]

        for(int iic = warp_idx*ic_per_warp; iic < IC; iic += num_warps*ic_per_warp) {
                // generate im2col tile per warp
#pragma unroll
                for(int i0 = 0; i0 < ic_per_warp; i0 += WARP_SIZE) {
                        const int iict = i0 + lane_idx;
                        if(iict >= ic_per_warp) {
                                break;
                        }
                        const uint32_t src_offset = (iic + iict) * IW * IH + blockIdx.z * (IW * IH * IC);
                        for(int ib = 0; ib < image_block; ib ++) {
                                const uint32_t image_index = blockIdx.x * image_block + ib;
                                const int iow = image_index % OW;
                                const int ioh = image_index / OW;
                                const uint32_t dst_offset = ib * tile_cols + iict * kernel_size;
                                for(int ik = 0; ik < kernel_size; ik ++) {
                                        const int iiw = iow * s0 + (ik % KW) * d0 - p0;
                                        const int iih = ioh * s1 + (ik / KW) * d1 - p1;
                                        if(iiw >= 0 && iiw < IW && iih >= 0 && iih < IH) {
                                                warp_data[dst_offset + ik] = __float2half(image[src_offset + iih * IW + iiw]);
                                        } else {
                                                warp_data[dst_offset + ik] = __float2half(0.0f);
                                        }
                                }
                        }
                }

                // perform partial gemm
                frag_kernel km;
                frag_im2col im;
                frag_output om; // [OC, OW*OH]
                for(int ib = 0; ib < image_block; ib += 16) { // OW * OH
                        // recycle previus accumulator
                        if(iic == warp_idx*ic_per_warp) { // first
                                nvcuda::wmma::fill_fragment(om, 0.0f);
                        } else {
                                nvcuda::wmma::load_matrix_sync(om, warp_output + ib*16, 16, nvcuda::wmma::mem_row_major);
                        }
                        for(int itc = 0; itc < tile_cols; itc += 16) { // k
                                nvcuda::wmma::load_matrix_sync(km, kernel + (oc_offset * im2col_overhead) + (iic * kernel_size) + itc, im2col_overhead);
                                nvcuda::wmma::load_matrix_sync(im, warp_data + (ib * tile_cols) + itc, tile_cols);
                                nvcuda::wmma::mma_sync(om, km, im, om);
                        }
                        nvcuda::wmma::store_matrix_sync(warp_output + ib*16, om, 16, nvcuda::wmma::mem_row_major);
                }
        }
        __syncthreads();
        {
                // reduce warps
                const int block_size = blockDim.x * blockDim.y;
                const int thread_index = (threadIdx.y * blockDim.x + threadIdx.x);
                for(int ib = 0; ib < image_block; ib += 16) {
                        for(int i = thread_index; i < 256; i += block_size) {
                                int row = i / 16;
                                int col = i % 16;
                                float acc = 0.0f;
                                for(int w = 0; w < num_warps; w ++) {
                                        const float* src = (const float*)(im2mem + w * warp_data_size + im2col_tile*sizeof(half));
                                        acc += src[ib*16 + row*16 + col];
                                }
                                const uint32_t dst_offset = (oc_offset + row) * image_size + blockIdx.x * image_block + (ib + col) + blockIdx.z * (image_size*OC);
                                dest[dst_offset] = acc;
                        }
                }
        }
#else
   NO_DEVICE_CODE;
#endif
}


void ggml_cuda_op_conv2d_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const half *    src0_d = (const half *)src0->data;
    const float *   src1_d = (const float *)src1->data;
    float *         dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int image_block = 32;
    constexpr int ic_per_warp = 16;

    const int max_warps = 2;
    int num_warps = std::max(1, std::min(max_warps, (int)(src1->ne[2] / ic_per_warp)));

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    int IW = src1->ne[0], IH = src1->ne[1];
    int KW = src0->ne[0], KH = src0->ne[1];
    int OW = dst->ne[0],  OH = dst->ne[1];
    int IC = src0->ne[2], OC = src0->ne[3];
    GGML_ASSERT(IC == src1->ne[2] && OC == dst->ne[2]);

    dim3 block_nums((OW * OH) / image_block, OC / 16, dst->ne[3]);
    dim3 block_dim(32, num_warps);

    int shram = num_warps * (image_block * ic_per_warp * KW * KH * sizeof(half) + image_block*16*sizeof(float)/* oc per block */);

    im2gemm<image_block, ic_per_warp><<<block_nums, block_dim, shram, stream>>>(src0_d, src1_d, dst_d,
        IW, IH, IC, OW, OH, OC, KW, KW*KH, OW*OH, IC*KW*KH, p0, p1, s0, s1, d0, d1);
    CUDA_CHECK(cudaGetLastError());
}

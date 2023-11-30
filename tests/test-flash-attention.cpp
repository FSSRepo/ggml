#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#define GGML_USE_CUBLAS

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct test_model {
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

void load_model(test_model & model, bool use_gpu = false) {

    float Query[30] = { // [3, 4, 2]
                // z0
                2, 4, 2,
                4, 2, 1,
                4, 1, 3,
                4, 2, 2,

                // z1
                2, 1, 1,
                4, 2, 1,
                1, 1, 3,
                4, 2, 1
    };

    float Key[24] = { // [3, 4, 2]
        // z0
        2, 4, 2,
        4, 2, 1,
        4, 2, 3,
        1, 2, 1,

        // z1
        3, 1, 3,
        4, 2, 1,
        1, 1, 2,
        4, 3, 1
    };

    float Value[24] = { // [4, 3, 2]
        // z0
        2, 4, 2, 1,
        2, 1, 4, 2,
        1, 4, 2, 3,


        // z1
        1, 4, 2, 1,
        2, 1, 1, 2,
        1, 4, 3, 3,
    };

    size_t buffer_size = 0;
    {
        buffer_size += 30 * ggml_type_sizef(GGML_TYPE_F32); // tensor q
        buffer_size += 24 * ggml_type_sizef(GGML_TYPE_F32); // tensor k
        buffer_size += 24 * ggml_type_sizef(GGML_TYPE_F32); // tensor v
        buffer_size += 1024;
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 3;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.q = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 3, 4, 2);
    model.k = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 3, 4, 2);
    model.v = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 4, 3, 2);

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.q);
    ggml_allocr_alloc(alloc, model.k);
    ggml_allocr_alloc(alloc, model.v);

    ggml_backend_tensor_set(model.q, Query, 0, ggml_nbytes(model.q));
    ggml_backend_tensor_set(model.k, Key, 0, ggml_nbytes(model.k));
    ggml_backend_tensor_set(model.v, Value, 0, ggml_nbytes(model.v));

    ggml_allocr_free(alloc);
}

struct ggml_cgraph * build_graph(const test_model& model, struct ggml_allocr * allocr) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    struct ggml_tensor* result = ggml_flash_attn(ctx0, model.q, model.k, model.v, false);
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute_graph(const test_model & model, struct ggml_allocr * allocr) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

    ggml_backend_buffer_t buf_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_tensor * result = compute_graph(model, allocr);
    float* data = new float[ggml_nelements(result)];

    ggml_backend_tensor_get(result, data, 0, ggml_nbytes(result));
    printf("\nPerforming test:\n");

    for(int i = 0; i < ggml_nelements(result); i ++) {
        if(i > 0 && (i % result->ne[0] == 0)) {
            printf("\n");
        }
        printf("%2.6f ", data[i]);
    }

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}

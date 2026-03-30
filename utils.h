#pragma once
#include <cmath>
#include "ggml/include/ggml.h"

#define DPRINT_3d(name, a, b) \
    fprintf(stderr, "[%s:%d] %s: a=[%lld,%lld,%lld] b=[%lld,%lld,%lld]\n", \
        __func__, __LINE__, name, \
        (a)->ne[0],(a)->ne[1],(a)->ne[2], \
        (b)->ne[0],(b)->ne[1],(b)->ne[2])

#define DPRINT_4d(name, a, b) \
    fprintf(stderr, "[%s:%d] %s: a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n", \
        __func__, __LINE__, name, \
        (a)->ne[0],(a)->ne[1],(a)->ne[2],(a)->ne[3], \
        (b)->ne[0],(b)->ne[1],(b)->ne[2],(b)->ne[3])

static int32_t log_bucket_pos(int32_t rel_pos, int bucket_size, int max_position) {
    int mid = bucket_size / 2;
    if (rel_pos > -mid && rel_pos < mid)
        return rel_pos;

    int sign = (rel_pos > 0) ? 1 : -1;
    double abs_pos = (double)std::abs(rel_pos);
    double log_pos = std::ceil(
        std::log(abs_pos / mid) /
        std::log((double)(max_position - 1) / mid) *
        (double)(mid - 1)
    ) + mid;
    return (int32_t)(sign * log_pos);
}

static void gather_batch_custom_op(
    struct ggml_tensor* dst, 
    const struct ggml_tensor* dummy,
    const struct ggml_tensor* src,
    int ith, int nth, void* userdata
) {
    (void)ith; (void)nth; (void)dummy;
    const int32_t* idx  = (const int32_t*)userdata;
    const float* in = (const float*)src->data;
    float* out  = (float*)dst->data;

    const int seq = dst->ne[0];
    const int n_heads = dst->ne[2];
    const int n_pos = src->ne[0];
    const int batch_size = dst->ne[3];

    for (int b = 0; b < batch_size; b++)
        for (int h = 0; h < n_heads; h++)
            for (int i = 0; i < seq; i++)
                for (int j = 0; j < seq; j++)
                    out[j + i*seq + h*seq*seq + b*seq*seq*n_heads] = in[idx[j + i*seq] + i*n_pos + h*n_pos*seq + b*n_pos*seq*n_heads];
}
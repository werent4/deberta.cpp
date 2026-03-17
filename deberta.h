#pragma once
#include <string>
#include <map>
#include <vector>
#include "ggml/include/ggml.h"

struct deberta_hparams {
    int vocab_size;
    int max_position_embeddings;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_hidden_layers;
    int position_buckets;
    int max_relative_positions;
    int ftype;
};

struct deberta_model {
    ggml_context* ctx;
    ggml_type wtype; 
    std::map<std::string, struct ggml_tensor*> tensors;
    deberta_hparams hparams;
};

static ggml_type ftype_to_ggml_type(int ftype) {
    switch (ftype) {
        case 0: return GGML_TYPE_F32;
        case 1: return GGML_TYPE_F16;
        case 2: return GGML_TYPE_Q4_0;
        case 3: return GGML_TYPE_Q4_1;
        default: return GGML_TYPE_COUNT; // invalid
    }
}


struct deberta_ctx {
    deberta_model model;
};

bool deberta_load_hparams(FILE* f, deberta_model & model);

bool deberta_print_tensors(FILE* f);

bool deberta_calc_mem_req(FILE* f, size_t& model_mem_req);

bool deberta_load_weights(FILE* f, struct deberta_model* model);

struct deberta_ctx* deberta_load_from_file(const std::string& fname);

void deberta_free(deberta_ctx* ctx);

ggml_tensor* build_delta(ggml_context* ctx, int seq_len, int k);

struct ggml_cgraph* deberta_build_graph(
    struct deberta_ctx* ctx,
    struct ggml_context* compute_ctx,
    const std::vector<int>& input_ids
);
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

    int   embedding_size;        // v3: 128, v1: == hidden_size
    int   type_vocab_size;       // 
    int   position_biased_input; // v3: 0 (false!), v1: 1
    float layer_norm_eps;  
};

struct deberta_model {
    ggml_context* ctx;
    ggml_type wtype; 
    std::map<std::string, struct ggml_tensor*> tensors;
    deberta_hparams hparams;
};

struct deberta_ctx {
    deberta_model model;
};

struct deberta_attn_tensors {
    ggml_tensor *q_w, *q_b;
    ggml_tensor *k_w, *k_b;
    ggml_tensor *v_w, *v_b;
    ggml_tensor *out_w, *out_b;
    ggml_tensor *ln_w, *ln_b;
};

struct deberta_inter_ffn_tensors {
    ggml_tensor *inter_w, *inter_b;
    ggml_tensor *out_w, *out_b;
    ggml_tensor *ln_w, *ln_b;
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

bool deberta_load_hparams(FILE* f, deberta_model & model);

bool deberta_print_tensors(FILE* f);

bool deberta_calc_mem_req(FILE* f, size_t& model_mem_req);

bool deberta_load_weights(FILE* f, struct deberta_model* model);

struct deberta_ctx* deberta_load_from_file(const std::string& fname);

void deberta_free(deberta_ctx* ctx);

struct ggml_cgraph* deberta_build_graph(
    struct deberta_ctx* ctx,
    struct ggml_context* compute_ctx,
    const std::vector<int>& input_ids
);
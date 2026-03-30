#pragma once
#include <string>
#include <map>
#include <vector>
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"

#define DEBERTA_MAX_NODES 8192
#define HYPER_MAGIC_SIZE 15 // skip hparams + magic (15 integers)

enum deberta_device {
    DEBERTA_DEVICE_CPU,
    DEBERTA_DEVICE_CUDA,
};

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

    int embedding_size;        // v3: 128, v1: == hidden_size
    int type_vocab_size;       // 
    int position_biased_input; // v3: 0 (false!), v1: 1
    float layer_norm_eps;  
    int pos_att_flags;
};

inline bool hparams_use_c2p(const deberta_hparams& h) { return h.pos_att_flags & 1; }
inline bool hparams_use_p2c(const deberta_hparams& h) { return h.pos_att_flags & 2; }

struct deberta_model {
    ggml_context* ctx;
    ggml_type wtype; 
    std::map<std::string, struct ggml_tensor*> tensors;
    deberta_hparams hparams;

    ggml_backend* backend = NULL;
    ggml_backend_buffer_t buffer_w = NULL;
};

struct deberta_ctx {
    deberta_model model;

    ggml_backend_t cpu_backend = NULL;  // gather on CPU offload
    ggml_backend_sched_t sched = NULL;

    ggml_context* ctx_precomp = NULL;
    ggml_tensor* c2p_idx = NULL; // [seq, seq]
    ggml_tensor* p2c_idx = NULL; // [seq, seq]
    int cached_seq_len = 0; // for calculation of p2c/c2p_idx 
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

struct deberta_ctx* deberta_load_from_file(const std::string& fname, const deberta_device device);

void deberta_free(deberta_ctx* ctx);
// batch 

bool deberta_eval(
    deberta_ctx* ctx,
    const int n_threads,
    const std::vector<std::vector<int>> & input_ids,
    const std::vector<std::vector<int>> & attention_mask,
    std::vector<float> & output
);
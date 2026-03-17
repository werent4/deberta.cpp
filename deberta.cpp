#include "deberta.h"
#include <cmath>
#include <cstdio>
#include <cstring>

bool deberta_load_hparams(FILE* f, deberta_model& model) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }

    int magic;
    fread(&magic, sizeof(int), 1, f);
    if (magic != 0x67676d6c) {
        fprintf(stderr, "invalid magic\n");
        fclose(f);
        return false;
    }

    deberta_hparams& hparams = model.hparams;
    fread(&hparams.vocab_size, sizeof(int), 1, f);
    fread(&hparams.max_position_embeddings, sizeof(int), 1, f);
    fread(&hparams.hidden_size, sizeof(int), 1, f);
    fread(&hparams.intermediate_size, sizeof(int), 1, f);
    fread(&hparams.num_attention_heads, sizeof(int), 1, f);
    fread(&hparams.num_hidden_layers, sizeof(int), 1, f);
    fread(&hparams.position_buckets, sizeof(int), 1, f);
    fread(&hparams.max_relative_positions, sizeof(int), 1, f);
    fread(&hparams.ftype, sizeof(int), 1, f);

    printf("vocab_size = %d\n", hparams.vocab_size);
    printf("max_position_embeddings = %d\n", hparams.max_position_embeddings);
    printf("hidden_size = %d\n", hparams.hidden_size);
    printf("intermediate_size = %d\n", hparams.intermediate_size);
    printf("num_attention_heads = %d\n", hparams.num_attention_heads);
    printf("num_hidden_layers = %d\n", hparams.num_hidden_layers);
    printf("position_buckets = %d\n", hparams.position_buckets);
    printf("max_relative_positions = %d\n", hparams.max_relative_positions);
    printf("ftype = %d\n", hparams.ftype);

    fseek(f, 0, SEEK_SET); // move file pointer to the beginning of file

    return true;
}

bool deberta_calc_mem_req(FILE* f, size_t& model_mem_req) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }
    model_mem_req = 0;
    fseek(f, 10 * sizeof(int), SEEK_SET); // skip hparams + magic (10 integers)

    while (true) {
        int n_dims, name_len, ftype;
        if (fread(&n_dims, sizeof(int), 1, f) != 1) break;
        if (fread(&name_len, sizeof(int), 1, f) != 1) break;
        if (fread(&ftype, sizeof(int), 1, f) != 1) break;

        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            if (fread(&dims[i], sizeof(int), 1, f) != 1) break;
        }

        char layer_name[256];
        if (fread(layer_name, sizeof(char), name_len, f) != (size_t)name_len) break;
        layer_name[name_len] = '\0';

        long num_elements = 1;
        for (int i = 0; i < n_dims; i++) num_elements *= dims[i];

        auto tensor_size = ggml_type_size((ggml_type)ftype) * num_elements;
        model_mem_req += tensor_size;
        model_mem_req += ggml_tensor_overhead(); // ggml tensor metadata overhead

        fseek(f, tensor_size, SEEK_CUR);
    }

    model_mem_req += 1024 * 1024; // 1MB for ggml context overhead; but do I rly need this?
    fseek(f, 0, SEEK_SET); // reset file pointer to the beginning
    return true;
}

bool deberta_load_weights(FILE* f, struct deberta_model* model) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }
    auto& tensors = model->tensors;

    fseek(f, 10 * sizeof(int), SEEK_SET); // skip hparams + magic (10 integers)
    while (true) {
        int n_dims, name_len, ftype;
        if (fread(&n_dims, sizeof(int), 1, f) != 1) break;
        if (fread(&name_len, sizeof(int), 1, f) != 1) break;
        if (fread(&ftype, sizeof(int), 1, f) != 1) break;

        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            if (fread(&dims[i], sizeof(int), 1, f) != 1) break;
        }

        char layer_name[256];
        if (fread(layer_name, sizeof(char), name_len, f) != (size_t)name_len) break;
        layer_name[name_len] = '\0';

        long num_elements = 1;
        for (int i = 0; i < n_dims; i++) num_elements *= dims[i];

        auto tensor_size = ggml_type_size((ggml_type)ftype) * num_elements;
        int64_t ne[4] = { dims[0], dims[1], dims[2], dims[3] };
        struct ggml_tensor* tensor = ggml_new_tensor(model->ctx, (ggml_type)ftype, n_dims, ne);
        if (!tensor) {
            fprintf(stderr, "failed to allocate tensor for layer '%s'\n", layer_name);
            return false;
        }
        if (fread(tensor->data, 1, tensor_size, f) != (size_t)tensor_size) {
            fprintf(stderr, "failed to read tensor data for layer '%s'\n", layer_name);
            return false;
        }
        tensors[layer_name] = tensor;
    }
    fseek(f, 0, SEEK_SET); // reset file pointer to the beginning
    return true;
}

struct deberta_ctx* deberta_load_from_file(const std::string & fname) {
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open file '%s'\n", fname.c_str());
        return nullptr;
    }

    struct deberta_ctx* new_deberta_ctx = new struct deberta_ctx();
    deberta_model& model = new_deberta_ctx->model;

    if (!deberta_load_hparams(f, model)) {
        delete new_deberta_ctx;
        return nullptr;
    }

    ggml_type wtype = ftype_to_ggml_type(model.hparams.ftype);
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "invalid ftype %d\n", model.hparams.ftype);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }
    model.wtype = wtype;

    size_t model_mem_req = 0;
    if (!deberta_calc_mem_req(f, model_mem_req)) {
        fprintf(stderr, "%s: failed to calculate memory requirements for model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }
    printf("model memory requirement: %.2f MB\n", model_mem_req / 1024.0 / 1024.0);

    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ model_mem_req,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
        fprintf(stderr, "%s: failed to initialize ggml context for model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    if (!deberta_load_weights(f, &model)) {
        fprintf(stderr, "%s: failed to load weights from model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    fclose(f);
    return new_deberta_ctx;
}

void deberta_free(deberta_ctx* ctx) {
    if (!ctx) return;
    if (ctx->model.ctx) {
        ggml_free(ctx->model.ctx);
    }
    delete ctx;
}

// forward
ggml_tensor* build_delta(ggml_context* ctx, int seq_len, int k) {
    ggml_tensor* delta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, seq_len, seq_len);
    int* data = (int*)delta->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int d = i - j + k;
            d = d < 0 ? 0 : d;
            d = d > 2*k-1 ? 2*k-1 : d;
            data[i * seq_len + j] = d;
        }
    }
    return delta;
}

static void ggml_gather_cr(
    struct ggml_tensor* dst,
    const struct ggml_tensor* placeholder, 
    const struct ggml_tensor* src,
    const struct ggml_tensor* idx,
    int ith, int nth, void* userdata
) {
    int seq_len = idx->ne[0];
    float* src_data = (float*)src->data;
    int*   idx_data = (int*)idx->data;
    float* dst_data = (float*)dst->data;

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int d = idx_data[i * seq_len + j];
            dst_data[i * seq_len + j] = src_data[i * src->ne[0] + d];
        }
    }
}

static void ggml_gather_rc(
    struct ggml_tensor* dst,
    const struct ggml_tensor* placeholder, 
    const struct ggml_tensor* src,
    const struct ggml_tensor* idx,
    int ith, int nth, void* userdata
) {
    int seq_len = idx->ne[0];
    float* src_data = (float*)src->data;
    int*   idx_data = (int*)idx->data;
    float* dst_data = (float*)dst->data;

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int d = idx_data[i * seq_len + j];
            dst_data[i * seq_len + j] = src_data[d * src->ne[1] + j];
        }
    }
}

struct ggml_cgraph* deberta_build_graph(
    struct deberta_ctx* ctx,
    struct ggml_context* compute_ctx,
    const std::vector<int>& input_ids
) {
    int seq_len = input_ids.size();
    ggml_tensor* input_ids_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, seq_len); // todo: select type based on model.wtype
    memcpy(input_ids_tensor->data, input_ids.data(), seq_len * sizeof(int));
    ggml_tensor* word_embeddings = ctx->model.tensors["embeddings.word_embeddings.weight"];
    ggml_tensor* x = ggml_get_rows(compute_ctx, word_embeddings, input_ids_tensor);

    ggml_tensor* ln_w = ctx->model.tensors["embeddings.LayerNorm.weight"];
    ggml_tensor* ln_b = ctx->model.tensors["embeddings.LayerNorm.bias"];

    x = ggml_norm(compute_ctx, x, 1e-7f); // todo: use actual eps from model file
    x = ggml_mul(compute_ctx, x, ln_w);
    x = ggml_add(compute_ctx, x, ln_b);

    int N = ctx->model.hparams.num_hidden_layers;
    for (int i = 0; i < N; i++) {
        std::string layer = "encoder.layer." + std::to_string(i);
        // Self-Attention
        ggml_tensor* q_w = ctx->model.tensors[layer + ".attention.self.query_proj.weight"];
        ggml_tensor* q_b = ctx->model.tensors[layer + ".attention.self.query_proj.bias"];
        ggml_tensor* k_w = ctx->model.tensors[layer + ".attention.self.key_proj.weight"];
        ggml_tensor* k_b = ctx->model.tensors[layer + ".attention.self.key_proj.bias"];
        ggml_tensor* v_w = ctx->model.tensors[layer + ".attention.self.value_proj.weight"];
        ggml_tensor* v_b = ctx->model.tensors[layer + ".attention.self.value_proj.bias"];
        ggml_tensor* attn_out_w = ctx->model.tensors[layer + ".attention.output.dense.weight"];
        ggml_tensor* attn_out_b = ctx->model.tensors[layer + ".attention.output.dense.bias"];
        ggml_tensor* attn_ln_w = ctx->model.tensors[layer + ".attention.output.LayerNorm.weight"];
        ggml_tensor* attn_ln_b = ctx->model.tensors[layer + ".attention.output.LayerNorm.bias"];

        ggml_tensor* Q_c = ggml_mul_mat(compute_ctx, q_w, x);
        Q_c = ggml_add(compute_ctx, Q_c, q_b);
        ggml_tensor* K_c = ggml_mul_mat(compute_ctx, k_w, x);
        K_c = ggml_add(compute_ctx, K_c, k_b);
        ggml_tensor* V_c = ggml_mul_mat(compute_ctx, v_w, x);
        V_c = ggml_add(compute_ctx, V_c, v_b);

        ggml_tensor* P = ctx->model.tensors["encoder.rel_embeddings.weight"];
        ggml_tensor* K_r = ggml_mul_mat(compute_ctx, k_w, P);
        K_r = ggml_add(compute_ctx, K_r, k_b);
        ggml_tensor* Q_r = ggml_mul_mat(compute_ctx, q_w, P);
        Q_r = ggml_add(compute_ctx, Q_r, q_b);

        ggml_tensor* delta = build_delta(compute_ctx, seq_len, ctx->model.hparams.position_buckets);
        ggml_tensor* A_cc = ggml_mul_mat(compute_ctx, K_c, Q_c);

        ggml_tensor* A_cr_full = ggml_mul_mat(compute_ctx, K_r, Q_c);
        ggml_tensor* out_cr = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, seq_len, seq_len); // todo: select type based on model.wtype
        ggml_tensor* A_cr = ggml_map_custom3(compute_ctx, out_cr, A_cr_full, delta, ggml_gather_cr, 1, nullptr);

        ggml_tensor* A_rc_full = ggml_mul_mat(compute_ctx, K_c, Q_r);
        ggml_tensor* out_rc = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, seq_len, seq_len); // todo: select type based on model.wtype
        ggml_tensor* A_rc = ggml_map_custom3(compute_ctx, out_rc, A_rc_full, delta, ggml_gather_rc, 1, nullptr);

        ggml_tensor* A = ggml_add(compute_ctx, A_cc, A_cr);
        A = ggml_add(compute_ctx, A, A_rc);
        float scale = 1.0f / sqrtf(3.0f * ctx->model.hparams.hidden_size);
        A = ggml_scale(compute_ctx, A, scale);


        ggml_tensor* H = ggml_soft_max(compute_ctx, A);
        ggml_tensor* V_c_T = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, V_c));
        H = ggml_mul_mat(compute_ctx, V_c_T, H);

        H = ggml_mul_mat(compute_ctx, attn_out_w, H);
        H = ggml_add(compute_ctx, H, attn_out_b);

        // residual + LayerNorm
        H = ggml_add(compute_ctx, x, H);
        H = ggml_norm(compute_ctx, H, 1e-7f);
        H = ggml_mul(compute_ctx, H, attn_ln_w);
        H = ggml_add(compute_ctx, H, attn_ln_b);

        x = H;

        // FFN
        ggml_tensor* ffn_w1 = ctx->model.tensors[layer + ".intermediate.dense.weight"];
        ggml_tensor* ffn_b1 = ctx->model.tensors[layer + ".intermediate.dense.bias"];
        ggml_tensor* ffn_w2 = ctx->model.tensors[layer + ".output.dense.weight"];
        ggml_tensor* ffn_b2 = ctx->model.tensors[layer + ".output.dense.bias"];
        ggml_tensor* ffn_ln_w = ctx->model.tensors[layer + ".output.LayerNorm.weight"];
        ggml_tensor* ffn_ln_b = ctx->model.tensors[layer + ".output.LayerNorm.bias"];

        ggml_tensor* h = ggml_mul_mat(compute_ctx, ffn_w1, x);
        h = ggml_add(compute_ctx, h, ffn_b1);
        h = ggml_gelu(compute_ctx, h);
        h = ggml_mul_mat(compute_ctx, ffn_w2, h);
        h = ggml_add(compute_ctx, h, ffn_b2);

        // residual + LayerNorm
        h = ggml_add(compute_ctx, x, h);
        h = ggml_norm(compute_ctx, h, 1e-7f); // todo: use actual eps from model file
        h = ggml_mul(compute_ctx, h, ffn_ln_w);
        h = ggml_add(compute_ctx, h, ffn_ln_b);
        x = h;
    }

    struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(gf, x);
    return gf;
}
#include "deberta.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ggml/include/ggml-cpu.h"

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
    fread(&hparams.embedding_size,        sizeof(int),   1, f);
    fread(&hparams.type_vocab_size,       sizeof(int),   1, f);
    fread(&hparams.position_biased_input, sizeof(int),   1, f);
    fread(&hparams.layer_norm_eps,        sizeof(float), 1, f);

    printf("vocab_size = %d\n", hparams.vocab_size);
    printf("max_position_embeddings = %d\n", hparams.max_position_embeddings);
    printf("hidden_size = %d\n", hparams.hidden_size);
    printf("intermediate_size = %d\n", hparams.intermediate_size);
    printf("num_attention_heads = %d\n", hparams.num_attention_heads);
    printf("num_hidden_layers = %d\n", hparams.num_hidden_layers);
    printf("position_buckets = %d\n", hparams.position_buckets);
    printf("max_relative_positions = %d\n", hparams.max_relative_positions);
    printf("ftype = %d\n", hparams.ftype);
    printf("embedding_size = %d\n", hparams.embedding_size);
    printf("type_vocab_size = %d\n", hparams.type_vocab_size);
    printf("position_biased_input = %d\n", hparams.position_biased_input);
    printf("layer_norm_eps = %f\n", hparams.layer_norm_eps);

    fseek(f, 0, SEEK_SET); // move file pointer to the beginning of file

    return true;
}

bool deberta_calc_mem_req(FILE* f, size_t& model_mem_req) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }
    model_mem_req = 0;
    fseek(f, 14 * sizeof(int), SEEK_SET); // skip hparams + magic (10 integers)

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

    fseek(f, 14 * sizeof(int), SEEK_SET); // skip hparams + magic (10 integers)
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
static void gather_custom_op(
    struct ggml_tensor* dst, 
    const struct ggml_tensor* dummy,
    const struct ggml_tensor* src,
    int ith, int nth, void* userdata
) {
    (void)ith; (void)nth; (void)dummy;
    const int32_t* idx  = (const int32_t*)userdata;
    const float*   in   = (const float*)src->data;
    float*         out  = (float*)dst->data;

    const int seq     = dst->ne[0];
    const int n_heads = dst->ne[2];
    const int n_pos   = src->ne[0];

    for (int h = 0; h < n_heads; h++)
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                out[j + i*seq + h*seq*seq] =
                    in[idx[j + i*seq] + i*n_pos + h*n_pos*seq];
}

static ggml_tensor* ggml_gather_axis1(
    ggml_context* ctx,
    ggml_tensor*  src,
    ggml_tensor*  idx_tensor,
    int seq, int n_heads
) {
    ggml_tensor* dummy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq, seq, n_heads);
    return ggml_map_custom2(ctx, dummy, src, gather_custom_op, 1, (void*)idx_tensor->data);
}

static int32_t log_bucket_pos(int32_t rel_pos, int bucket_size, int max_position) {
    int mid = bucket_size / 2;
    if (rel_pos > -mid && rel_pos < mid)
        return rel_pos;

    int sign = (rel_pos > 0) ? 1 : -1;
    float abs_pos = (float)std::abs(rel_pos);
    float log_pos = std::ceil(
        std::log(abs_pos / mid) /
        std::log((float)(max_position - 1) / mid) *
        (float)(mid - 1)
    ) + mid;
    return (int32_t)(sign * log_pos);
}

static ggml_tensor* deberta_build_embeddings(
    struct ggml_context* compute_ctx,
    struct deberta_ctx* ctx,
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

    return x;
}

static ggml_tensor* deberta_build_attention(
    ggml_context* cctx,
    ggml_tensor* x, // [hidden, seq]
    deberta_attn_tensors& T,
    ggml_tensor* rel_emb,  
    int n_heads,
    int head_dim,
    int seq,
    int max_rel,
    int max_pos
) {
    const int hidden = n_heads * head_dim;
    const float scale = sqrtf((float)(head_dim * 3));
    // c2c
    ggml_tensor* Q = ggml_mul_mat(cctx, T.q_w, x);
    Q = ggml_add(cctx, Q, T.q_b);
    
    ggml_tensor* K = ggml_mul_mat(cctx, T.k_w, x);
    K = ggml_add(cctx, K, T.k_b);
    
    ggml_tensor* V = ggml_mul_mat(cctx, T.v_w, x);
    V = ggml_add(cctx, V, T.v_b);

    // ggml layout: ne[0]=head_dim, ne[1]=n_heads, ne[2]=seq
    Q = ggml_reshape_3d(cctx, Q, head_dim, n_heads, seq);
    K = ggml_reshape_3d(cctx, K, head_dim, n_heads, seq);
    V = ggml_reshape_3d(cctx, V, head_dim, n_heads, seq);

    Q = ggml_scale(cctx, Q, 1.0f / scale);

    Q = ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3));  // [head_dim, seq, n_heads]
    K = ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3));
    V = ggml_cont(cctx, ggml_permute(cctx, V, 1, 2, 0, 3)); 

    ggml_tensor* scores = ggml_mul_mat(cctx, K, Q);

    // c2p
    const int att_span = max_rel;
    const int n_pos = 2 * att_span;

    size_t offset = (size_t)(max_rel - att_span) * rel_emb->nb[1];
    ggml_tensor* rel_slice = ggml_view_2d(cctx, rel_emb, rel_emb->ne[0], n_pos, rel_emb->nb[1], 0); // offset


    ggml_tensor* pos_key = ggml_mul_mat(cctx, T.k_w, rel_slice);
    pos_key = ggml_add(cctx, pos_key,
                  ggml_repeat(cctx,
                      ggml_reshape_2d(cctx, T.k_b, T.k_b->ne[0], 1),
                      pos_key));

    pos_key = ggml_reshape_3d(cctx, pos_key, head_dim, n_heads, n_pos);
    pos_key = ggml_cont(cctx, ggml_permute(cctx, pos_key, 0, 2, 1, 3)); // [head_dim, n_pos, n_heads]

    ggml_tensor* c2p_raw = ggml_mul_mat(cctx, pos_key, Q);
    ggml_tensor* c2p_idx = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, seq, seq);
    {
        int32_t* p = (int32_t*)c2p_idx->data;
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++) {
                int32_t raw_c2p = log_bucket_pos(i - j, att_span, max_pos);
                p[j + i*seq] = std::clamp(raw_c2p + att_span, 0, n_pos - 1);
            }
    }

    ggml_tensor* c2p = ggml_gather_axis1(cctx, c2p_raw, c2p_idx, seq, n_heads);
    scores = ggml_add(cctx, scores, c2p);

    // p2c 
    ggml_tensor* pos_query = ggml_mul_mat(cctx, T.q_w, rel_slice);
    pos_query = ggml_add(cctx, pos_query,
                    ggml_repeat(cctx,
                        ggml_reshape_2d(cctx, T.q_b, T.q_b->ne[0], 1),
                        pos_query));
    pos_query = ggml_reshape_3d(cctx, pos_query, head_dim, n_heads, n_pos);
    pos_query = ggml_cont(cctx, ggml_permute(cctx, pos_query, 0, 2, 1, 3)); 
    pos_query = ggml_scale(cctx, pos_query, 1.0f / scale);

    ggml_tensor* p2c_raw = ggml_mul_mat(cctx, pos_query, K);

    ggml_tensor* p2c_idx = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, seq, seq);
    {
        int32_t* p = (int32_t*)p2c_idx->data;
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++) {
                int32_t raw_p2c = log_bucket_pos(-(i - j), att_span, max_pos);
                p[j + i*seq] = std::clamp(raw_p2c + att_span, 0, n_pos - 1);
            }
    }

    ggml_tensor* p2c = ggml_gather_axis1(cctx, p2c_raw, p2c_idx, seq, n_heads);
    p2c = ggml_cont(cctx, ggml_permute(cctx, p2c, 1, 0, 2, 3));

    scores = ggml_add(cctx, scores, p2c);


    scores = ggml_soft_max(cctx, scores);


    ggml_tensor* ctx_layer = ggml_mul_mat(cctx, V, scores);
    ctx_layer = ggml_cont(cctx, ggml_permute(cctx, ctx_layer, 0, 2, 1, 3)); 
    ctx_layer = ggml_reshape_2d(cctx, ctx_layer, hidden, seq);

    ggml_tensor* attn_out = ggml_mul_mat(cctx, T.out_w, ctx_layer);
    attn_out = ggml_add(cctx, attn_out,
                   ggml_repeat(cctx,
                       ggml_reshape_2d(cctx, T.out_b, T.out_b->ne[0], 1),
                       attn_out));

    attn_out = ggml_add(cctx, attn_out, x);
    attn_out = ggml_norm(cctx, attn_out, 1e-7f);
    attn_out = ggml_add(cctx, ggml_mul(cctx, attn_out, T.ln_w), T.ln_b);

    return attn_out; 
}

static ggml_tensor* deberta_build_ffn(
    ggml_context* cctx,
    ggml_tensor* x, // [hidden, seq]
    deberta_inter_ffn_tensors& T
) {
    ggml_tensor* inter = ggml_mul_mat(cctx, T.inter_w, x);
    inter = ggml_add(cctx, inter,
                 ggml_repeat(cctx,
                     ggml_reshape_2d(cctx, T.inter_b, T.inter_b->ne[0], 1),
                     inter));

    inter = ggml_unary(cctx, inter, GGML_UNARY_OP_GELU_ERF);

    ggml_tensor* out = ggml_mul_mat(cctx, T.out_w, inter);
    out = ggml_add(cctx, out,
                 ggml_repeat(cctx,
                     ggml_reshape_2d(cctx, T.out_b, T.out_b->ne[0], 1),
                     out));
    out = ggml_add(cctx, out, x);
    out = ggml_norm(cctx, out, 1e-7f);
    out = ggml_add(cctx, ggml_mul(cctx, out, T.ln_w), T.ln_b);    
    return out;
}

struct ggml_cgraph* deberta_build_graph(
    struct deberta_ctx* ctx,
    struct ggml_context* compute_ctx,
    const std::vector<int>& input_ids
) {
    int seq_len = input_ids.size();
    int n_heads = ctx->model.hparams.num_attention_heads;
    int head_dim = ctx->model.hparams.hidden_size / n_heads;
    int max_rel = ctx->model.hparams.max_relative_positions;
    if (max_rel < 1) {
        max_rel = ctx->model.hparams.position_buckets;
    }
    int max_pos = ctx->model.hparams.max_position_embeddings;
    ggml_tensor* x = deberta_build_embeddings(compute_ctx, ctx, input_ids);

    std::string layer_prefix = "encoder.layer.";
    ggml_tensor* rel_emb = ctx->model.tensors["encoder.rel_embeddings.weight"];
    rel_emb = ggml_norm(compute_ctx, rel_emb, 1e-7f);
    rel_emb = ggml_add(compute_ctx,
            ggml_mul(compute_ctx, rel_emb, ctx->model.tensors["encoder.LayerNorm.weight"]),
            ctx->model.tensors["encoder.LayerNorm.bias"]);

    int N = ctx->model.hparams.num_hidden_layers;
    for (int i = 0; i < N; i++) {
        deberta_attn_tensors attn_tensors = {
            .q_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.query_proj.weight"],
            .q_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.query_proj.bias"],
            .k_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.key_proj.weight"],
            .k_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.key_proj.bias"],
            .v_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.value_proj.weight"],
            .v_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.self.value_proj.bias"],
            .out_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.output.dense.weight"],
            .out_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.output.dense.bias"],
            .ln_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.output.LayerNorm.weight"],
            .ln_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".attention.output.LayerNorm.bias"],
        };

        x = deberta_build_attention(compute_ctx, x, attn_tensors, rel_emb, n_heads, head_dim, seq_len, max_rel, max_pos);

        deberta_inter_ffn_tensors inter_ffn_tensors = {
            .inter_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".intermediate.dense.weight"],
            .inter_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".intermediate.dense.bias"],
            .out_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".output.dense.weight"],
            .out_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".output.dense.bias"],
            .ln_w = ctx->model.tensors[layer_prefix + std::to_string(i) + ".output.LayerNorm.weight"],
            .ln_b = ctx->model.tensors[layer_prefix + std::to_string(i) + ".output.LayerNorm.bias"],
        };
        x = deberta_build_ffn(compute_ctx, x, inter_ffn_tensors);
    }

    struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(gf, x);
    return gf;
}
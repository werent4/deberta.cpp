#include "deberta.h"
#include "utils.h"
#include <algorithm>
// #include <cmath>
// #include <limits>
#include <cstdio>
#include <cstring>
#ifdef GGML_USE_CUDA
#include "ggml/include/ggml-cuda.h"
#endif

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
    fread(&hparams.pos_att_flags, sizeof(int), 1, f);

    if (hparams.max_relative_positions < 1) {
        hparams.max_relative_positions = hparams.position_buckets;
    }

    // printf("vocab_size = %d\n", hparams.vocab_size);
    // printf("max_position_embeddings = %d\n", hparams.max_position_embeddings);
    // printf("hidden_size = %d\n", hparams.hidden_size);
    // printf("intermediate_size = %d\n", hparams.intermediate_size);
    // printf("num_attention_heads = %d\n", hparams.num_attention_heads);
    // printf("num_hidden_layers = %d\n", hparams.num_hidden_layers);
    // printf("position_buckets = %d\n", hparams.position_buckets);
    // printf("max_relative_positions = %d\n", hparams.max_relative_positions);
    // printf("ftype = %d\n", hparams.ftype);
    // printf("embedding_size = %d\n", hparams.embedding_size);
    // printf("type_vocab_size = %d\n", hparams.type_vocab_size);
    // printf("position_biased_input = %d\n", hparams.position_biased_input);
    // printf("layer_norm_eps = %e\n", hparams.layer_norm_eps);
    // printf("pos_att_flags = %d (c2p=%d p2c=%d)\n",
    //     hparams.pos_att_flags,
    //     hparams_use_c2p(hparams),
    //     hparams_use_p2c(hparams));

    fseek(f, 0, SEEK_SET); // move file pointer to the beginning of file

    return true;
}

static bool deberta_backend_init(deberta_model& model, const deberta_device device) { // todo: select device with id
    switch (device)
    {
    case DEBERTA_DEVICE_CPU:
        model.backend = ggml_backend_cpu_init();
        break;
    case DEBERTA_DEVICE_CUDA:
        #ifdef GGML_USE_CUDA
        model.backend = ggml_backend_cuda_init(0);
        fprintf(stderr, "NOTE: some nodes will be allocated on CPU\n");
        #endif
    default:
        break;
    }

    if (model.backend == NULL) {
        return false;
    }
    return true;
}

static bool deberta_create_weights_tensors(FILE* f, struct deberta_model* model) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }
    auto& tensors = model->tensors;

    fseek(f, HYPER_MAGIC_SIZE * sizeof(int), SEEK_SET); // skip hparams + magic (14 integers)
    while (true) {
        int n_dims, name_len, ftype;
        if (fread(&n_dims, sizeof(int), 1, f) != 1) break;
        if (fread(&name_len, sizeof(int), 1, f) != 1) break;
        if (fread(&ftype, sizeof(int), 1, f) != 1) break;

        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            if (fread(&dims[i], sizeof(int), 1, f) != 1) break;
        }

        std::string layer_name(name_len, '\0');
        if (fread(layer_name.data(), sizeof(char), name_len, f) != (size_t)name_len) break;

        long num_elements = 1;
        for (int i = 0; i < n_dims; i++) num_elements *= dims[i];

        auto tensor_size = ggml_type_size((ggml_type)ftype) * num_elements;
        int64_t ne[4] = { dims[0], dims[1], dims[2], dims[3] };
        struct ggml_tensor* tensor = ggml_new_tensor(model->ctx, (ggml_type)ftype, n_dims, ne);
        if (!tensor) {
            fprintf(stderr, "failed to allocate tensor for layer '%s'\n", layer_name);
            return false;
        }

        tensors[layer_name] = tensor;
        fseek(f, tensor_size, SEEK_CUR);
    }
    fseek(f, 0, SEEK_SET); // reset file pointer to the beginning
    return true;
}

static bool deberta_load_weights_tensors(FILE* f, struct deberta_model* model) {
    if (!f) {
        fprintf(stderr, "failed to open file\n");
        return false;
    }
    auto& tensors = model->tensors;
    std::vector<char> read_buf;
    fseek(f, HYPER_MAGIC_SIZE * sizeof(int), SEEK_SET); // skip hparams + magic (14 integers)
    while (true) {
        int n_dims, name_len, ftype;
        if (fread(&n_dims, sizeof(int), 1, f) != 1) break;
        if (fread(&name_len, sizeof(int), 1, f) != 1) break;
        if (fread(&ftype, sizeof(int), 1, f) != 1) break;

        int32_t nelements = 1;
        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            if (fread(&dims[i], sizeof(int), 1, f) != 1) break;
            nelements *= dims[i];
        }

        std::string layer_name(name_len, '\0');
        if (fread(layer_name.data(), sizeof(char), name_len, f) != (size_t)name_len) break;

        if (tensors.find(layer_name) == tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, layer_name.c_str());
            return false;
        }
        auto tensor = tensors[layer_name];

        ggml_set_name(tensor, layer_name.c_str());
        if (ggml_nelements(tensor) != nelements) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, layer_name.c_str());
            return false;
        }

        if (tensor->ne[0] != dims[0] || tensor->ne[1] != dims[1] || tensor->ne[2] != dims[2] || tensor->ne[3] != dims[3]) {
            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                    __func__, layer_name.c_str(), 
                    (int) tensor->ne[0], (int) tensor->ne[1], tensor->ne[2], (int) tensor->ne[3], 
                    dims[0], dims[1], dims[2], dims[3]);
            return false;
        }

        const size_t bpe = ggml_type_size((ggml_type)ftype);
        if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, layer_name.c_str(), ggml_nbytes(tensor), nelements*bpe);
            return false;
        }


        if (ggml_backend_buffer_is_host(model->buffer_w)) {
            // for some backends such as CPU and Metal, the tensor data is in system memory and we can read directly into it
            fread(tensor->data, 1, ggml_nbytes(tensor), f);
        } else {
            // read into a temporary buffer first, then copy to device memory
            read_buf.resize(ggml_nbytes(tensor));
            fread(read_buf.data(), 1, ggml_nbytes(tensor), f);
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
        }
    }
    fseek(f, 0, SEEK_SET); // reset file pointer to the beginning
    return true;
}

struct deberta_ctx* deberta_load_from_file(const std::string & fname, const deberta_device device) {
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

    size_t n_tensors = 16 * model.hparams.num_hidden_layers + 6;
    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
        fprintf(stderr, "%s: failed to initialize ggml context for model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    if(!deberta_backend_init(model, device)) {
        fprintf(stderr, "%s: failed to init backend (model file '%s')\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;   
    }

    new_deberta_ctx->cpu_backend = ggml_backend_cpu_init();
    if (!new_deberta_ctx->cpu_backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    ggml_backend_t backends[] = {model.backend, new_deberta_ctx->cpu_backend};
    new_deberta_ctx->sched = ggml_backend_sched_new(backends, NULL, 2, DEBERTA_MAX_NODES, false, false);
    if (!new_deberta_ctx->sched) {
        fprintf(stderr, "%s: failed to init backend sched\n", __func__);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    if (!deberta_create_weights_tensors(f, &model)) {
        fprintf(stderr, "%s: failed to create tensors for weights from model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer_w) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", __func__);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    if (!deberta_load_weights_tensors(f, &model)) {
        fprintf(stderr, "%s: failed to create tensors for weights from model file '%s'\n", __func__, fname);
        deberta_free(new_deberta_ctx);
        return nullptr;
    }

    fclose(f);
    return new_deberta_ctx;
}

void deberta_free(deberta_ctx* ctx) {
    if (!ctx) return;
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    if (ctx->model.buffer_w) ggml_backend_buffer_free(ctx->model.buffer_w);
    if (ctx->model.backend) ggml_backend_free(ctx->model.backend);
    if (ctx->ctx_precomp) ggml_free(ctx->ctx_precomp);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->cpu_backend) ggml_backend_free(ctx->cpu_backend);
    delete ctx;
}

/// batch forward
static ggml_tensor* ggml_gather_batch_axis1(
    ggml_context* ctx,
    ggml_backend_sched_t sched,
    ggml_backend_t cpu_backend,
    ggml_tensor* src,
    ggml_tensor* idx_tensor,
    int seq, int n_heads, int batch_size
) {
    ggml_backend_sched_set_tensor_backend(sched, src, cpu_backend);
    ggml_tensor* dummy = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, seq, seq, n_heads, batch_size);
    ggml_backend_sched_set_tensor_backend(sched, dummy, cpu_backend);
    return ggml_map_custom2(ctx, dummy, src, gather_batch_custom_op, 1, (void*)idx_tensor->data);
}

static ggml_tensor* deberta_build_batch_embeddings(
    struct ggml_context* compute_ctx,
    const struct deberta_ctx* ctx,
    ggml_tensor* input_ids,
    int batch_size,
    int seq_len
) {    
    ggml_tensor* word_embeddings = ctx->model.tensors.at("embeddings.word_embeddings.weight");
    input_ids = ggml_reshape_1d(compute_ctx, input_ids, seq_len * batch_size);
    ggml_tensor* x = ggml_get_rows(compute_ctx, word_embeddings, input_ids);
    x = ggml_reshape_3d(compute_ctx, x, ctx->model.hparams.hidden_size, seq_len, batch_size);

    ggml_tensor* ln_w = ctx->model.tensors.at("embeddings.LayerNorm.weight");
    ggml_tensor* ln_b = ctx->model.tensors.at("embeddings.LayerNorm.bias");

    x = ggml_norm(compute_ctx, x, ctx->model.hparams.layer_norm_eps);
    x = ggml_mul(compute_ctx, x, ln_w);
    x = ggml_add(compute_ctx, x, ln_b);

    return x;
}

static ggml_tensor* deberta_build_batch_attention(
    ggml_context* cctx,
    const deberta_ctx* ctx,
    ggml_tensor* x, // [hidden, seq, batch]
    ggml_tensor* attn_masks, // [seq, batch]
    deberta_attn_tensors& T,
    ggml_tensor* rel_emb,  
    ggml_tensor* c2p_idx,
    ggml_tensor* p2c_idx,
    int n_heads,
    int head_dim,
    int seq,
    int max_rel,
    int max_pos
) {
    int batch_size = x->ne[2];
    const int hidden = n_heads * head_dim;
    const int scale_factor = 1 + (int)(hparams_use_c2p(ctx->model.hparams) == true) + (int)(hparams_use_p2c(ctx->model.hparams) == true);
    const float scale = sqrtf((float)(head_dim * scale_factor));
    // c2c
    // [hid_dim, seq, batch]
    ggml_tensor* Q = ggml_mul_mat(cctx, T.q_w, x);
    Q = ggml_add(cctx, Q, T.q_b);
    
    ggml_tensor* K = ggml_mul_mat(cctx, T.k_w, x);
    K = ggml_add(cctx, K, T.k_b);
    
    ggml_tensor* V = ggml_mul_mat(cctx, T.v_w, x);
    V = ggml_add(cctx, V, T.v_b);

    Q = ggml_reshape_4d(cctx, Q, head_dim, n_heads, seq, batch_size);
    K = ggml_reshape_4d(cctx, K, head_dim, n_heads, seq, batch_size);
    V = ggml_reshape_4d(cctx, V, head_dim, n_heads, seq, batch_size); // [head_dim, n_heads, seq, batch]

    Q = ggml_scale(cctx, Q, 1.0f / scale);

    Q = ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)); // [head_dim, seq, n_heads, batch]
    K = ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3));
    V = ggml_cont(cctx, ggml_permute(cctx, V, 1, 2, 0, 3)); // [n_heads, head_dim, seq, batch]

    ggml_tensor* scores = ggml_mul_mat(cctx, K, Q);

    const int att_span = max_rel;
    const int n_pos = 2 * att_span;
    
    size_t offset = (size_t)(max_rel - att_span) * rel_emb->nb[1];
    ggml_tensor* rel_slice = ggml_view_2d(cctx, rel_emb, rel_emb->ne[0], n_pos, rel_emb->nb[1], offset);
    
    // c2p
    if (hparams_use_c2p(ctx->model.hparams)) {
        ggml_tensor* pos_key = ggml_mul_mat(cctx, T.k_w, rel_slice);
        pos_key = ggml_add(cctx, pos_key,
                    ggml_repeat(cctx,
                        ggml_reshape_2d(cctx, T.k_b, T.k_b->ne[0], 1),
                        pos_key));

        pos_key = ggml_reshape_3d(cctx, pos_key, head_dim, n_heads, n_pos);
        pos_key = ggml_cont(cctx, ggml_permute(cctx, pos_key, 0, 2, 1, 3)); // [head_dim, n_pos, n_heads]

        ggml_tensor* c2p_raw = ggml_mul_mat(cctx, pos_key, Q);
        ggml_tensor* c2p = ggml_gather_batch_axis1(cctx, ctx->sched, ctx->cpu_backend, c2p_raw, c2p_idx, seq, n_heads, batch_size);
        scores = ggml_add(cctx, scores, c2p);
    }

    // p2c 
    if (hparams_use_p2c(ctx->model.hparams)) {
        ggml_tensor* pos_query = ggml_mul_mat(cctx, T.q_w, rel_slice);
        pos_query = ggml_add(cctx, pos_query,
                    ggml_repeat(cctx,
                        ggml_reshape_2d(cctx, T.q_b, T.q_b->ne[0], 1),
                        pos_query));

        pos_query = ggml_reshape_3d(cctx, pos_query, head_dim, n_heads, n_pos);
        pos_query = ggml_cont(cctx, ggml_permute(cctx, pos_query, 0, 2, 1, 3)); 
        pos_query = ggml_scale(cctx, pos_query, 1.0f / scale);

        ggml_tensor* p2c_raw = ggml_mul_mat(cctx, pos_query, K);
        ggml_tensor* p2c = ggml_gather_batch_axis1(cctx, ctx->sched, ctx->cpu_backend, p2c_raw, p2c_idx, seq, n_heads, batch_size);
        p2c = ggml_cont(cctx, ggml_permute(cctx, p2c, 1, 0, 2, 3));

        scores = ggml_add(cctx, scores, p2c); // [seq_q, seq_k, n_heads, batch]
    }

    attn_masks = ggml_reshape_4d(cctx, attn_masks, seq, 1, 1, batch_size);
    attn_masks = ggml_repeat(cctx, attn_masks, scores);

    scores = ggml_add(cctx, scores, attn_masks);
    scores = ggml_soft_max(cctx, scores);

    ggml_tensor* ctx_layer = ggml_mul_mat(cctx, V, scores);
    ctx_layer = ggml_cont(cctx, ggml_permute(cctx, ctx_layer, 0, 2, 1, 3)); // [head_dim, n_heads, seq, batch]
    ctx_layer = ggml_reshape_3d(cctx, ctx_layer, hidden, seq, batch_size);

    ggml_tensor* attn_out = ggml_mul_mat(cctx, T.out_w, ctx_layer);
    attn_out = ggml_add(cctx, attn_out,
                   ggml_repeat(cctx,
                       ggml_reshape_3d(cctx, T.out_b, T.out_b->ne[0], 1, 1),
                       attn_out));

    attn_out = ggml_add(cctx, attn_out, x);
    attn_out = ggml_norm(cctx, attn_out, ctx->model.hparams.layer_norm_eps);
    attn_out = ggml_add(cctx, ggml_mul(cctx, attn_out, T.ln_w), T.ln_b);

    return attn_out;
}

static ggml_tensor* deberta_build_batch_ffn(
    ggml_context* cctx,
    const deberta_ctx* ctx,
    ggml_tensor* x, // [hidden, seq, batchh]
    deberta_inter_ffn_tensors& T
) {
    ggml_tensor* inter = ggml_mul_mat(cctx, T.inter_w, x);
    inter = ggml_add(cctx, inter,
                 ggml_repeat(cctx,
                     ggml_reshape_3d(cctx, T.inter_b, T.inter_b->ne[0], 1, 1),
                     inter));

    inter = ggml_unary(cctx, inter, GGML_UNARY_OP_GELU_ERF);

    ggml_tensor* out = ggml_mul_mat(cctx, T.out_w, inter);
    out = ggml_add(cctx, out,
                 ggml_repeat(cctx,
                     ggml_reshape_3d(cctx, T.out_b, T.out_b->ne[0], 1, 1),
                     out));
    out = ggml_add(cctx, out, x);
    out = ggml_norm(cctx, out, ctx->model.hparams.layer_norm_eps);
    out = ggml_add(cctx, ggml_mul(cctx, out, T.ln_w), T.ln_b); 
    return out;
}

static struct ggml_cgraph* deberta_build_graph_batch(
    const deberta_ctx* ctx,
    int batch_size,
    int seq_len
) {
    static size_t buf_size = ggml_tensor_overhead()*DEBERTA_MAX_NODES + ggml_graph_overhead_custom(DEBERTA_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf.data(),
        .no_alloc   = true,
    };
    struct ggml_context* compute_ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, DEBERTA_MAX_NODES, false);

    int n_heads = ctx->model.hparams.num_attention_heads;
    int head_dim = ctx->model.hparams.hidden_size / n_heads;
    int max_rel = ctx->model.hparams.max_relative_positions;   
    int max_pos = ctx->model.hparams.max_position_embeddings;

    ggml_tensor* input_ids = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_I32, seq_len, batch_size); // todo: select type based on model.wtype
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor* attn_masks = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, seq_len, batch_size);
    ggml_set_name(attn_masks, "attention_masks");
    ggml_set_input(attn_masks);

    ggml_tensor* x = deberta_build_batch_embeddings(compute_ctx, ctx, input_ids, batch_size, seq_len);

    std::string layer_prefix = "encoder.layer.";
    ggml_tensor* rel_emb = ctx->model.tensors.at("encoder.rel_embeddings.weight");
    rel_emb = ggml_norm(compute_ctx, rel_emb, ctx->model.hparams.layer_norm_eps);
    rel_emb = ggml_add(compute_ctx,
            ggml_mul(compute_ctx, rel_emb, ctx->model.tensors.at("encoder.LayerNorm.weight")),
            ctx->model.tensors.at("encoder.LayerNorm.bias"));



    int N = ctx->model.hparams.num_hidden_layers;
    for (int i = 0; i < N; i++) {
        deberta_attn_tensors attn_tensors = {
            .q_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.query_proj.weight"),
            .q_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.query_proj.bias"),
            .k_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.key_proj.weight"),
            .k_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.key_proj.bias"),
            .v_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.value_proj.weight"),
            .v_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.self.value_proj.bias"),
            .out_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.output.dense.weight"),
            .out_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.output.dense.bias"),
            .ln_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.output.LayerNorm.weight"),
            .ln_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".attention.output.LayerNorm.bias"),
        };

        x = deberta_build_batch_attention(
            compute_ctx, 
            ctx,
            x, 
            attn_masks, 
            attn_tensors, 
            rel_emb, 
            ctx->c2p_idx, 
            ctx->p2c_idx,
            n_heads, 
            head_dim,
            seq_len,
            max_rel,
            max_pos
        );

        deberta_inter_ffn_tensors inter_ffn_tensors = {
            .inter_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".intermediate.dense.weight"),
            .inter_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".intermediate.dense.bias"),
            .out_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".output.dense.weight"),
            .out_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".output.dense.bias"),
            .ln_w = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".output.LayerNorm.weight"),
            .ln_b = ctx->model.tensors.at(layer_prefix + std::to_string(i) + ".output.LayerNorm.bias"),
        };
        x = deberta_build_batch_ffn(compute_ctx, ctx, x, inter_ffn_tensors);
    }

    ggml_build_forward_expand(gf, x);
    ggml_free(compute_ctx);
    return gf;
}

bool deberta_eval(
    deberta_ctx* ctx,
    const int n_threads,
    const std::vector<std::vector<int>> & input_ids,
    const std::vector<std::vector<int>> & attention_mask,
    std::vector<float> & output
) {

    int batch_size = input_ids.size();
    int seq_len = input_ids[0].size();

    if (ctx->cached_seq_len != seq_len) {
        if (ctx->ctx_precomp) {
            ggml_free(ctx->ctx_precomp);
            ctx->ctx_precomp = NULL;
        }

        const int att_span = ctx->model.hparams.max_relative_positions;
        const int n_pos = 2 * att_span;

        // c2p indexes
        size_t precomp_size = 2 * ggml_tensor_overhead() + 2 * seq_len * seq_len * sizeof(int) + 1024;
        struct ggml_init_params precomp_ctx_params = {
            .mem_size   = precomp_size,
            .mem_buffer = NULL, 
            .no_alloc   = false,
        };
        ctx->ctx_precomp = ggml_init(precomp_ctx_params);

        ggml_tensor* c2p_idx = ggml_new_tensor_2d(ctx->ctx_precomp, GGML_TYPE_I32, seq_len, seq_len);
        {
            int32_t* p = (int32_t*)c2p_idx->data;
            for (int i = 0; i < seq_len; i++)
                for (int j = 0; j < seq_len; j++) {
                    int32_t raw_c2p = log_bucket_pos(i - j, att_span, ctx->model.hparams.max_position_embeddings);
                    p[j + i*seq_len] = std::clamp(raw_c2p + att_span, 0, n_pos - 1);
                }
        }
        ctx->c2p_idx = c2p_idx;

        // p2c indexes
        ggml_tensor* p2c_idx = ggml_new_tensor_2d(ctx->ctx_precomp, GGML_TYPE_I32, seq_len, seq_len);
        {
            int32_t* p = (int32_t*)p2c_idx->data;
            for (int i = 0; i < seq_len; i++)
                for (int j = 0; j < seq_len; j++) {
                    int32_t raw_p2c = log_bucket_pos(-(i - j), att_span, ctx->model.hparams.max_position_embeddings);
                    p[j + i*seq_len] = std::clamp(raw_p2c + att_span, 0, n_pos - 1);
                }
        }
        ctx->p2c_idx = p2c_idx;

        // upd cacheed len
        ctx->cached_seq_len = seq_len;
    }

    struct ggml_cgraph* gf = deberta_build_graph_batch(ctx, batch_size, seq_len);
    ggml_backend_sched_alloc_graph(ctx->sched, gf);

    struct ggml_tensor* input_ids_tensor = ggml_graph_get_tensor(gf, "input_ids");
    for (size_t i = 0; i < batch_size; i++) {
        int offset = i * seq_len * sizeof(int);
        ggml_backend_tensor_set(input_ids_tensor, input_ids[i].data(), offset, seq_len * sizeof(int));
    }

    struct ggml_tensor* attention_mask_tensor = ggml_graph_get_tensor(gf, "attention_masks");
    std::vector<float> mask_row(seq_len);
    for (size_t i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask_row[j] = attention_mask[i][j] == 1 ? 0.0f : -INFINITY;
        }
        int offset = i * seq_len * sizeof(float);
        ggml_backend_tensor_set(attention_mask_tensor, mask_row.data(), offset, seq_len * sizeof(float));
    }
    
    if (ggml_backend_is_cpu(ctx->model.backend)) {
        ggml_backend_cpu_set_n_threads(ctx->model.backend, n_threads);
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);
    struct ggml_tensor* last_hidden_layer = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1); // [hidden, seq, batch];

    output.resize(batch_size * seq_len * ctx->model.hparams.hidden_size);
    ggml_backend_tensor_get(last_hidden_layer, output.data(), 0, ggml_nbytes(last_hidden_layer));
    return true;
}
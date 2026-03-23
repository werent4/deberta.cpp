#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include "deberta.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.bin \n", argv[0]);
        return 1;
    }

    std::vector<std::vector<int>> input_ids;
    std::vector<std::vector<int>> attention_mask;
    input_ids = {
        {279, 3185, 3208, 277, 262, 8358, 260, 0, 0, 0, 0, 0, 0},
        {7222, 1101, 1836, 1449, 614, 3909, 265, 838, 514, 264, 57304, 371, 260}
    };
    attention_mask = {
        {1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

    if (input_ids.size() != attention_mask.size()) {
        fprintf(stderr, "input_ids and attention_mask must have the same batch size\n");
        return 1;
    }
    if (input_ids.empty() || attention_mask.empty() || input_ids[0].empty() || attention_mask[0].empty()) {
        fprintf(stderr, "no input sequences provided\n");
        return 1;
    }
    if (input_ids[0].size() != input_ids[1].size()) {
        fprintf(stderr, "all input sequences must have the same length\n");
        return 1;
    }
    if (attention_mask[0].size() != attention_mask[1].size()) {
        fprintf(stderr, "all attention mask sequences must have the same length\n");
        return 1;
    }


    deberta_ctx* new_deberta_ctx = deberta_load_from_file(argv[1]);
    if (!new_deberta_ctx) {
        fprintf(stderr, "failed to load model from file '%s'\n", argv[1]);
        return 1;
    }

    struct ggml_init_params compute_params = {
        /*.mem_size   =*/ 2ull * 1024 * 1024 * 1024,  // 2GB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context* compute_ctx = ggml_init(compute_params);
    if (!compute_ctx) {
        fprintf(stderr, "failed to initialize ggml context for computation\n");
        deberta_free(new_deberta_ctx);
        return 1;
    }    

    deberta_batch_input batch_input = {
        .input_ids = input_ids,
        .attention_mask = attention_mask
    };
    struct ggml_cgraph* graph = deberta_build_graph_batch(new_deberta_ctx, compute_ctx, batch_input);
    if (!graph) {
        fprintf(stderr, "failed to build computation graph\n");
        deberta_free(new_deberta_ctx);
        ggml_free(compute_ctx);
        return 1;
    }
    ggml_graph_compute_with_ctx(compute_ctx, graph, 1);
    struct ggml_tensor* output = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1); // [hidden, seq, batch];
    float* data = (float*)output->data;
    int batch = output->ne[2];
    int seq = output->ne[1];
    int hidden = output->ne[0];

    // / FINAL OUTPUT !!!!!!!
    FILE* f = fopen("cpp_batch_out.txt", "w");
    for (size_t b = 0; b < batch_input.batch_size(); b++) {
        for (int t = 0; t < seq; t++) {
            for (int i = 0; i < hidden; i++) {
                fprintf(f, "%.6f", data[i + hidden * t + b*seq*hidden]);
                if (i < hidden - 1) fprintf(f, " ");
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
    printf("wrote %d tokens to cpp_batch_out.txt\n", batch_input.seq_len());

    // float* data = (float*)output->data;
    printf("first 10 values (batch 1): ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", data[1*seq*hidden + i]);
    }
    printf("\n");


}
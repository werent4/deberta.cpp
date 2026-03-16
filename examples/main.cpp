#include <cstdio>
#include <vector>
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include "deberta.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.bin\n", argv[0]);
        return 1;
    }
    std::vector<int> input_ids = {1, 31414, 232, 328, 2}; // [CLS] Hello world ! [SEP]

    deberta_ctx* new_deberta_ctx = deberta_load_from_file(argv[1]);
    if (!new_deberta_ctx) {
        fprintf(stderr, "failed to load model from file '%s'\n", argv[1]);
        return 1;
    }

    struct ggml_init_params compute_params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* compute_ctx = ggml_init(compute_params);
    if (!compute_ctx) {
        fprintf(stderr, "failed to initialize ggml context for computation\n");
        deberta_free(new_deberta_ctx);
        return 1;
    }

    struct ggml_cgraph* graph = deberta_build_graph(new_deberta_ctx, compute_ctx, input_ids);
    if (!graph) {
        fprintf(stderr, "failed to build computation graph\n");
        deberta_free(new_deberta_ctx);
        ggml_free(compute_ctx);
        return 1;
    }

    ggml_graph_compute_with_ctx(compute_ctx, graph, 1);
    struct ggml_tensor* output = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);

    printf("output shape: [%lld, %lld]\n", output->ne[0], output->ne[1]);
    float* data = (float*)output->data;
    printf("embeddings[0]: ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");

    deberta_free(new_deberta_ctx);
    ggml_free(compute_ctx);
    return 0;
}
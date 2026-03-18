#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include "deberta.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.bin [token1 token2 ...]\n", argv[0]);
        return 1;
    }

    std::vector<int> input_ids;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            input_ids.push_back(std::atoi(argv[i]));
        }
    } else {
        std::vector<int> input_ids = {
            7898,   267,  6865, 30256,   261, 51803,   261, 22272,  1196,   315,
            1206,   275, 24503, 13451,   416,  4994,   275,  5022,   780,   267,
            3037,   260,   383,  1181,   266,  2041,  1472,   288,   780,   261,
            399,   313,   722,   475,  6741,  5549,  2377,  4642,   261,   262,
            9026,  2377,   261,   263,   262, 13752,  2057,   964,  2832,   260,
            855,  2097,   958,  4425,   709,  3295,   417,   315,   362, 85903,
            1931,   280, 12868,   288,   905,  1600,   260,   344,  1812,   261,
            22272,  1181,   262,  1284,   265,   262,   393,   271,  8910,  2281,
            2225,   267,   820,   335,   313,  2280,  2858,  8202,   267,   266,
            907,  1231, 97073,   554,   705,   287, 23088,  4008,   705,   285,
            260,   620,  8202,   261,   313,   284,   288,   262, 12581,   265,
            262,  1788,   280,   268, 30229,   283,   266,  7623,  1559,  1711,
            261,  1789,   349,  1079,   654,  9026,  2377,   268,   457,  1151,
            263,   692,   261,   500,   262,   455,   271, 28621,  2025,   802,
            59574, 14477,   260,   383,   327,   722,   375,  2025, 26252,  4642,
            261,   500,   262,  1162,   271, 13680,  1513,   958,  1432,   709,
            267,   319,  8202,  2182,   803,   911,   261,   263,  1181,   262,
            1788,   280,   268,   305,   271,  1445,   530, 79172,   260,   383,
            722, 85903,  1931,   280, 12868,   268,   267,  1292,   261,  1151,
            261,   892,   263,   798,   261,   263,   284,  8334,   271,  1025,
            475,   631,   264, 26182, 23760,   261,   315,  7378,  1206,  7883,
            260,  4087,   808,   275,   262,  1788, 14565,   261, 22272,  2346,
            270, 26758,   267,   692,   267,   266,  2225,  1231,   299,  2262,
            73710,   705,   261,   399,   313,   284, 16255,   267,  2361,   375,
            30867,   336,  4642,   260,   344, 16789,   261,   313,  2286,   264,
            780,   416,  3917,  2513,   271, 17042,   268,   268,   834,   267,
            26323,   260
        };
    }

    int seq_len = input_ids.size();

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

    struct ggml_cgraph* graph = deberta_build_graph(new_deberta_ctx, compute_ctx, input_ids);
    if (!graph) {
        fprintf(stderr, "failed to build computation graph\n");
        deberta_free(new_deberta_ctx);
        ggml_free(compute_ctx);
        return 1;
    }

    ggml_graph_compute_with_ctx(compute_ctx, graph, 1);
    struct ggml_tensor* output = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);
    float* data = (float*)output->data;

    auto print_tensor = [&](const char* name) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            struct ggml_tensor* node = ggml_graph_node(graph, i);
            if (strcmp(ggml_get_name(node), name) == 0) {
                float* d = (float*)node->data;
                printf("%s: ", name);
                for (int j = 0; j < 8; j++) printf("%.4f ", d[j]);
                printf("\n");
                return;
            }
        }
        printf("%s: not found\n", name);
    };

    // / FINAL OUTPUT !!!!!!!
    FILE* f = fopen("cpp_out.txt", "w");
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < 768; i++) {
            fprintf(f, "%.6f", data[t * 768 + i]);
            if (i < 767) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("wrote %d tokens to cpp_out.txt\n", seq_len);

    printf("first 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", data[i]);
    }
    printf("\n");

    deberta_free(new_deberta_ctx);
    ggml_free(compute_ctx);
    return 0;
}
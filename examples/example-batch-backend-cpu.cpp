#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml/include/ggml.h"
#include "deberta.h"

int main(int argc, char ** argv) {
#ifdef GGML_USE_CUDA
    printf("btw, cuda is compiled!\n");
#endif

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
    ggml_gallocr_t allocr = NULL;
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_deberta_ctx->model.backend));
    }

    std::vector<float> output;

    deberta_eval(
        new_deberta_ctx,
        allocr,
        1,
        input_ids,
        attention_mask,
        output
    );

    float* data = output.data();
    int batch = input_ids.size();
    int seq = input_ids[0].size();
    int hidden = new_deberta_ctx->model.hparams.hidden_size;

    // / FINAL OUTPUT !!!!!!!
    FILE* f = fopen("cpp_batch_out.txt", "w");
    for (size_t b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            for (int i = 0; i < hidden; i++) {
                fprintf(f, "%.6f", data[i + hidden * t + b*seq*hidden]);
                if (i < hidden - 1) fprintf(f, " ");
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
    printf("wrote %d tokens to cpp_batch_out.txt\n", seq);

    // float* data = (float*)output->data;
    printf("first 10 values (batch 1): ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", data[1*seq*hidden + i]);
    }
    printf("\n");

    ggml_gallocr_free(allocr);
    deberta_free(new_deberta_ctx);
    return 0;
}
#include "../include/transformer_opencl.h"
#include <cstdlib>
#include <iostream>
#include <cmath>

int main() {
    int hidden = 32;
    int seq = 8;
    TransformerLayer layer(hidden, 4, 0.01f);

    Tensor x({seq, hidden});
    for (int i = 0; i < seq * hidden; ++i) x.h_data[i] = float((i % 7 - 3) * 0.123f);

    // Reference forward (normal)
#ifdef _WIN32
    _putenv_s("USE_AVX2_CPU", "");
#else
    unsetenv("USE_AVX2_CPU");
#endif
    Tensor out_ref = layer.Forward(x);

    // PoC forward (force AVX2 CPU)
#ifdef _WIN32
    _putenv_s("USE_AVX2_CPU", "1");
#else
    setenv("USE_AVX2_CPU", "1", 1);
#endif
    Tensor out_poc = layer.Forward(x);

    // compare
    float max_abs = 0.0f; float max_rel = 0.0f;
    for (size_t i = 0; i < out_ref.h_data.size(); ++i) {
        float a = out_ref.h_data[i];
        float b = out_poc.h_data[i];
        float absd = std::fabs(a - b);
        float rel = (std::fabs(a) > 1e-8f) ? absd / std::fabs(a) : absd;
        max_abs = std::max(max_abs, absd);
        max_rel = std::max(max_rel, rel);
    }
    std::cout << "PoC Transformer forward diff: max_abs=" << max_abs << " max_rel=" << max_rel << std::endl;
    if (max_rel > 1e-3f) {
        std::cerr << "Mismatch exceeds tolerance" << std::endl;
        return 1;
    }
    std::cout << "Transformer PoC forward check passed" << std::endl;
    return 0;
}

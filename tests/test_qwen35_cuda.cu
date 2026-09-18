/* CUDA parity test for Prism Bonsai (qwen35 / PQ2_0) on the CUDA backend.
 *
 * The oracle is ds4's own CPU reference code, reached through the
 * DS4_TEST_HOOKS entry points below (pq2_0_row_f32 and the double-precision
 * row dot in ds4.c), so nothing here re-implements the format:
 *
 *   1. Row dequant (token embeddings) must match the reference bit-exactly.
 *   2. The decode matvec (MMVQ) and the prefill tile (MMQ) must match the
 *      reference within a documented tolerance, at the model's real shapes.
 *   3. With activations that are exactly representable in the Q8_1 form the
 *      kernels use, the matmul must match the reference to float rounding -
 *      the check that catches a wrong weight-tile layout.
 *
 * Build: make test-qwen35-cuda */

#include "ds4_gpu.h"
#include "ds4_mmq.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <vector>

/* Defined in ds4.c under DS4_TEST_HOOKS (built as ds4_cuda_test_hooks.o). */
extern "C" int ds4_test_pq2_0_ref_row(const void *blocks, uint64_t row,
                                      uint64_t in_dim, float *out);
extern "C" int ds4_test_pq2_0_ref_matvec(const void *blocks, uint64_t out_dim,
                                         uint64_t in_dim, const float *x,
                                         float *out);

namespace {

constexpr int QK = 128;         // values per PQ2_0 block
constexpr int BLOCK_BYTES = 34; // fp16 scale + 32 code bytes

struct block_pq2_0_test {
    uint16_t d;
    uint8_t qs[QK / 4];
};

static_assert(sizeof(block_pq2_0_test) == BLOCK_BYTES, "unexpected PQ2_0 layout");

/* Model shapes: (out_dim, in_dim) pairs the Bonsai trunk actually matmuls.
 * kEmbd is the embedding lookup width (qwen35.embedding_length). */
constexpr int kEmbd = 5120;

struct shape { int M; int K; };

const shape kShapes[] = {
    {17408,  5120},  // ffn_gate / ffn_up
    { 5120, 17408},  // ffn_down
    { 5120,  6144},  // attn_output / lin_out
    { 6144,  5120},  // q/k/v projections (narrow)
};

uint32_t g_rng = 0x9e3779b9u;

static uint32_t next_rand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* Uniform in [-1, 1]. */
static float frand(void) {
    return ((float)(next_rand() & 0xffffffu) / 8388608.0f) - 1.0f;
}

static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Random blocks with realistic positive scales.  Codes span the full 0..3
 * alphabet so the +2d level and the negative level are both exercised. */
static void fill_blocks(std::vector<block_pq2_0_test> &blocks) {
    for (auto &b : blocks) {
        b.d = float_to_half(0.002f + 0.02f * std::fabs(frand()));
        for (uint8_t &byte : b.qs) byte = (uint8_t)(next_rand() & 0xffu);
    }
}

static bool cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool close_enough(const std::vector<float> &got,
                         const std::vector<float> &expected,
                         float abs_tol, float rel_tol, const char *label) {
    float worst = 0.0f;
    size_t worst_i = 0;
    int failures = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float limit = abs_tol + rel_tol * std::fabs(expected[i]);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
        if (!std::isfinite(got[i]) || diff > limit) failures++;
    }
    std::fprintf(stderr,
                 "%s: max_abs=%g (at %zu) failures=%d/%zu: %s\n",
                 label, worst, worst_i, failures, got.size(),
                 failures == 0 ? "PASS" : "FAIL");
    return failures == 0;
}

/* Relative L2 error over the whole output.  For the random-activation cases
 * this is the honest criterion: the kernels quantize the activation to the
 * Q8_1 form (one fp16 scale plus int8 codes per 32 values), so individual
 * outputs near zero carry a large *relative* error while the aggregate error
 * stays a small fraction of the signal.  A wrong weight-tile layout, by
 * contrast, puts ~100% of the energy into the error. */
static bool relative_l2_ok(const std::vector<float> &got,
                           const std::vector<float> &expected, float tol,
                           const char *label) {
    double err2 = 0.0, ref2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = (double)got[i] - (double)expected[i];
        err2 += d * d;
        ref2 += (double)expected[i] * (double)expected[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    const double rel = std::sqrt(err2) / std::sqrt(ref2 > 0.0 ? ref2 : 1.0);
    std::fprintf(stderr,
                 "%s: rel_l2=%g (tol %g) max_abs=%g rms_ref=%g: %s\n",
                 label, rel, tol, max_abs,
                 std::sqrt(ref2 / (double)(got.empty() ? 1 : got.size())),
                 rel <= tol ? "PASS" : "FAIL");
    return rel <= tol;
}

/* 1. Row lookup: the embedding path dequantizes rows; it must do so
 * bit-exactly, since it feeds the folded inverse transform. */
bool test_row_lookup() {
    constexpr int n_rows = 64;
    constexpr int in_dim = kEmbd;
    std::vector<block_pq2_0_test> blocks(
        (size_t)n_rows * (in_dim / QK));
    fill_blocks(blocks);

    void *d_blocks = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc blocks") ||
        !cuda_ok(cudaMalloc(&d_out, (size_t)in_dim * sizeof(float)),
                 "alloc row out")) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(d_blocks, blocks.data(),
                                 blocks.size() * sizeof(blocks[0]),
                                 cudaMemcpyHostToDevice, stream),
                 "copy blocks")) {
        return false;
    }

    std::vector<float> ref(in_dim), got(in_dim);
    bool ok = true;
    for (int row = 0; row < n_rows; row += 19) {
        if (ds4_test_pq2_0_ref_row(blocks.data(), (uint64_t)row, in_dim,
                                   ref.data()) != 0) {
            std::fprintf(stderr, "reference row %d failed\n", row);
            ok = false;
            break;
        }
        const int rc = ds4_mmq_pq2_0_rows_f32(d_out, d_blocks, nullptr,
                                              (uint64_t)row, 1u, in_dim,
                                              stream);
        if (!cuda_ok(cudaMemcpyAsync(got.data(), d_out,
                                     got.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy row out") ||
            !cuda_ok(cudaStreamSynchronize(stream), "sync row")) {
            ok = false;
            break;
        }
        if (rc != 0) {
            std::fprintf(stderr, "ds4_mmq_pq2_0_rows_f32 returned %d\n", rc);
            ok = false;
            break;
        }
        for (int i = 0; i < in_dim; i++) {
            if (got[i] != ref[i]) {
                std::fprintf(stderr,
                             "row %d element %d: got %g expected %g (bit-exact "
                             "match required)\n",
                             row, i, got[i], ref[i]);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    /* Multi-row (prefill) lookup driven by a device token array. */
    if (ok) {
        const int n_tok = 8;
        std::vector<int32_t> tokens = {3, 17, 0, 63, 42, 5, 60, 31};
        std::vector<float> multi_ref((size_t)n_tok * in_dim);
        for (int t = 0; t < n_tok; t++) {
            ds4_test_pq2_0_ref_row(blocks.data(), (uint64_t)tokens[t], in_dim,
                                   multi_ref.data() + (size_t)t * in_dim);
        }
        int32_t *d_tokens = nullptr;
        float *d_multi = nullptr;
        if (!cuda_ok(cudaMalloc(&d_tokens, tokens.size() * sizeof(int32_t)),
                     "alloc tokens") ||
            !cuda_ok(cudaMalloc(&d_multi, multi_ref.size() * sizeof(float)),
                     "alloc multi out") ||
            !cuda_ok(cudaMemcpyAsync(d_tokens, tokens.data(),
                                     tokens.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream),
                     "copy tokens")) {
            ok = false;
        } else {
            const int rc = ds4_mmq_pq2_0_rows_f32(d_multi, d_blocks, d_tokens,
                                                  0u, (uint32_t)n_tok,
                                                  (uint32_t)in_dim, stream);
            std::vector<float> multi_got(multi_ref.size());
            if (!cuda_ok(cudaMemcpyAsync(multi_got.data(), d_multi,
                                         multi_got.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy multi out") ||
                !cuda_ok(cudaStreamSynchronize(stream), "sync multi")) {
                ok = false;
            } else if (rc != 0) {
                std::fprintf(stderr, "multi-row lookup returned %d\n", rc);
                ok = false;
            } else {
                int bad = 0;
                for (size_t i = 0; i < multi_got.size(); i++) {
                    if (multi_got[i] != multi_ref[i]) bad++;
                }
                std::fprintf(stderr,
                             "PQ2_0 CUDA row lookup (%d tokens x %d): "
                             "mismatches=%d: %s\n",
                             n_tok, in_dim, bad, bad == 0 ? "PASS" : "FAIL");
                ok = ok && bad == 0;
            }
            cudaFree(d_multi);
            cudaFree(d_tokens);
        }
    }

    cudaFree(d_blocks);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* Runs one shape through the decode vec kernel and the prefill MMQ kernel,
 * comparing both against the ds4 CPU reference. */
bool run_shape(const shape &s, int n_tok, const char *label,
               bool exact_activations) {
    const int M = s.M;
    const int K = s.K;
    std::vector<block_pq2_0_test> blocks((size_t)M * (K / QK));
    fill_blocks(blocks);

    std::vector<float> x((size_t)n_tok * K);
    if (exact_activations) {
        /* Constant per 32-value block: Q8_1 quantization is then exact (the
         * block amax is its own value, so every element codes to +-127), which
         * leaves the weight-tile layout as the only thing under test. */
        for (int t = 0; t < n_tok; t++) {
            for (int k = 0; k < K; k += 32) {
                const float v = 0.05f + 0.5f * std::fabs(frand());
                for (int j = 0; j < 32; j++) x[(size_t)t * K + k + j] = v;
            }
        }
    } else {
        for (float &v : x) v = 0.5f * frand();
    }

    std::vector<float> ref((size_t)n_tok * M, 0.0f);
    for (int t = 0; t < n_tok; t++) {
        if (ds4_test_pq2_0_ref_matvec(blocks.data(), (uint64_t)M, (uint64_t)K,
                                      x.data() + (size_t)t * K,
                                      ref.data() + (size_t)t * M) != 0) {
            std::fprintf(stderr, "%s: reference matvec failed\n", label);
            return false;
        }
    }

    void *d_blocks = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = false;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc weights") ||
        !cuda_ok(cudaMalloc(&d_x, x.size() * sizeof(float)), "alloc x") ||
        !cuda_ok(cudaMalloc(&d_out, ref.size() * sizeof(float)), "alloc out")) {
        return false;
    }

    do {
        if (!cuda_ok(cudaMemcpyAsync(d_blocks, blocks.data(),
                                     blocks.size() * sizeof(blocks[0]),
                                     cudaMemcpyHostToDevice, stream),
                     "copy weights") ||
            !cuda_ok(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "copy x")) {
            break;
        }

        /* MMQ output is column-major per column: out[col*M + row]. */
        std::vector<float> mmq(ref.size(), 0.0f);
        const int rc_mmq = ds4_mmq_pq2_0_dense(d_blocks, d_x, d_out, M, n_tok,
                                               K, stream);
        if (!cuda_ok(cudaMemcpyAsync(mmq.data(), d_out,
                                     mmq.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy mmq out") ||
            !cuda_ok(cudaStreamSynchronize(stream), "sync mmq")) {
            break;
        }
        if (rc_mmq != 0) {
            std::fprintf(stderr, "%s: MMQ returned %d\n", label, rc_mmq);
            break;
        }
        char mmq_label[160];
        std::snprintf(mmq_label, sizeof(mmq_label), "%s MMQ (M=%d K=%d N=%d)",
                      label, M, K, n_tok);
        const bool mmq_ok = exact_activations
            ? close_enough(mmq, ref, 1e-5f, 1e-3f, mmq_label)
            : relative_l2_ok(mmq, ref, 0.05f, mmq_label);

        bool vec_ok = true;
        if (n_tok == 1) {
            std::vector<float> vec(M, 0.0f);
            std::vector<float> vec_ref(ref.begin(), ref.end());
            const int rc_vec = ds4_mmq_pq2_0_dense_vec(d_blocks, d_x, d_out,
                                                       M, 1, K, stream);
            if (!cuda_ok(cudaMemcpyAsync(vec.data(), d_out,
                                         vec.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy vec out") ||
                !cuda_ok(cudaStreamSynchronize(stream), "sync vec")) {
                break;
            }
            if (rc_vec != 0) {
                std::fprintf(stderr, "%s: MMVQ returned %d\n", label, rc_vec);
                break;
            }
            char vec_label[160];
            std::snprintf(vec_label, sizeof(vec_label), "%s MMVQ (M=%d K=%d)",
                          label, M, K);
            vec_ok = exact_activations
                ? close_enough(vec, vec_ref, 1e-5f, 1e-3f, vec_label)
                : relative_l2_ok(vec, vec_ref, 0.05f, vec_label);
        }
        ok = mmq_ok && vec_ok;
    } while (false);

    cudaFree(d_blocks);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* 2. Guards: shapes the kernels cannot serve must be refused, not misread. */
bool test_shape_guards() {
    std::vector<block_pq2_0_test> blocks(64 * (512 / QK));
    fill_blocks(blocks);
    std::vector<float> x(512, 0.1f);
    std::vector<float> out(64, 0.0f);

    void *d_blocks = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc weights") ||
        !cuda_ok(cudaMalloc(&d_x, x.size() * sizeof(float)), "alloc x") ||
        !cuda_ok(cudaMalloc(&d_out, out.size() * sizeof(float)),
                 "alloc out")) {
        return false;
    }

    const int bad_k = ds4_mmq_pq2_0_dense(d_blocks, d_x, d_out, 64, 1, 128,
                                          stream);
    const int bad_rows = ds4_mmq_pq2_0_rows_f32(d_out, d_blocks, nullptr, 0u,
                                                1u, 100u, stream);
    const bool ok = bad_k != 0 && bad_rows != 0;
    std::fprintf(stderr, "PQ2_0 shape guards (K%%256, in_dim%%128): %s\n",
                 ok ? "PASS" : "FAIL");

    cudaFree(d_blocks);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* 3. Host wiring: the entries the qwen35 CUDA graph will call
 * (ds4_gpu_embed_token(s)_quant_tensor and ds4_gpu_matmul_quant_tensor) must
 * resolve a PQ2_0 weight out of the model map and dispatch to the kernels
 * above.  This drives the real host path, not the raw kernel entries. */
bool test_host_wiring() {
    constexpr int n_vocab = 8;
    constexpr int in_dim = kEmbd;
    constexpr uint64_t arena_bytes = (uint64_t)64 << 20;

    std::vector<block_pq2_0_test> embd((size_t)n_vocab * (in_dim / QK));
    fill_blocks(embd);

    void *arena = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        std::fprintf(stderr, "host wiring: mmap failed\n");
        return false;
    }
    std::memcpy(arena, embd.data(), embd.size() * sizeof(embd[0]));

    bool ok = true;
    if (ds4_gpu_init() != 1 || ds4_gpu_set_model_map(arena, arena_bytes) != 1) {
        std::fprintf(stderr, "host wiring: backend/model-map setup failed\n");
        munmap(arena, arena_bytes);
        return false;
    }

    /* Single-token embedding lookup, bit-exact against the reference. */
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    std::vector<float> ref(in_dim), got(in_dim);
    if (!out) {
        std::fprintf(stderr, "host wiring: tensor alloc failed\n");
        ok = false;
    }
    for (int token = 0; ok && token < n_vocab; token += 3) {
        ds4_test_pq2_0_ref_row(embd.data(), (uint64_t)token, in_dim, ref.data());
        if (!ds4_gpu_embed_token_quant_tensor(out, arena, arena_bytes, 0,
                                              142u, n_vocab, (uint32_t)token,
                                              in_dim) ||
            !ds4_gpu_tensor_read(out, 0, got.data(),
                                 got.size() * sizeof(float))) {
            std::fprintf(stderr, "host wiring: embed_token(%d) failed\n", token);
            ok = false;
            break;
        }
        for (int i = 0; i < in_dim; i++) {
            if (got[i] != ref[i]) {
                std::fprintf(stderr,
                             "host wiring: embed_token(%d) element %d got %g "
                             "expected %g\n",
                             token, i, got[i], ref[i]);
                ok = false;
                break;
            }
        }
    }
    std::fprintf(stderr, "PQ2_0 host embed_token (bit-exact): %s\n",
                 ok ? "PASS" : "FAIL");

    /* Multi-token embedding lookup through the device token array. */
    if (ok) {
        const int n_tok = 4;
        std::vector<int32_t> tokens = {7, 0, 5, 2};
        ds4_gpu_tensor *tok = ds4_gpu_tensor_alloc(
            (uint64_t)n_tok * sizeof(int32_t));
        ds4_gpu_tensor *multi = ds4_gpu_tensor_alloc(
            (uint64_t)n_tok * in_dim * sizeof(float));
        std::vector<float> multi_ref((size_t)n_tok * in_dim), multi_got(multi_ref.size());
        for (int t = 0; t < n_tok; t++) {
            ds4_test_pq2_0_ref_row(embd.data(), (uint64_t)tokens[t], in_dim,
                                   multi_ref.data() + (size_t)t * in_dim);
        }
        const bool wrote = tok && multi &&
            ds4_gpu_tensor_write(tok, 0, tokens.data(),
                                 tokens.size() * sizeof(int32_t)) &&
            ds4_gpu_embed_tokens_quant_tensor(multi, tok, arena, arena_bytes, 0,
                                              142u, n_vocab, (uint32_t)n_tok,
                                              in_dim) &&
            ds4_gpu_tensor_read(multi, 0, multi_got.data(),
                                multi_got.size() * sizeof(float));
        int bad = wrote ? 0 : 1;
        if (wrote) {
            for (size_t i = 0; i < multi_got.size(); i++) {
                if (multi_got[i] != multi_ref[i]) bad++;
            }
        }
        std::fprintf(stderr,
                     "PQ2_0 host embed_tokens (%d tokens, bit-exact): %s\n",
                     n_tok, bad == 0 ? "PASS" : "FAIL");
        ok = ok && bad == 0;
        ds4_gpu_tensor_free(multi);
        ds4_gpu_tensor_free(tok);
    }
    ds4_gpu_tensor_free(out);

    /* Dense matmul through the graph entry, decode and prefill batch. */
    const shape s = {5120, 6144};
    const uint64_t w_bytes = (uint64_t)s.M * (s.K / QK) * sizeof(block_pq2_0_test);
    if (ok && w_bytes > arena_bytes) {
        std::fprintf(stderr, "host wiring: arena too small for matmul\n");
        ok = false;
    }
    if (ok) {
        std::vector<block_pq2_0_test> w(s.M * (s.K / QK));
        fill_blocks(w);
        std::memcpy(arena, w.data(), w.size() * sizeof(w[0]));

        for (const int n_tok : {1, 4}) {
            std::vector<float> x((size_t)n_tok * s.K);
            for (float &v : x) v = 0.5f * frand();
            std::vector<float> ref_mm((size_t)n_tok * s.M, 0.0f);
            for (int t = 0; t < n_tok; t++) {
                ds4_test_pq2_0_ref_matvec(w.data(), (uint64_t)s.M, (uint64_t)s.K,
                                          x.data() + (size_t)t * s.K,
                                          ref_mm.data() + (size_t)t * s.M);
            }
            ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(
                (uint64_t)n_tok * s.K * sizeof(float));
            ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(
                (uint64_t)n_tok * s.M * sizeof(float));
            std::vector<float> got_mm(ref_mm.size());
            const bool ran = gx && go &&
                ds4_gpu_tensor_write(gx, 0, x.data(), x.size() * sizeof(float)) &&
                ds4_gpu_matmul_quant_tensor(go, arena, arena_bytes, 0, 142u,
                                            (uint64_t)s.K, (uint64_t)s.M, gx,
                                            (uint64_t)n_tok) &&
                ds4_gpu_tensor_read(go, 0, got_mm.data(),
                                    got_mm.size() * sizeof(float));
            char label[128];
            std::snprintf(label, sizeof(label),
                          "host matmul_quant M=%d K=%d N=%d", s.M, s.K, n_tok);
            const bool this_ok = ran && relative_l2_ok(got_mm, ref_mm, 0.05f, label);
            if (!ran) {
                std::fprintf(stderr, "%s: host dispatch failed\n", label);
            }
            ok = ok && this_ok;
            ds4_gpu_tensor_free(go);
            ds4_gpu_tensor_free(gx);
        }
    }

    munmap(arena, arena_bytes);
    return ok;
}

} // namespace

int main() {
    if (ds4_mmq_init(0) != 0) {
        std::fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }

    const bool rows_ok = test_row_lookup();
    const bool guards_ok = test_shape_guards();
    const bool host_ok = test_host_wiring();

    /* Random activations: the kernels quantize the activation to the Q8_1
     * form, so the outputs are compared by relative L2 error against the
     * double-precision reference (criterion documented above). */
    bool shapes_ok = true;
    for (const shape &s : kShapes) {
        shapes_ok = run_shape(s, 1, "random1", false) && shapes_ok;
    }

    /* Prefill batches: one column tile boundary (8), a MMQ_DP4A-sized batch
     * (64) and a 256-column batch. */
    shapes_ok = run_shape(kShapes[0], 8, "prefill8", false) && shapes_ok;
    shapes_ok = run_shape(kShapes[1], 64, "prefill64", false) && shapes_ok;
    shapes_ok = run_shape(kShapes[2], 256, "prefill256", false) && shapes_ok;

    /* Exact-activation pass: near-float-rounding agreement is required, which
     * is what makes a wrong tile index visible. */
    bool exact_ok = true;
    for (const shape &s : kShapes) {
        exact_ok = run_shape(s, 1, "exact1", true) && exact_ok;
    }
    exact_ok = run_shape(kShapes[0], 64, "exact64", true) && exact_ok;

    const bool ok = rows_ok && guards_ok && host_ok && shapes_ok && exact_ok;
    std::fprintf(stderr, "PQ2_0 CUDA parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* Numeric verification for the Laguna CRACK mixed-quant CUDA kernels.
 *
 * The community CRACK Laguna exports use a hybrid layout the branch now
 * accepts on CUDA: Q4_K or Q6_K embedding and dense experts, Q8_0 attention
 * and shared experts, routed gate/up in Q4_K or Q6_K, routed down in Q4_K,
 * Q6_K or mixed, and a Q6_K output head. This test verifies the new kernels
 * against CPU references built from the same block formulas ds4.c uses:
 *
 *   1. ds4_gpu_matmul_quant_tensor / ds4_gpu_matmul_q6_K_tensor for Q4_K
 *      and Q6_K weight matrices at single-row (kernel path) and batched
 *      (dequant+GEMM path) token counts.
 *   2. ds4_gpu_embed_token_quant_tensor for Q4_K and Q6_K embedding tables.
 *   3. ds4_gpu_glm_routed_moe_batch_tensor for routed gate/up Q4_K with
 *      Q6_K down (the CRACK Q4_K_M layout) and all-Q6_K (the Q6_K export),
 *      including the q8_K activation quantization the kernels perform.
 *
 * Requires a CUDA device. Run with:
 *   make test-laguna-crack-kernels
 */

#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

static int g_failures;

#define CHECK(cond, fmt, ...)                                            \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__);           \
            g_failures++;                                                \
        } else {                                                         \
            printf("  ok: " fmt "\n", ##__VA_ARGS__);                    \
        }                                                                \
    } while (0)

/* ---------- format helpers (mirror ds4.c / dev_* kernels) ---------- */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void q4_k_get_scale_min(int j, const uint8_t *q,
                               uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63u;
        *m  = q[j + 4] & 63u;
    } else {
        *sc = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)    | ((q[j - 0] >> 6) << 4);
    }
}

/* Flat element k inside the row, matching dev_q4_K_value. */
static float q4_K_value(const block_q4_K *blocks, uint32_t k) {
    const block_q4_K *block = blocks + k / QK_K;
    const uint32_t idx = k & (QK_K - 1u);
    const uint32_t group = idx >> 5u;
    const uint32_t lane = idx & 31u;
    uint8_t sc = 0, m = 0;
    q4_k_get_scale_min((int)group, block->scales, &sc, &m);
    const uint32_t byte_off = (group >> 1u) * 32u + lane;
    const uint32_t shift = (group & 1u) * 4u;
    const uint32_t q = (block->qs[byte_off] >> shift) & 0x0Fu;
    return f16_to_f32(block->d) * (float)sc * (float)q -
           f16_to_f32(block->dmin) * (float)m;
}

/* Flat element k inside the row, matching dev_q6_K_value. */
static float q6_K_value(const block_q6_K *blocks, uint32_t k) {
    const block_q6_K *block = blocks + k / QK_K;
    const uint32_t idx = k & (QK_K - 1u);
    const uint32_t half = idx >> 7u;
    const uint32_t within = idx & 127u;
    const uint32_t lane = within & 31u;
    const uint32_t quarter = within >> 5u;
    const uint32_t ql_base = half * 64u;
    const uint32_t qh_base = half * 32u;
    const uint32_t scale_base = half * 8u;
    uint32_t q = 0;
    int32_t scale = 0;
    if (quarter == 0u) {
        q = (block->ql[ql_base + lane] & 0x0Fu) |
            (((block->qh[qh_base + lane] >> 0u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 0u];
    } else if (quarter == 1u) {
        q = (block->ql[ql_base + 32u + lane] & 0x0Fu) |
            (((block->qh[qh_base + lane] >> 2u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 2u];
    } else if (quarter == 2u) {
        q = (block->ql[ql_base + lane] >> 4u) |
            (((block->qh[qh_base + lane] >> 4u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 4u];
    } else {
        q = (block->ql[ql_base + 32u + lane] >> 4u) |
            (((block->qh[qh_base + lane] >> 6u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 6u];
    }
    return f16_to_f32(block->d) * (float)scale * (float)((int32_t)q - 32);
}

/* CPU mirror of q8_K_quantize_kernel: first-occurrence max-abs scan,
 * iscale = -127/maxv, d = 1/iscale, bsums per 16. */
static void q8_K_quantize(block_q8_K *yb, const float *x) {
    float max_abs = -1.0f;
    float maxv = 0.0f;
    for (int i = 0; i < QK_K; i++) {
        const float a = fabsf(x[i]);
        if (a > max_abs) { max_abs = a; maxv = x[i]; }
    }
    if (max_abs == 0.0f) {
        yb->d = 0.0f;
        memset(yb->qs, 0, sizeof(yb->qs));
        memset(yb->bsums, 0, sizeof(yb->bsums));
        return;
    }
    const float iscale = -127.0f / maxv;
    for (int i = 0; i < QK_K; i++) {
        int qv = (int)lrintf(iscale * x[i]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[i] = (int8_t)qv;
    }
    for (int j = 0; j < QK_K / 16; j++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[j * 16 + i];
        yb->bsums[j] = (int16_t)sum;
    }
    yb->d = 1.0f / iscale;
}

static float silu_f32(float v) {
    return v / (1.0f + expf(-v));
}

/* ---------- deterministic data ---------- */

static uint32_t g_lcg = 0x12345678u;

static uint32_t lcg(void) {
    g_lcg = g_lcg * 1103515245u + 12345u;
    return g_lcg >> 8;
}

static void fill_random(void *dst, size_t n) {
    uint8_t *p = dst;
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)lcg();
}

static void random_q4_K_rows(block_q4_K *blocks, uint32_t rows,
                             uint32_t blocks_per_row) {
    fill_random(blocks, (size_t)rows * blocks_per_row * sizeof(*blocks));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q4_K *blk = blocks + (size_t)r * blocks_per_row + b;
            blk->d    = 0x3C00u; /* 1.0 */
            blk->dmin = 0x3800u; /* 0.5 */
        }
    }
}

static void random_q6_K_rows(block_q6_K *blocks, uint32_t rows,
                             uint32_t blocks_per_row) {
    fill_random(blocks, (size_t)rows * blocks_per_row * sizeof(*blocks));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q6_K *blk = blocks + (size_t)r * blocks_per_row + b;
            blk->d = 0x3C00u; /* 1.0 */
        }
    }
}

static void random_f32(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = ((float)(lcg() % 20000) / 10000.0f) - 1.0f;
    }
}

/* ---------- synthetic model mapping ---------- */

/* Page-aligned host buffer registered once as the "model map"; weights are
 * placed at 4096-byte offsets so the CUDA range resolver maps them. */
static uint8_t *g_model;
static uint64_t g_model_size;

static void model_map_init(uint64_t size) {
    const uint64_t padded = size + (1u << 20);
    void *p = mmap(NULL, padded, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "FAIL: mmap for synthetic model\n");
        exit(1);
    }
    memset(p, 0, padded);
    g_model = (uint8_t *)p;
    g_model_size = padded;
    if (!ds4_gpu_set_model_map(g_model, g_model_size)) {
        fprintf(stderr, "FAIL: ds4_gpu_set_model_map\n");
        exit(1);
    }
}

/* ---------- test 1: dense matmul ---------- */

static float matmul_error(const void *weights, uint32_t type,
                          uint32_t in_dim, uint32_t out_dim,
                          const float *x, uint32_t n_tokens,
                          const float *got) {
    const uint32_t blocks_per_row = in_dim / QK_K;
    float worst = 0.0f;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            double ref = 0.0, norm = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) {
                const float wv = (type == 12u)
                    ? q4_K_value((const block_q4_K *)weights +
                                     (size_t)r * blocks_per_row, i)
                    : q6_K_value((const block_q6_K *)weights +
                                     (size_t)r * blocks_per_row, i);
                const float prod = (double)wv * x[(size_t)t * in_dim + i];
                ref += prod;
                norm += fabsf(prod);
            }
            const float gotv = got[(size_t)t * out_dim + r];
            const float err = fabsf((float)ref - gotv);
            const float lim = 0.005f * fabsf((float)ref) + 0.002f * norm;
            if (err > lim && err > worst) worst = err / (lim > 1e-6f ? lim : 1e-6f);
        }
    }
    return worst;
}

static void test_dense_matmul(uint32_t type) {
    const uint32_t in_dim = 512, out_dim = 6;
    const uint32_t blocks_per_row = in_dim / QK_K;
    const uint64_t block_bytes = (type == 12u) ? sizeof(block_q4_K)
                                               : sizeof(block_q6_K);
    const uint64_t weight_bytes =
        (uint64_t)out_dim * blocks_per_row * block_bytes;
    const uint64_t weight_off = g_model_size - (1u << 20) - weight_bytes;
    void *weights = g_model + weight_off;

    if (type == 12u) {
        random_q4_K_rows((block_q4_K *)weights, out_dim, blocks_per_row);
    } else {
        random_q6_K_rows((block_q6_K *)weights, out_dim, blocks_per_row);
    }

    for (uint32_t n_tokens = 1; n_tokens <= 4; n_tokens += 3) {
        float *host_x = malloc((size_t)n_tokens * in_dim * sizeof(float));
        float *host_out = malloc((size_t)n_tokens * out_dim * sizeof(float));
        random_f32(host_x, (size_t)n_tokens * in_dim);

        ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * out_dim * sizeof(float));
        if (!xt || !ot) { fprintf(stderr, "FAIL: tensor alloc\n"); exit(1); }
        ds4_gpu_tensor_write(xt, 0, host_x,
                             (uint64_t)n_tokens * in_dim * sizeof(float));

        int rc = (type == 12u)
            ? ds4_gpu_matmul_quant_tensor(ot, g_model, g_model_size,
                                          weight_off, 12u,
                                          in_dim, out_dim, xt, n_tokens)
            : ds4_gpu_matmul_q6_K_tensor(ot, g_model, g_model_size,
                                         weight_off,
                                         in_dim, out_dim, xt, n_tokens);
        CHECK(rc != 0, "%s matmul rc (n_tokens=%u)", type == 12u ? "Q4_K" : "Q6_K", n_tokens);
        if (rc) {
            ds4_gpu_tensor_read(ot, 0, host_out,
                                (uint64_t)n_tokens * out_dim * sizeof(float));
            const float rel = matmul_error(weights, type, in_dim, out_dim,
                                           host_x, n_tokens, host_out);
            CHECK(rel <= 1.0f, "%s matmul vs CPU ref (n_tokens=%u, worst=%.3f of limit)",
                  type == 12u ? "Q4_K" : "Q6_K", n_tokens, rel);
        }
        ds4_gpu_tensor_free(ot);
        ds4_gpu_tensor_free(xt);
        free(host_out);
        free(host_x);
    }
}

/* ---------- test 2: embedding gather ---------- */

static void test_embed(uint32_t type) {
    const uint32_t n_embd = 512, n_vocab = 4;
    const uint32_t blocks_per_row = n_embd / QK_K;
    const uint64_t block_bytes = (type == 12u) ? sizeof(block_q4_K)
                                               : sizeof(block_q6_K);
    const uint64_t table_bytes =
        (uint64_t)n_vocab * blocks_per_row * block_bytes;
    const uint64_t table_off = g_model_size - (1u << 20) - table_bytes -
                               65536u;
    void *table = g_model + table_off;

    if (type == 12u) {
        random_q4_K_rows((block_q4_K *)table, n_vocab, blocks_per_row);
    } else {
        random_q6_K_rows((block_q6_K *)table, n_vocab, blocks_per_row);
    }

    const uint32_t token = 2;
    float host_out[512];
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    const int rc = ds4_gpu_embed_token_quant_tensor(
            ot, g_model, g_model_size, table_off, type,
            n_vocab, token, n_embd);
    CHECK(rc != 0, "%s embed rc", type == 12u ? "Q4_K" : "Q6_K");
    if (rc) {
        ds4_gpu_tensor_read(ot, 0, host_out, n_embd * sizeof(float));
        float worst = 0.0f;
        for (uint32_t i = 0; i < n_embd; i++) {
            const float ref = (type == 12u)
                ? q4_K_value((const block_q4_K *)table +
                                 (size_t)token * blocks_per_row, i)
                : q6_K_value((const block_q6_K *)table +
                                 (size_t)token * blocks_per_row, i);
            const float err = fabsf(ref - host_out[i]);
            if (err > worst) worst = err;
        }
        CHECK(worst < 1e-4f, "%s embed vs CPU dequant (worst abs=%.2e)",
              type == 12u ? "Q4_K" : "Q6_K", worst);
    }
    ds4_gpu_tensor_free(ot);
}

/* ---------- test 3: routed MoE (mixed and all-Q6_K) ---------- */

struct moe_case {
    const char *name;
    uint32_t gate_type;  /* 12 = Q4_K, 14 = Q6_K */
    uint32_t down_type;  /* 12 or 14 */
};

static void test_moe(struct moe_case c) {
    const uint32_t in_dim = 512, mid_dim = 256, out_dim = 512;
    const uint32_t n_total = 2, n_used = 2;
    const uint32_t gb_in = in_dim / QK_K;
    const uint32_t gb_mid = mid_dim / QK_K;
    const uint64_t gu_block = (c.gate_type == 12u) ? sizeof(block_q4_K)
                                                   : sizeof(block_q6_K);
    const uint64_t dn_block = (c.down_type == 12u) ? sizeof(block_q4_K)
                                                   : sizeof(block_q6_K);
    const uint64_t gu_row = (uint64_t)gb_in * gu_block;
    const uint64_t dn_row = (uint64_t)gb_mid * dn_block;
    const uint64_t gu_expert = (uint64_t)mid_dim * gu_row;
    const uint64_t dn_expert = (uint64_t)out_dim * dn_row;
    const uint64_t gu_bytes = (uint64_t)n_total * gu_expert;
    const uint64_t dn_bytes = (uint64_t)n_total * dn_expert;

    /* Lay the three expert tensors out in disjoint regions below the same
     * 1 MiB pad used by the other tests; carving sequentially downward
     * guarantees no overlap. */
    uint64_t cursor = g_model_size - (1u << 20);
    cursor -= gu_bytes;
    const uint64_t gate_off = cursor;
    cursor -= gu_bytes;
    const uint64_t up_off = cursor;
    cursor -= dn_bytes;
    const uint64_t down_off = cursor;

    if (c.gate_type == 12u) {
        random_q4_K_rows((block_q4_K *)(g_model + gate_off),
                         n_total * mid_dim, gb_in);
        random_q4_K_rows((block_q4_K *)(g_model + up_off),
                         n_total * mid_dim, gb_in);
    } else {
        random_q6_K_rows((block_q6_K *)(g_model + gate_off),
                         n_total * mid_dim, gb_in);
        random_q6_K_rows((block_q6_K *)(g_model + up_off),
                         n_total * mid_dim, gb_in);
    }
    if (c.down_type == 12u) {
        random_q4_K_rows((block_q4_K *)(g_model + down_off),
                         n_total * out_dim, gb_mid);
    } else {
        random_q6_K_rows((block_q6_K *)(g_model + down_off),
                         n_total * out_dim, gb_mid);
    }

    /* One token routed to both experts, second listed first so the order
     * sensitivity of the accumulation is exercised. */
    const int32_t selected_host[2] = { 1, 0 };
    const float weight_host[2] = { 0.6f, 0.4f };
    float x_host[512];
    random_f32(x_host, in_dim);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(2 * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(
            (uint64_t)n_used * mid_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor_write(xt, 0, x_host, in_dim * sizeof(float));
    ds4_gpu_tensor_write(sel, 0, selected_host, sizeof(selected_host));
    ds4_gpu_tensor_write(wt, 0, weight_host, sizeof(weight_host));

    const int rc = ds4_gpu_glm_routed_moe_batch_tensor(
            out, mid, g_model, g_model_size,
            gate_off, up_off, down_off,
            c.gate_type, c.gate_type, c.down_type,
            gu_expert, gu_row, gu_expert, gu_row,
            dn_expert, dn_row,
            in_dim, mid_dim, out_dim,
            sel, wt, n_total, n_used, 0, xt, 1,
            n_used * mid_dim, true);
    CHECK(rc != 0, "%s MoE rc", c.name);
    if (rc) {
        float got[512];
        ds4_gpu_tensor_read(out, 0, got, out_dim * sizeof(float));

        /* CPU reference: q8_K-quantize each block of x, per-expert per-row
         * gate/up dots, silu*up*router weight, then q8_K-quantize the mid
         * row and run the down projection. All block math mirrors the
         * dev_* kernels and ds4.c's CPU reference formulas. */
        block_q8_K xq[8];
        for (uint32_t b = 0; b < gb_in; b++) {
            q8_K_quantize(&xq[b], x_host + (size_t)b * QK_K);
        }

        double worst_rel = 0.0;
        for (uint32_t r = 0; r < out_dim; r++) {
            double ref = 0.0, norm = 0.0;
            for (uint32_t slot = 0; slot < n_used; slot++) {
                const uint32_t expert = (uint32_t)selected_host[slot];
                float mid_host[256];
                for (uint32_t row = 0; row < mid_dim; row++) {
                    double g = 0.0, u = 0.0;
                    for (uint32_t i = 0; i < in_dim; i++) {
                        const float qx =
                            xq[i / QK_K].d * (float)xq[i / QK_K].qs[i % QK_K];
                        if (c.gate_type == 12u) {
                            g += q4_K_value((const block_q4_K *)(g_model + gate_off) +
                                                ((size_t)expert * mid_dim + row) * gb_in, i) * qx;
                            u += q4_K_value((const block_q4_K *)(g_model + up_off) +
                                                ((size_t)expert * mid_dim + row) * gb_in, i) * qx;
                        } else {
                            g += q6_K_value((const block_q6_K *)(g_model + gate_off) +
                                                ((size_t)expert * mid_dim + row) * gb_in, i) * qx;
                            u += q6_K_value((const block_q6_K *)(g_model + up_off) +
                                                ((size_t)expert * mid_dim + row) * gb_in, i) * qx;
                        }
                    }
                    mid_host[row] = silu_f32((float)g) * (float)u *
                                    weight_host[slot];
                }
                block_q8_K mq[4];
                for (uint32_t b = 0; b < gb_mid; b++) {
                    q8_K_quantize(&mq[b], mid_host + (size_t)b * QK_K);
                }
                for (uint32_t i = 0; i < mid_dim; i++) {
                    const float qm =
                        mq[i / QK_K].d * (float)mq[i / QK_K].qs[i % QK_K];
                    const float dw = (c.down_type == 12u)
                        ? q4_K_value((const block_q4_K *)(g_model + down_off) +
                                         (size_t)expert * out_dim * gb_mid +
                                         (size_t)r * gb_mid, i)
                        : q6_K_value((const block_q6_K *)(g_model + down_off) +
                                         (size_t)expert * out_dim * gb_mid +
                                         (size_t)r * gb_mid, i);
                    ref += (double)dw * qm;
                    norm += fabsf(dw * qm);
                }
            }
            const float err = fabsf((float)ref - got[r]);
            const float lim = 0.005f * fabsf((float)ref) + 0.002f * (float)norm;
            const double rel = (double)err / (lim > 1e-6f ? lim : 1e-6f);
            if (rel > worst_rel) worst_rel = rel;
        }
        CHECK(worst_rel <= 1.0, "%s MoE vs CPU ref (worst=%.3f of limit)",
              c.name, worst_rel);
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(sel);
    ds4_gpu_tensor_free(xt);
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        printf("no CUDA devices visible; skipping\n");
        return 0;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "FAIL: ds4_gpu_init\n");
        return 1;
    }

    printf("Laguna CRACK kernel tests:\n");
    model_map_init(64ull << 20);

    test_dense_matmul(12u);   /* Q4_K */
    test_dense_matmul(14u);   /* Q6_K */
    test_embed(12u);
    test_embed(14u);

    const struct moe_case cases[] = {
        { "Q4_K gate/up + Q6_K down (CRACK Q4_K_M)", 12u, 14u },
        { "all-Q6_K (CRACK Q6_K)", 14u, 14u },
        { "all-Q4_K down+up sanity", 12u, 12u },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        test_moe(cases[i]);
    }

    printf("\n%s (%d failures)\n", g_failures ? "FAILED" : "PASSED",
           g_failures);
    return g_failures ? 1 : 0;
}

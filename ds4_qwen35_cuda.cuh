/* Prism Bonsai (qwen35): the folded-weight activation transform on CUDA.
 * Included by ds4_cuda.cu so model residency, streams and temporary
 * allocations have the same lifetime as the other CUDA paths.
 *
 * A folded Prism export stores the matmul weights in the rotated basis, so
 * the runtime rotates the activation instead:
 *
 *   forward:  a' = H_bs(s * a)     every folded matmul input
 *   inverse:  x  = s * H_bs(z)     token-embedding row lookups
 *
 * H_bs is the normalized Sylvester Walsh-Hadamard transform over blocks of
 * block_size consecutive values and s is the sign vector of the weight's
 * input width.  H is symmetric and orthogonal with H*H = I, which is why the
 * inverse only swaps the two steps.  ds4_hadamard_* in ds4.c is the
 * reference; these kernels are element-for-element the same butterfly.
 *
 * The gated delta-net output projection is folded over the grouped head order
 * instead of the tiled one, so its input is reordered from [hd][nk][rep] to
 * [hd][rep][nk] before the rotation (ds4_hadamard_gdn_permute). */

#include <stdint.h>

namespace qwen35_cuda {

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static int launched(const char *what) {
    return cuda_ok(cudaGetLastError(), what);
}

/* Undo of the gdn reorder: element i of the grouped order comes from element
 * h + hd*(k + nk*r) of the tiled order, with i = h + hd*(r + rep*k).  It is a
 * bijection but, with nk != rep, not its own inverse, so only this explicit
 * map defines it (same map as ds4_hadamard_gdn_permute). */
__device__ __forceinline__ uint32_t gdn_src(uint32_t i, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint32_t h = i % hd;
    const uint32_t rest = i / hd;        /* r + rep*k */
    const uint32_t r = rest % rep;
    const uint32_t k = rest / rep;
    return h + hd * (k + nk * r);
}

/* One block of one row per CUDA block; blockDim.x must equal bs.  The butterfly
 * runs in shared memory: at each stage a thread takes its partner's value
 * before any write, so the two barriers per stage are what makes the exchange
 * safe.  Values stay per-thread registers between stages; the 10 stages of a
 * 1024-point transform are cheap next to the matmuls that follow. */
__global__ void fold_rotate(const float * __restrict__ src, float * __restrict__ dst,
                            const float * __restrict__ signs, uint32_t n, uint32_t bs,
                            int inverse) {
    extern __shared__ float tile[];
    const uint32_t tid = threadIdx.x;
    const uint32_t base = blockIdx.x * bs;
    const uint32_t i = base + tid;
    const float *row = src + (uint64_t) blockIdx.y * n;
    float *out = dst + (uint64_t) blockIdx.y * n;

    float v = row[i];
    if (signs && !inverse) v *= signs[i];
    tile[tid] = v;
    __syncthreads();

    for (uint32_t len = 1; len < bs; len <<= 1) {
        const float a = tile[tid];
        const float b = tile[tid ^ len];
        __syncthreads();
        tile[tid] = (tid & len) ? b - a : a + b;
        __syncthreads();
    }

    float r = tile[tid] * (1.0f / sqrtf((float) bs));
    if (signs && inverse) r *= signs[i];
    out[i] = r;
}

__global__ void fold_gdn_permute(float * __restrict__ dst, const float * __restrict__ src,
                                 uint32_t n, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* blockIdx.y is the row: a prefill chunk permutes hundreds of rows, and
     * one launch for the whole chunk costs the same GPU work as one launch per
     * row while removing hundreds of host launches per fold. */
    const uint64_t off = (uint64_t) blockIdx.y * n;
    dst[off + i] = src[off + gdn_src(i, hd, nk, rep)];
}

} // namespace

/* Validate the fold's shape contract: one or more rows of n values, n a whole
 * number of block_size blocks, block_size a power of two that fits a CUDA
 * block, and (for the gdn case) a head geometry that tiles the row exactly. */
static int ds4_qwen35_fold_shape(uint32_t n, uint32_t n_tok, uint32_t bs,
                                 uint32_t gdn, uint32_t hd, uint32_t nk, uint32_t rep) {
    if (n == 0 || n_tok == 0) return 0;
    if (bs == 0 || bs > 1024 || (bs & (bs - 1)) != 0) return 0;
    if (n % bs != 0) return 0;
    if (gdn) {
        if (hd == 0 || nk == 0 || rep == 0) return 0;
        if ((uint64_t) hd * nk * rep != n) return 0;
    }
    return 1;
}

static int ds4_qwen35_fold_launch(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t bs,
        const ds4_gpu_tensor *signs, uint32_t gdn, uint32_t hd, uint32_t nk,
        uint32_t rep, int inverse) {
    if (!ds4_qwen35_fold_shape(n, n_tok, bs, gdn, hd, nk, rep)) return 0;
    if (!qwen35_cuda::tensor(x, (uint64_t) n * n_tok * sizeof(float))) return 0;
    if (signs && !qwen35_cuda::tensor(signs, (uint64_t) n * sizeof(float))) return 0;

    const float *src = (const float *) x->ptr;
    float *dst = (float *) x->ptr;
    if (gdn) {
        /* The reorder mixes values across the whole row, so it cannot ride
         * along inside the block-local butterfly: permute into the shared
         * CUDA temporary first, then rotate from there into the tensor. */
        float *tmp = (float *) cuda_tmp_alloc((uint64_t) n * n_tok * sizeof(float),
                                              "Bonsai gdn fold permute");
        if (!tmp) return 0;
        qwen35_cuda::fold_gdn_permute<<<dim3((n + 255u) / 256u, n_tok), 256, 0, cuda_decode_stream()>>>(
            tmp, (const float *) x->ptr, n, hd, nk, rep);
        src = tmp;
    }

    const float *sg = signs ? (const float *) signs->ptr : NULL;
    qwen35_cuda::fold_rotate<<<dim3(n / bs, n_tok), bs, bs * sizeof(float),
                               cuda_decode_stream()>>>(src, dst, sg, n, bs, inverse);
    return qwen35_cuda::launched("Bonsai fold");
}

/* Forward fold for every folded matmul input: a' = H_bs(s * a).  When gdn is
 * set the tiled-to-grouped reorder is applied first, as the ssm_out
 * projection requires. */
extern "C" int ds4_gpu_qwen35_fold_forward_tensor(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t block_size,
        const ds4_gpu_tensor *signs, uint32_t gdn, uint32_t hd, uint32_t nk,
        uint32_t rep) {
    return ds4_qwen35_fold_launch(x, n, n_tok, block_size, signs, gdn, hd, nk,
                                  rep, /*inverse=*/0);
}

/* Inverse fold for token-embedding lookups: x = s * H_bs(z). */
extern "C" int ds4_gpu_qwen35_fold_inverse_tensor(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t block_size,
        const ds4_gpu_tensor *signs) {
    return ds4_qwen35_fold_launch(x, n, n_tok, block_size, signs, /*gdn=*/0,
                                  0, 0, 0, /*inverse=*/1);
}

/* Gated output norm of the Bonsai linear layer: the qwen4 kernel with the
 * silu gate this family uses (ds4_qwen35_ref_linear) instead of sigmoid. */
extern "C" int ds4_gpu_qwen35_gdn_out_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H,
        uint32_t D, float eps) {
    const uint64_t n = (uint64_t) T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !qwen35_cuda::tensor(out, n * 4) ||
        !qwen35_cuda::tensor(z, n * 4)) {
        return 0;
    }
    const char *w = cuda_resolve_weight_ptr(map, off, (uint64_t) D * 4, 0, "Bonsai lin_norm");
    if (!w) return 0;
    qwen4_cuda::gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>(
        (float *) out->ptr, (const float *) z->ptr, (const float *) w, H, D,
        eps, /*gate_silu=*/1u);
    return qwen35_cuda::launched("Bonsai gdn out");
}

/* Attention prep of the Bonsai full-attention layer: the qwen4 kernel with no
 * indexer slot (this model has no sparse indexer), so it produces the roped
 * q with its raw sigmoid gate, the roped k and the v store.  The q and k
 * per-head norms both carry a gamma of length D and the same eps. */
extern "C" int ds4_gpu_qwen35_attn_prep_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate, ds4_gpu_tensor *kc,
        ds4_gpu_tensor *vc, const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp,
        const ds4_gpu_tensor *vp, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint32_t T,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot, uint32_t pos0,
        uint32_t cap, float base, float eps) {
    using namespace qwen35_cuda;
    const uint64_t qb = (uint64_t) T * H * D * 4, kb = (uint64_t) T * Hkv * D * 4;
    if (!T || !H || !Hkv || H % Hkv || D < 32 || D > 256 || D % 32 ||
        nrot > 64 || nrot > D || nrot % 2 || (uint64_t) pos0 + T > cap ||
        !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) ||
        !tensor(kc, (uint64_t) cap * Hkv * D * 2) ||
        !tensor(vc, (uint64_t) cap * Hkv * D * 2) ||
        !tensor(pos3, (uint64_t) cap * 16)) {
        return 0;
    }
    const char *gq = cuda_resolve_weight_ptr(map, qo, (uint64_t) D * 4, 0, "Bonsai attn_q_norm");
    const char *gk = cuda_resolve_weight_ptr(map, ko, (uint64_t) D * 4, 0, "Bonsai attn_k_norm");
    if (!gq || !gk) return 0;
    qwen4_cuda::attn_prep<<<dim3(H + Hkv, T), 32, 0, cuda_decode_stream()>>>(
        (float *) q->ptr, (float *) gate->ptr, (__half *) kc->ptr,
        (__half *) vc->ptr, /*iqout=*/NULL, /*ikc=*/NULL,
        (const float *) qg->ptr, (const float *) kp->ptr,
        (const float *) vp->ptr, /*iq=*/NULL, /*ik=*/NULL,
        (const uint32_t *) pos3->ptr, (const float *) gq, (const float *) gk,
        /*giq=*/NULL, H, Hkv, D, /*Hi=*/0, /*Di=*/0, pos0, eps,
        qwen4_cuda::rope(nrot, base));
    return launched("Bonsai attn prep");
}

/* Session-path test for Prism Bonsai (qwen35 / PQ2_0) on the CUDA backend.
 *
 * The oracle is ds4's own CPU reference, reached through the DS4_TEST_HOOKS
 * entry point ds4_test_qwen35_ref_greedy (ds4.c), so nothing here
 * re-implements the model, the block format or the greedy loop, and no
 * external implementation is involved.
 *
 * For one prompt and one step count, the reference ids are computed once and
 * the session path must reproduce them exactly through every entry point the
 * session/server code uses:
 *
 *   1. plain decode: create, sync, then argmax/eval per step;
 *   2. prefix reuse: a second sync that extends the checkpoint (the graph state
 *      is not rebuilt and the ids still match);
 *   3. rewind + feed-back: rewind to the prompt, then force the decoded ids
 *      back through the session (the stale-graph replay path);
 *   4. invalidate + resync: the rebuild path;
 *   5. the context bound: a decode past the graph's capacity fails instead of
 *      reading outside the caches.
 *
 * Build and run:
 *   DS4_TEST_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
 *       make test-qwen35-session */

#include "ds4.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Defined in ds4.c under DS4_TEST_HOOKS (built as ds4_cuda_test_hooks.o). */
extern int ds4_test_qwen35_ref_greedy(ds4_engine *e, const int *tokens, int n_tokens,
                                      int steps, int *out, int out_cap);

#define MAX_STEPS 16

static const char kPromptText[] = "The capital of France is";

static int failures;

static void report(const char *what, bool ok) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static void print_ids(const char *label, const int *ids, int n) {
    printf("      %s:", label);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
}

static bool ids_equal(const int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static int first_difference(const int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return i;
    }
    return -1;
}

/* Shared by the scenarios: emit the next greedy token, then advance. */
static int step_decode(ds4_session *s, int *out, int steps, char *err, size_t errlen) {
    for (int i = 0; i < steps; i++) {
        const int best = ds4_session_argmax(s);
        if (best < 0) {
            snprintf(err, errlen, "argmax failed at step %d", i);
            return 1;
        }
        out[i] = best;
        if (i + 1 < steps && ds4_session_eval(s, best, err, errlen) != 0) return 1;
    }
    return 0;
}

/* 1. Plain decode from a fresh session. */
static bool scenario_plain(ds4_engine *e, const ds4_tokens *prompt, int steps,
                           const int *want) {
    ds4_session *s = NULL;
    char err[200] = "";
    int got[MAX_STEPS];
    bool ok = true;
    if (ds4_session_create(&s, e, prompt->len + steps + 2) != 0) {
        report("plain: session create", false);
        return false;
    }
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        report("plain: session sync", false);
        ds4_session_free(s);
        return false;
    }
    ok = ds4_session_pos(s) == prompt->len;
    report("plain: position after sync equals the prompt length", ok);
    ok = ds4_session_ctx(s) == prompt->len + steps + 2;
    report("plain: session reports the created context size", ok);
    if (step_decode(s, got, steps, err, sizeof(err)) != 0) {
        printf("FAIL  plain: decode failed: %s\n", err);
        failures++;
        ds4_session_free(s);
        return false;
    }
    ok = ids_equal(got, want, steps);
    report("plain: decoded ids equal the CPU reference", ok);
    if (!ok) {
        print_ids("reference", want, steps);
        print_ids("session  ", got, steps);
        printf("      first difference at step %d\n", first_difference(got, want, steps));
    }
    ds4_session_free(s);
    return ok;
}

/* 2. A second sync extends the checkpoint: the ids must continue unchanged. */
static bool scenario_prefix_reuse(ds4_engine *e, const ds4_tokens *prompt, int steps,
                                  const int *want) {
    ds4_session *s = NULL;
    ds4_tokens ext = {0};
    char err[200] = "";
    int got[MAX_STEPS];
    const int half = steps / 2;
    bool ok = true;
    if (ds4_session_create(&s, e, prompt->len + steps + 2) != 0 ||
        ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        report("prefix reuse: session create and sync", false);
        ds4_session_free(s);
        return false;
    }
    if (step_decode(s, got, half, err, sizeof(err)) != 0) {
        printf("FAIL  prefix reuse: first half decode failed: %s\n", err);
        failures++;
        ds4_session_free(s);
        return false;
    }
    ds4_tokens_copy(&ext, prompt);
    for (int i = 0; i < half; i++) ds4_tokens_push(&ext, got[i]);
    if (ds4_session_sync(s, &ext, err, sizeof(err)) != 0) {
        printf("FAIL  prefix reuse: extending sync failed: %s\n", err);
        failures++;
        ds4_tokens_free(&ext);
        ds4_session_free(s);
        return false;
    }
    ok = ds4_session_pos(s) == ext.len;
    report("prefix reuse: position after the extending sync", ok);
    if (step_decode(s, got + half, steps - half, err, sizeof(err)) != 0) {
        printf("FAIL  prefix reuse: second half decode failed: %s\n", err);
        failures++;
        ds4_tokens_free(&ext);
        ds4_session_free(s);
        return false;
    }
    ok = ids_equal(got, want, steps);
    report("prefix reuse: ids equal the CPU reference across both syncs", ok);
    if (!ok) {
        print_ids("reference", want, steps);
        print_ids("session  ", got, steps);
        printf("      first difference at step %d\n", first_difference(got, want, steps));
    }
    ds4_tokens_free(&ext);
    ds4_session_free(s);
    return ok;
}

/* 3. Rewind to the prompt, then feed the decoded ids back: the graph replays
 * the kept tokens, and the argmax after each feed must repeat the reference. */
static bool scenario_rewind_replay(ds4_engine *e, const ds4_tokens *prompt, int steps,
                                   const int *want) {
    ds4_session *s = NULL;
    char err[200] = "";
    int got[MAX_STEPS];
    bool ok = true;
    if (ds4_session_create(&s, e, prompt->len + steps + 2) != 0 ||
        ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        report("rewind: session create and sync", false);
        ds4_session_free(s);
        return false;
    }
    if (step_decode(s, got, steps, err, sizeof(err)) != 0) {
        printf("FAIL  rewind: decode before the rewind failed: %s\n", err);
        failures++;
        ds4_session_free(s);
        return false;
    }
    ds4_session_rewind(s, prompt->len);
    ok = ds4_session_pos(s) == prompt->len;
    report("rewind: position is back at the prompt", ok);
    for (int i = 0; i < steps && ok; i++) {
        if (ds4_session_eval(s, want[i], err, sizeof(err)) != 0) {
            printf("FAIL  rewind: feed-back eval failed at step %d: %s\n", i, err);
            failures++;
            ok = false;
            break;
        }
        ok = ds4_session_pos(s) == prompt->len + i + 1;
        if (!ok) {
            printf("FAIL  rewind: position after feeding step %d is %d, expected %d\n",
                   i, ds4_session_pos(s), prompt->len + i + 1);
            failures++;
            break;
        }
        if (i + 1 < steps) {
            const int next = ds4_session_argmax(s);
            ok = next == want[i + 1];
            if (!ok) {
                printf("FAIL  rewind: argmax after feeding step %d is %d, reference %d\n",
                       i, next, want[i + 1]);
                failures++;
            }
        }
    }
    report("rewind: the replay reproduces the reference ids", ok);
    ds4_session_free(s);
    return ok;
}

/* 4. Invalidate clears the checkpoint; a fresh sync rebuilds the state and the
 * ids still match. */
static bool scenario_invalidate(ds4_engine *e, const ds4_tokens *prompt, int steps,
                                const int *want) {
    ds4_session *s = NULL;
    char err[200] = "";
    int got[MAX_STEPS];
    bool ok = true;
    if (ds4_session_create(&s, e, prompt->len + steps + 2) != 0 ||
        ds4_session_sync(s, prompt, err, sizeof(err)) != 0 ||
        step_decode(s, got, steps, err, sizeof(err)) != 0) {
        report("invalidate: create, sync and first decode", false);
        ds4_session_free(s);
        return false;
    }
    ds4_session_invalidate(s);
    ok = ds4_session_pos(s) == 0;
    report("invalidate: the checkpoint is empty", ok);
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        printf("FAIL  invalidate: resync failed: %s\n", err);
        failures++;
        ds4_session_free(s);
        return false;
    }
    if (step_decode(s, got, steps, err, sizeof(err)) != 0) {
        printf("FAIL  invalidate: decode after the resync failed: %s\n", err);
        failures++;
        ds4_session_free(s);
        return false;
    }
    ok = ids_equal(got, want, steps);
    report("invalidate: ids after the rebuild equal the reference", ok);
    ds4_session_free(s);
    return ok;
}

/* 5. A decode past the graph's context capacity is refused, not run. */
static bool scenario_context_bound(ds4_engine *e, const ds4_tokens *prompt) {
    ds4_session *s = NULL;
    char err[200] = "";
    const int ctx = prompt->len + 2;
    int refused_at = -1;
    if (ds4_session_create(&s, e, ctx) != 0 ||
        ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        report("context bound: session create and sync", false);
        ds4_session_free(s);
        return false;
    }
    /* Two more tokens fill the context: pose 0 and 1 succeed, the third eval
     * has no room. */
    for (int i = 0; i < 4; i++) {
        const int best = ds4_session_argmax(s);
        if (best < 0 || ds4_session_eval(s, best, err, sizeof(err)) != 0) {
            refused_at = i;
            break;
        }
    }
    const bool ok = refused_at == 2;
    report("context bound: the third decode past capacity is refused", ok);
    if (!ok) printf("      refused at eval %d (expected 2), error: %s\n", refused_at, err);
    ds4_session_free(s);
    return ok;
}

int main(void) {
    /* The reference pass takes over a minute; keep the report visible while it
     * runs rather than flushing it in one block at exit. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) model = getenv("DS4_BONSAI_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "test_qwen35_session: set DS4_TEST_MODEL to the Bonsai GGUF\n");
        return 2;
    }

    const int steps = getenv("DS4_TEST_STEPS") ? atoi(getenv("DS4_TEST_STEPS")) : 12;
    if (steps < 2 || steps > MAX_STEPS) {
        fprintf(stderr, "test_qwen35_session: DS4_TEST_STEPS must be 2..%d\n", MAX_STEPS);
        return 2;
    }

    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .context_size = 64,
        /* Threads left at the engine default: the CPU reference is the slow
         * part of this test and it is row-parallel across the host pool. */
        .quality = false,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_create_with_gpu_config(&e, &opt, NULL) != 0) {
        fprintf(stderr, "test_qwen35_session: engine open failed\n");
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, kPromptText, &prompt);
    if (prompt.len <= 0) {
        fprintf(stderr, "test_qwen35_session: tokenizing the prompt produced no tokens\n");
        ds4_engine_close(e);
        return 1;
    }
    printf("prompt: \"%s\" -> %d tokens:", kPromptText, prompt.len);
    for (int i = 0; i < prompt.len; i++) printf(" %d", prompt.v[i]);
    printf("\nsteps: %d greedy tokens\n\n", steps);

    int want[MAX_STEPS];
    if (ds4_test_qwen35_ref_greedy(e, prompt.v, prompt.len, steps, want, MAX_STEPS) != steps) {
        fprintf(stderr, "test_qwen35_session: the reference run failed\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(e);
        return 1;
    }
    print_ids("CPU reference (the oracle)", want, steps);
    printf("\n");

    scenario_plain(e, &prompt, steps, want);
    scenario_prefix_reuse(e, &prompt, steps, want);
    scenario_rewind_replay(e, &prompt, steps, want);
    scenario_invalidate(e, &prompt, steps, want);
    scenario_context_bound(e, &prompt);

    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    printf("\nqwen35 session path: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_HPP
#define COMMON_HPP

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include "src/common/z_magic.hpp"

#include "utils/bench_mode.hpp"
#include "utils/res.hpp"
#include "utils/timer.hpp"

#define ABS(a) ((a) > 0 ? (a) : (-(a)))

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

#define MIN3(a, b, c) MIN2(a, MIN2(b, c))
#define MAX3(a, b, c) MAX2(a, MAX2(b, c))

#define IMPLICATION(cause, effect) (!(cause) || !!(effect))

#if defined(_WIN32) && !defined(__GNUC__)
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#define IFMT "%" PRId64

#define OK 0
#define FAIL 1

enum { CRIT = 1, WARN = 2 };

#define SAFE(f, s) \
    do { \
        int status__ = (f); \
        if (status__ != OK) { \
            if ((s) == CRIT || (s) == WARN) { \
                BENCHDNN_PRINT(0, \
                        "Error: Function '%s' at (%s:%d) returned '%d'\n", \
                        __FUNCTION__, __FILE__, __LINE__, status__); \
                fflush(0); \
                if ((s) == CRIT) exit(1); \
            } \
            return status__; \
        } \
    } while (0)

#define SAFE_V(f) \
    do { \
        int status__ = (f); \
        if (status__ != OK) { \
            BENCHDNN_PRINT(0, \
                    "Error: Function '%s' at (%s:%d) returned '%d'\n", \
                    __FUNCTION__, __FILE__, __LINE__, status__); \
            fflush(0); \
            exit(1); \
        } \
    } while (0)

extern int verbose;
extern bool canonical;
extern bool mem_check;
extern bool attr_same_pd_check;
extern bool check_ref_impl;
extern std::string skip_impl; /* empty or "" means skip nothing */
extern std::string driver_name;

#define BENCHDNN_PRINT(v, fmt, ...) \
    do { \
        if (verbose >= v) { \
            printf(fmt, __VA_ARGS__); \
            /* printf("[%d][%s:%d]" fmt, v, __func__, __LINE__, __VA_ARGS__); */ \
            fflush(0); \
        } \
    } while (0)

//NOLINTBEGIN(bugprone-macro-parentheses)
// dnnl_common.hpp:119:5: error: expected ';' at end of declaration list [clang-diagnostic-error]
//  119 |     BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
//      |     ^
#define BENCHDNN_DISALLOW_COPY_AND_ASSIGN(T) \
    T(const T &) = delete; \
    T &operator=(const T &) = delete;
//NOLINTEND(bugprone-macro-parentheses)

/* perf */
extern double max_ms_per_prb; // max time spend per prb in ms
extern double default_max_ms_per_prb; // default max time spend per prb in ms
extern int min_times_per_prb; // min number of runs per prb
extern int fix_times_per_prb; // if non-zero run prb that many times
extern int default_fix_times_per_prb; // 0, rely on time criterion
extern int repeats_per_prb; // test repeats per prb
extern int default_repeats_per_prb; // default test repeats per prb

extern bool fast_ref;
extern bool default_fast_ref;
extern bool allow_enum_tags_only;
extern int test_start;

/* global stats */
struct stat_t {
    int tests;
    int passed;
    int failed;
    int skipped;
    int mistrusted;
    int unimplemented;
    int invalid_arguments;
    int listed;
    std::unordered_map<std::string, double[timer::timer_t::mode_t::n_modes]> ms;
    // Key is the number of the test, value is the repro string.
    std::map<int, std::string> failed_cases;
};
extern stat_t benchdnn_stat;

const char *state2str(res_state_t state);

namespace skip_reason {
extern std::string case_not_supported;
extern std::string data_type_not_supported;
extern std::string invalid_case;
extern std::string not_enough_ram;
extern std::string skip_impl_hit;
extern std::string skip_start;
} // namespace skip_reason

dir_t str2dir(const char *str);
void parse_result(res_t &res, const char *pstr);

/* misc */
void init_fp_mode();

void *zmalloc(size_t size, size_t align);
void zfree(void *ptr);
void set_zmalloc_max_expected_size(size_t size);

bool str2bool(const char *str);
const char *bool2str(bool value);

bool match_regex(const char *str, const char *pattern);
bool skip_start(res_t *res, int idx = benchdnn_stat.tests);

using bench_f = int (*)(int, char **);
std::string locate_file(const std::string &fname);
int batch(const char *fname, bench_f bench);

/* returns 1 with given probability */
int flip_coin(ptrdiff_t seed, float probability);

int64_t div_up(const int64_t a, const int64_t b);
size_t div_up(const size_t a, const size_t b);
int64_t rnd_up(const int64_t a, const int64_t b);
size_t rnd_up(const size_t a, const size_t b);
int64_t next_pow2(int64_t a);
int mxcsr_cvt(float f);

/* set '0' across *arr:+size */
void array_set(char *arr, size_t size);

/* wrapper to dnnl_sgemm
 * layout = 'F' - column major
 * layout = 'C' - row major*/
void gemm(const char *layout, const char *transa, const char *transb, int64_t m,
        int64_t n, int64_t k, const float alpha, const float *a,
        const int64_t lda, const float *b, const int64_t ldb, const float beta,
        float *c, const int64_t ldc);

int sanitize_desc(int &ndims, std::vector<std::reference_wrapper<int64_t>> d,
        std::vector<std::reference_wrapper<int64_t>> h,
        std::vector<std::reference_wrapper<int64_t>> w,
        const std::vector<int64_t> &def_values, const char *str,
        bool must_have_spatial = false);

void print_dhw(bool &print_d, bool &print_h, bool &print_w, int ndims,
        const std::vector<int64_t> &d, const std::vector<int64_t> &h,
        const std::vector<int64_t> &w);

int benchdnn_getenv_int(const char *name, int default_value);
std::string benchdnn_getenv_string(const char *name);

// Responsible for printing service information.
struct summary_t {
    // Prints up to 10 failed cases reproducers at the end of the run.
    bool failed_cases = true;
};

extern summary_t summary;

std::string smart_bytes(double bytes);
#endif

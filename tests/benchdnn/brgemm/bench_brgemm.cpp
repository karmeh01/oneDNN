/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"

#include "brgemm/brgemm.hpp"

namespace brgemm {

#if ((defined(DNNL_X64) && DNNL_X64 == 1) \
        || (defined(DNNL_AARCH64) && DNNL_AARCH64 == 1)) \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE

void check_correctness(const settings_t &s) {
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_bia_dt : s.bia_dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_ld : s.ld)
    for_(const auto &i_alpha : s.alpha)
    for_(const auto &i_beta : s.beta)
    for_(const auto &i_batch_size : s.batch_size)
    for_(const auto &i_brgemm_attr : s.brgemm_attr)
    for_(const auto &i_batch_kind : s.batch_kind)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        const prb_t prb(s.prb_vdims, i_dt, i_stag, i_wtag, i_dtag, i_strides,
                i_ld, i_bia_dt, i_alpha, i_beta, i_batch_size, i_brgemm_attr,
                i_batch_kind, i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;
        BENCHDNN_PRINT(1, "run: %s\n", prb.str());

        res_t res {};
        doit(&prb, &res);

        parse_result(res, prb.str());

        if (has_bench_mode_bit(mode_bit_t::perf)) {
            perf_report_t pr(&prb, s.perf_template);
            pr.report(&res, prb.str());
        }
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    static constexpr int n_inputs = 3;

    if (s.prb_vdims.ndims > 2) {
        fprintf(stderr,
                "ERROR: brgemm driver: problem descriptor supports only "
                "MxK:KxN notion.\n"),
                fflush(stderr);
        SAFE_V(FAIL);
    }

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: brgemm driver: `dt` option expects either a single "
                    "input or three inputs in SRC, WEI, and DST order. Current "
                    "size is: \"%ld\"\n",
                    (long)i_dt.size()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `strides` option expects three inputs in format "
                    "`[SRC]:[WEI]:[DST]` (two colons must be present).");
            return FAIL;
        }
    }

    for (const auto &i_strides : s.strides) {
        const bool src_dst_strided_input = !i_strides[STRIDES_SRC].empty()
                || !i_strides[STRIDES_DST].empty();
        if (src_dst_strided_input) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `strides` option supports only weights strides and "
                    "expects input in the format `:[WEI]:` (two colons must be "
                    "present).");
            return FAIL;
        }

        const bool strided_input = !i_strides[STRIDES_WEI].empty();
        if (!strided_input) continue;

        for (const auto &i_wtag : s.wtag) {
            const bool no_stride_with_tag = IMPLICATION(
                    i_wtag != def.wtag[0], i_strides[STRIDES_WEI].empty());

            if (!no_stride_with_tag) {
                BENCHDNN_PRINT(0, "%s\n",
                        "ERROR: both `strides` and `tag` knobs can not be used "
                        "with `wei` tensor.\n");
                return FAIL;
            }
        }
    }

    return OK;
}

static const std::string help_alpha
        = "FLOAT    (Default: 1.f)\n    Specifies real value corresponding to "
          "scaling of accumulator result: `C = alpha * A * B`.\n";

static const std::string help_beta
        = "FLOAT    (Default: 0.f)\n    Specifies real value corresponding to "
          "adding a part of accumulator result: `C = A * B + beta * C`.\n";

static const std::string help_batch_size
        = "UINT    (Default: `1`)\n    Specifies a batch size that indicates "
          "how many batches per kernel call will be used.\n";

static const std::string help_ld
        = "UINT:UINT:UINT    (Default: not specified)\n    Specifies "
          "LDA:LDB:LDD values. If some values are skipped, the default one (K, "
          "N, or N) will be used. If there are no post-ops, LDC will reuse "
          "LDD, otherwise expect LDC always dense.\n";

static const std::string help_brgemm_attr
        = "STRING    (Default: empty)\n    Specifies BRGeMM kernel attributes. "
          "If some values are skipped, the default one will be used.\n";

static const std::string help_batch_kind
        = "STRING    (Default: addr)\n    Specifies BRGeMM batch kind. "
          "Supported values are: `addr`, `offs`.\n";

int bench(int argc, char **argv) {
    // BRGeMM kernel support is available on x86 Intel CPU only.
    if (is_gpu()) return OK;
    driver_name = "brgemm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        auto cstr2str = [](const char *str) { return std::string(str); };
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_multivector_option(
                        s.ld, def.ld, atoi, argv[0], "ld", help_ld)
                || parse_vector_option(s.batch_size, def.batch_size, atoi,
                        argv[0], "bs", help_batch_size)
                || parse_vector_option(
                        s.alpha, def.alpha, atof, argv[0], "alpha", help_alpha)
                || parse_vector_option(
                        s.beta, def.beta, atof, argv[0], "beta", help_beta)
                || parse_vector_option(s.brgemm_attr, def.brgemm_attr, cstr2str,
                        argv[0], "brgemm-attr", help_brgemm_attr)
                || parse_vector_option(s.batch_kind, def.batch_kind, cstr2str,
                        argv[0], "batch-kind", help_batch_kind)
                || parse_attributes(s, def, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_perf_template(s.perf_template,
                        settings_t::perf_template_def, s.perf_template_csv(),
                        argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s);
        }
    }
    return parse_last_argument();
}

#else

int bench(int argc, char **argv) {
    BENCHDNN_PRINT(0, "%s\n",
            "INFO: brgemm driver: only x64, aarch64 backend is supported.");
    return OK;
}

#endif

} // namespace brgemm

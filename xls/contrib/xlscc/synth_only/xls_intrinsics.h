// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_INTRINSICS_H_
#define XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_INTRINSICS_H_

#include <xls_int.h>

#ifndef __SYNTHESIS__
// TODO(seanhaskell): Consider whether we can/should provide implementations for
// non-synthesized applications.
static_assert(false, "This header is only for synthesis");
#endif  // __SYNTHESIS__

namespace xls_intrinsics {

// TODO(seanhaskell): Decide whether intrinsics should live in XlsInt or here.

template <int Width, bool Signed>
inline typename XlsInt<Width, Signed>::index_t ctz(
    const XlsInt<Width, Signed>& in) {
  XlsInt<Width + 1, false> one_hot_out;
  asm("fn (fid)(a: bits[i]) -> bits[c] { "
      "  ret (aid): bits[c] = one_hot(a, lsb_prio=true, pos=(loc)) }"
      : "=r"(one_hot_out.storage)
      : "i"(Width), "c"(Width + 1), "a"(in.storage));

  typename XlsInt<Width, Signed>::index_t encode_out;
  asm("fn (fid)(a: bits[i]) -> bits[c] { "
      "  ret (aid): bits[c] = encode(a, pos=(loc)) }"
      : "=r"(encode_out.storage)
      : "i"(Width + 1), "c"(XlsInt<Width, Signed>::index_t::width),
        "a"(one_hot_out));

  // zero_ext is automatic
  return encode_out;
}

template <int Width, bool Signed>
inline typename XlsInt<Width, Signed>::index_t clz(
    const XlsInt<Width, Signed>& in) {
  return ctz(in.reverse());
}

}  // namespace xls_intrinsics

#endif  // XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_INTRINSICS_H_

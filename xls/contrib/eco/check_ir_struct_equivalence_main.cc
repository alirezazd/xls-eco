// Copyright 2026 The XLS Authors
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

// Checks STRUCTURAL equivalence of two XLS IR files by comparing their
// canonical graph fingerprints (XLSGraph::StructuralHash). It is the cheap
// counterpart to //xls/dev_tools:check_ir_equivalence_main (which proves
// LOGICAL equivalence with Z3): a single O(V+E) hash per input instead of an
// SMT solve, for the common case where two IRs are expected to be structurally
// identical (e.g. an ECO-patched IR vs. its intended revision).
//
// Soundness: equal fingerprints are a necessary condition for structural
// isomorphism, sound modulo hash collision. It reports "not equivalent" for
// IRs that are functionally equal but structurally different (commutative
// reorder is absorbed, but strength reduction / re-optimization is not), so a
// mismatch means "not structurally identical", not "not logically equivalent".

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/struct_equivalence.h"
#include "xls/contrib/eco/xls_ir_to_graph.h"

ABSL_FLAG(int, match_exit_code, 0,
          "Value to exit with if the IRs are structurally equivalent.");
ABSL_FLAG(int, mismatch_exit_code, 255,
          "Value to exit with if the IRs are not structurally equivalent.");
ABSL_FLAG(std::string, equivalence_report_path, "",
          "Path to write the equivalence report file.");

namespace {

constexpr std::string_view kUsage =
    "Checks structural equivalence of two XLS IR files via canonical graph "
    "hashing.\n\n"
    "Usage:\n"
    "  check_ir_struct_equivalence_main IR_A IR_B [flags]\n";

std::string FormatRoots(const std::vector<std::string>& roots) {
  constexpr std::size_t kMaxShown = 8;
  const std::size_t shown = std::min(roots.size(), kMaxShown);
  std::string s = absl::StrJoin(roots.begin(), roots.begin() + shown, ", ");
  if (roots.size() > shown) {
    absl::StrAppend(&s, ", +", roots.size() - shown, " more");
  }
  return s;
}

absl::StatusOr<bool> RealMain(
    const std::vector<std::string_view>& positional_args,
    std::string report_path) {
  if (positional_args.size() != 2) {
    return absl::InvalidArgumentError(
        "Exactly two positional IR files must be specified.");
  }
  const std::string ir_a(positional_args[0]);
  const std::string ir_b(positional_args[1]);

  XLS_ASSIGN_OR_RETURN(XLSGraph g1, xls::ParseIrFileToGraph(ir_a));
  XLS_ASSIGN_OR_RETURN(XLSGraph g2, xls::ParseIrFileToGraph(ir_b));

  const std::vector<std::size_t> v1 = g1.StructuralHash();
  const std::vector<std::size_t> v2 = g2.StructuralHash();
  const std::size_t h1 = v1.back();
  const std::size_t h2 = v2.back();
  const bool equivalent = h1 == h2;

  std::string report = absl::StrJoin(
      std::vector<std::string>{
          equivalent ? "Structural equivalence verification successful."
                     : "Structural equivalence verification failed.",
          absl::StrCat("  File 1: ", ir_a, "  (", g1.nodes.size(), " nodes, ",
                       g1.edges.size(), " edges, hash=", h1, ")"),
          absl::StrCat("  File 2: ", ir_b, "  (", g2.nodes.size(), " nodes, ",
                       g2.edges.size(), " edges, hash=", h2, ")"),
          equivalent ? "  Verdict: structurally equivalent."
                     : "  Verdict: NOT structurally equivalent."},
      "\n");
  if (!equivalent) {
    const absl::Span<const std::size_t> n1 =
        absl::MakeConstSpan(v1).first(v1.size() - 1);
    const absl::Span<const std::size_t> n2 =
        absl::MakeConstSpan(v2).first(v2.size() - 1);
    absl::StrAppend(&report, "\n  Mismatch roots (file 1): ",
                    FormatRoots(xls::MismatchRoots(g1, n1, n2)),
                    "\n  Mismatch roots (file 2): ",
                    FormatRoots(xls::MismatchRoots(g2, n2, n1)));
  }

  if (!report_path.empty()) {
    XLS_RETURN_IF_ERROR(
        xls::SetFileContents(std::filesystem::path(report_path), report));
  }
  LOG(INFO) << report;

  return equivalent;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::vector<std::string_view> positional = xls::InitXls(kUsage, argc, argv);
  std::string report_path = absl::GetFlag(FLAGS_equivalence_report_path);
  absl::StatusOr<bool> equivalent =
      RealMain(positional, std::move(report_path));
  if (!equivalent.ok()) {
    return xls::ExitStatus(equivalent.status());
  }
  return *equivalent ? absl::GetFlag(FLAGS_match_exit_code)
                     : absl::GetFlag(FLAGS_mismatch_exit_code);
}

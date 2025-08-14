#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir {
class AffineMap;
class Operation;
class Region;
class Value;
}  // namespace mlir

namespace mlir::utils {
enum class IteratorType : uint32_t;
}  // namespace mlir::utils

namespace mlir::xls {

struct Dim;
struct Operand;
struct Region;
struct LinalgGeneric;
struct BroadcastAnalysis;
struct OutputShapeInfo;
struct OperandIndexFn;

AffineMap EvalAffineMap(mlir::AffineMap mlir_map);

FailureOr<Dim> EvalDimension(mlir::utils::IteratorType iterator_type,
                             size_t dim_index);
FailureOr<Operand> EvalOperand(mlir::Value value, const std::string& name,
                               bool is_output, mlir::AffineMap indexing_map);
FailureOr<Region> EvalRegion(mlir::Region& mlir_region);

FailureOr<std::vector<Dim>> BuildDimensions(
    mlir::linalg::GenericOp& generic_op);
FailureOr<std::vector<Operand>> BuildOperands(
    mlir::linalg::GenericOp& generic_op);

FailureOr<LinalgEvalResults> EvalLinalgGeneric(mlir::Operation* op);

mlir::LogicalResult Validate(const LinalgGeneric& g);

FailureOr<std::vector<std::vector<bool>>> EvalOperandDimensionRefs(
    const LinalgGeneric& g);
FailureOr<std::vector<OperandIndexFn>> EvalOperandIndexers(
    const LinalgGeneric& g);
FailureOr<std::vector<SizeExpr>> EvalLoopExtents(
    const LinalgGeneric& g,
    const std::vector<std::vector<bool>>& operand_refs_dim);

FailureOr<BroadcastAnalysis> EvalBroadcast(const LinalgGeneric& g);
FailureOr<OutputShapeInfo> DeriveOutputShape(const LinalgGeneric& g,
                                             const BroadcastAnalysis& A,
                                             int out_operand_index);

std::string LinalgGenericToString(const LinalgGeneric& linalg);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_

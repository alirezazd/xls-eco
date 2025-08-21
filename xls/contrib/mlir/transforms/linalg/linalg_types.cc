// Copyright 2025 The XLS Authors
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

#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

ValueId::ValueId(int i) : id(i) {}

bool ValueId::operator==(const ValueId& other) const { return id == other.id; }

bool ValueId::operator<(const ValueId& other) const { return id < other.id; }

SizeExpr::SizeExpr() : kind(kConstant), constant_value(1), symbolic_name("") {}

SizeExpr::SizeExpr(int64_t value)
    : kind(kConstant), constant_value(value), symbolic_name("") {}

SizeExpr::SizeExpr(std::string name)
    : kind(kSymbolic), constant_value(0), symbolic_name(std::move(name)) {}

bool SizeExpr::isConstant() const { return kind == kConstant; }

bool SizeExpr::isSymbolic() const { return kind == kSymbolic; }

int64_t SizeExpr::getConstant() const { return constant_value; }

const std::string& SizeExpr::getSymbolic() const { return symbolic_name; }

}  // namespace mlir::xls

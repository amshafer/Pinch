//===- Dialect.h - Dialect definition for the Pinch IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Pinch language.
// See docs/Tutorials/Pinch/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_PINCH_DIALECT_H_
#define MLIR_TUTORIAL_PINCH_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace pinch {

/// This is the definition of the Pinch dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class PinchDialect : public mlir::Dialect {
public:
  explicit PinchDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "pinch"; }
};

/// Include the auto-generated header file containing the declarations of the
/// pinch operations.
#define GET_OP_CLASSES
#include "Ops.h.inc"

} // end namespace pinch
} // end namespace mlir

#endif // MLIR_TUTORIAL_PINCH_DIALECT_H_

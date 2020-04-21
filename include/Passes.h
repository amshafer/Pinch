//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_PASSES_H
#define MLIR_TUTORIAL_TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace pinch {
std::unique_ptr<Pass> createBorrowCheckerPass();

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createLowerToStdPass();

} // end namespace pinch
} // end namespace mlir

#endif // MLIR_TUTORIAL_PINCH_PASSES_H

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

} // end namespace pinch
} // end namespace mlir

#endif // MLIR_TUTORIAL_PINCH_PASSES_H

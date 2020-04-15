//===- BorrowCheckerInterface.td - Shape Inference Interface -*- tablegen -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef BORROWC_INFERENCE_INTERFACE
#define BORROWC_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def BorrowCheckerOpInterface : OpInterface<"BorrowChecker"> {
  let description = [{
      Interface to access a registered method to check the borrow and
          movement rules for a set of mlir
  }];

  let methods = [
    InterfaceMethod<"Perform borrow checking",
                    "void", "inferBorrowChecker">
  ];
}

#endif

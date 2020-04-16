//===- BorrowCheckerPass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Value.h"
#include "Dialect.h"
#include "Passes.h"
#include "BorrowChecker.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace pinch;
using namespace std;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "BorrowCheckerOpInterfaces.cpp.inc"

namespace {

/// Ownership information for a particular variable
class Owner {
public:
  OpResult result;
  int ref_count, mut_ref_count;


  Owner(OpResult res) {
    this->result = res;
    this->ref_count = 0;
    this->mut_ref_count = 0;
  }
};

/// The BorrowCheckerPass is a FunctionPass that performs intra-procedural
/// shape inference.
///
///    Algorithm: (for each function scope)
///
///   1) If the destination of a result is not yet known, insert it
///      into the symbol table
///
class BorrowCheckerPass : public mlir::FunctionPass<BorrowCheckerPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    llvm::ScopedHashTable<StringRef, Owner *> symbolTable;
    ScopedHashTableScope<StringRef, Owner *> var_scope(symbolTable);

    printf("-- starting borrow checker --\n");

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      llvm::dbgs() << "Borrow checking " << op->getName() << "\n";

      auto dstattr = op->getAttrOfType<StringAttr>("dst");
      if (!dstattr || dstattr.getValue() == "")
        return;

      auto dst = dstattr.getValue();
      if (auto variable = symbolTable.lookup(dst)) {
        // next step
      } else {
        // There should only be one result
        Owner *ow = new Owner(op->getResult(0));
        assert(ow);
        llvm::dbgs() << " - inserting into symbol table: " << dst << "\n";
        symbolTable.insert(dst, ow);
      }
    });

    for (auto itr = symbolTable.begin("a"); itr != symbolTable.end(); itr++) {
      llvm::dbgs() << "Found symbol " << *itr << "\n";  
    }
  }
};
} // end anonymous namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::pinch::createBorrowCheckerPass() {
  return std::make_unique<BorrowCheckerPass>();
}

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
  // name of this owning variable
  StringRef name;
  // the result that this variable was assigned
  OpResult result;
  // if this is a reference, here is the data it refers to
  Owner *ref_to;
  // number of references leased out to our value
  int ref_count, mut_ref_count;


  Owner(StringRef n, OpResult res, Owner *rt) {
    this->name = n;
    this->result = res;
    this->ref_count = 0;
    this->mut_ref_count = 0;
    this->ref_to = rt;
  }

  Owner(StringRef n, OpResult res)
      : Owner(n, res, NULL)
  {}
};

/// The BorrowCheckerPass is a FunctionPass that performs intra-procedural
/// shape inference.
///
///    Algorithm: (for each function scope)
///
///   1) Check the source of the data for a given operation and
///      make sure it is acceptable
///
///
///   2) if the destination of a result is not yet known, insert it
///      into the symbol table
///
///   3) If the destination is known, record that we moved the data
///      there
///
class BorrowCheckerPass : public mlir::FunctionPass<BorrowCheckerPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    vector<Owner *> owners;
    llvm::ScopedHashTable<StringRef, Owner *> symbolTable;
    ScopedHashTableScope<StringRef, Owner *> var_scope(symbolTable);

    printf("-- starting borrow checker --\n");

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      llvm::dbgs() << "Borrow checking " << op->getName() << "\n";

      // Check the source first
      auto srcattr = op->getAttrOfType<StringAttr>("src");
      if (srcattr && srcattr.getValue() != "") {
        auto src = srcattr.getValue();

        // step 1
      }

      // check the destination
      auto dstattr = op->getAttrOfType<StringAttr>("dst");
      if (!dstattr || dstattr.getValue() == "")
        return;

      auto dst = dstattr.getValue();
      if (auto variable = symbolTable.lookup(dst)) {
        // record that the value was moved to dst

        // step 2

      } else {
        // This must be a newly active variable
        // There should only be one result
        Owner *ow = new Owner(dst, op->getResult(0));
        assert(ow);
        llvm::dbgs() << " - inserting into symbol table: " << dst << "\n";
        symbolTable.insert(dst, ow);
        owners.push_back(ow);
      }
    });

    for (auto itr = owners.begin(); itr != owners.end(); itr++) {
      Owner *ow = *itr;
      StringRef name = ow->name;
      llvm::dbgs() << "Found symbol " << name << "\n";
      delete ow;
    }
  }
};
} // end anonymous namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::pinch::createBorrowCheckerPass() {
  return std::make_unique<BorrowCheckerPass>();
}

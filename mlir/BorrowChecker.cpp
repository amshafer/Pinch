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

#define BORROW_CHECK(d, op, src, st)                               \
  if (!(d)->check((op), (src), (st))) { return signalPassFailure(); }

namespace {

enum OwType {
  u32 = 0,
  box,
  ref,
  mut,
};

enum OwType
owtype_from_type(Type ty)
{
  if (ty.isa<MemRefType>())
    return OwType::ref;
  else if (ty.isa<BoxType>())
    return OwType::box;
  else
    return OwType::u32;
}

/// Ownership information for a particular variable
class Owner {
public:
  // name of this owning variable
  StringRef name;
  // if this is a reference, here is the data it refers to
  Owner *ref_to;
  // number of references leased out to our value
  int ref_count, mut_ref_count;
  // Does this variable actually own anything?
  bool is_resident;
  OwType type;

  Owner(StringRef n, Owner *rt, OwType ty) {
    this->name = n;
    this->ref_count = 0;
    this->mut_ref_count = 0;
    this->ref_to = rt;
    this->is_resident = true;
    this->type = ty;
  }

  Owner(StringRef n)
      : Owner(n, NULL, OwType::u32)
  {}

  bool is_some_ref() {
    return this->type == OwType::ref
      || this->type == OwType::ref;
  }

  // This is a big state machine that checks the rules for
  // each type of operation. 'this' is the destination
  //
  // src may be NULL
  bool check(mlir::Operation *op, Owner *src,
             llvm::ScopedHashTable<StringRef, Owner *> &symbolTable) {
    if (op->getName().getStringRef().equals("pinch.borrow")) {
      assert(src);

      if (!src->is_resident) {
        op->emitError("Trying to borrow a reference to already moved variable");
        return false;
      }

      // fail if there is a mutable borrow already active
      if (src->mut_ref_count > 0) {
        op->emitError("Cannot borrow a shared reference while a "
                      "mutable reference is active");
        return false;
      }
      this->type = OwType::ref;
      src->ref_count++;
    } else if (op->getName().getStringRef().equals("pinch.borrow_mut")) {
      assert(src);

      if (!src->is_resident) {
        op->emitError("Trying to borrow a mutable reference to already moved variable");
        return false;
      }

      // we need to be the only reference
      if (src->mut_ref_count > 0 || src->ref_count > 0) {
        op->emitError("Cannot borrow a shared reference while a "
                      "mutable reference is active");
        return false;
      }
      this->type = OwType::mut;
      src->mut_ref_count++;
    } else if (op->getName().getStringRef().equals("pinch.deref")) {
      assert(src);

      if (!src->is_resident) {
        op->emitError("Trying to dereference already moved variable");
        return false;
      }
    }
    // else do nothing since it might be another dialect

    return true;
  }
};

/// The BorrowCheckerPass is a FunctionPass that performs static
/// checking of pointer rules
class BorrowCheckerPass : public PassWrapper<BorrowCheckerPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    vector<Owner *> owners;
    llvm::ScopedHashTable<StringRef, Owner *> symbolTable;
    ScopedHashTableScope<StringRef, Owner *> var_scope(symbolTable);

    // Add function arguments since they will not be previously known
    // these will be in the same order as the args are
    auto fsrcs = f.getAttrOfType<ArrayAttr>("src");
    if (fsrcs) {
      auto argsrcs = fsrcs.getValue();
      vector<StringRef> srcs;
      for (auto itr = argsrcs.begin(); itr != argsrcs.end(); itr++) {
        srcs.push_back((*itr).cast<StringAttr>().getValue());
      }

      int i = 0;
      for (auto itr = f.args_begin(); itr != f.args_end(); itr++) {

        // TODO check mutable reference args
        Owner *ow = new Owner(srcs[i], NULL,
                              owtype_from_type((*itr).getType()));
        symbolTable.insert(srcs[i], ow);
        i++;
      }
    }

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {

      auto srcattr = op->getAttrOfType<StringAttr>("src");
      // we need to handle return here since it needs access to the
      // symbol table
      if (op->getName().getStringRef().equals("pinch.return")
          && srcattr
          && srcattr.getValue() != "") {
        // check if src is in the symbol table
        auto srcown = symbolTable.lookup(srcattr.getValue());
        if (srcown) {
          // if it is some type of reference we will give a warning
          if (srcown->is_some_ref()) {
            op->emitError("returning reference to variable whose lifetime is ending");
            return signalPassFailure();
          }
        } else {
          op->emitError("Could not find return source " + srcattr.getValue());
          return signalPassFailure();
        }
      } else if (op->getName().getStringRef().equals("pinch.print")
                 || op->getName().getStringRef().equals("pinch.generic_call")
                 || op->getName().getStringRef().equals("pinch.add")
                 || op->getName().getStringRef().equals("pinch.mul")) {
        for (auto itr = op->operand_begin(); itr != op->operand_end(); itr++) {
          auto defop = (*itr).getDefiningOp();
          // if the defining op doesn't exist it must be an argument
          if (!defop)
            continue;

          // if the defining op is a box ignore it since we already tracked its
          // move
          if (defop->getResult(0).getType().isa<BoxType>())
            continue;

          auto dst = defop->getAttrOfType<StringAttr>("dst");

          // if dst is "" then it is a temp var, check src instead
          StringRef arg_src = dst.getValue();
          if (arg_src == "") {
            auto sa = (*itr).getDefiningOp()->getAttrOfType<StringAttr>("src");
            // if src is not found, it must be a constant temp
            if (sa)
              arg_src = sa.getValue();
          }

          // look it up in symbol table
          if (auto ow = symbolTable.lookup(arg_src)) {
            if (!ow->is_resident) {
              op->emitError("Trying to use value from already moved variable");
              return signalPassFailure();
            } else if (ow->mut_ref_count > 0) {
              op->emitError("Cannot move variable while a mutable reference has been loaned out");
              return signalPassFailure();
            }
          }
        }
      } else if (op->getName().getStringRef().equals("pinch.move")
                 && srcattr
                 && srcattr.getValue() != "") {
        if (auto ow = symbolTable.lookup(srcattr.getValue())) {
          
          if (!ow->is_resident) {
            op->emitError("Trying to move from already moved variable");
            return signalPassFailure();
          }

          if (ow->mut_ref_count > 0 || ow->ref_count > 0) {
            op->emitError("Invalid move after borrow");
            return signalPassFailure();
          }
          // mark the src as empty
          ow->is_resident = false;
        }
    }

      // check the destination
      auto dstattr = op->getAttrOfType<StringAttr>("dst");
      if (!dstattr || dstattr.getValue() == "")
        return;

      auto dst = dstattr.getValue();
      if (!symbolTable.lookup(dst)) {
        // This must be a newly active variable
        Owner *ow = new Owner(dst);
        assert(ow);
        symbolTable.insert(dst, ow);
        owners.push_back(ow);
      }

      auto dstown = symbolTable.lookup(dst);
      assert(dstown);

      // Check the source
      if (srcattr && srcattr.getValue() != "") {
        auto src = srcattr.getValue();

        auto srcown = symbolTable.lookup(src);
        if (srcown) {
          BORROW_CHECK(dstown, op, srcown, symbolTable);
        } else {
          op->emitError("owning src not found: " + src);
          return signalPassFailure();
        }
      } else {
        BORROW_CHECK(dstown, op, NULL, symbolTable);
      }
    });

    for (auto itr = owners.begin(); itr != owners.end(); itr++) {
      Owner *ow = *itr;
      delete ow;
    }
  }
};
} // end anonymous namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::pinch::createBorrowCheckerPass() {
  return std::make_unique<BorrowCheckerPass>();
}

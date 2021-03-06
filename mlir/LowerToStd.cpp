//====- LowerToStd.cpp - Lowering from Pinch+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Pinch operations to
// standard operations. This lowering expects that borrow checking
// has passed
//
//===----------------------------------------------------------------------===//
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PinchToStd RewritePatterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PinchToStd RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<pinch::ConstantOp> {
  using OpRewritePattern<pinch::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pinch::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    auto constantValue = op.value();

    auto indexAttr =
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    auto valueAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), constantValue);
    auto ptrType = mlir::MemRefType::get(llvm::makeArrayRef<int64_t>(1),
                                         rewriter.getIntegerType(32));

    Value vop = rewriter.create<ConstantOp>(op.getLoc(), valueAttr);
    Value iop = rewriter.create<ConstantOp>(op.getLoc(), indexAttr);

    // Replace this operation with the generated alloc.
    rewriter.replaceOpWithNewOp<AllocaOp>(op, ptrType);

    rewriter.create<StoreOp>(op.getLoc(), vop, op, iop);

    return success();
  }
};

struct BoxOpLowering : public OpRewritePattern<pinch::BoxOp> {
  using OpRewritePattern<pinch::BoxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pinch::BoxOp op,
                                PatternRewriter &rewriter) const final {
    auto constantValue = op.value();

    auto indexAttr =
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    auto valueAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), constantValue);
    auto ptrType = mlir::MemRefType::get(llvm::makeArrayRef<int64_t>(1),
                                         rewriter.getIntegerType(32));

    Value vop = rewriter.create<ConstantOp>(op.getLoc(), valueAttr);
    Value iop = rewriter.create<ConstantOp>(op.getLoc(), indexAttr);

    // Replace this operation with the generated alloc.
    rewriter.replaceOpWithNewOp<AllocOp>(op, ptrType);

    rewriter.create<StoreOp>(op.getLoc(), vop, op, iop);

    return success();
  }
};

struct DropOpLowering : public OpRewritePattern<pinch::DropOp> {
  using OpRewritePattern<pinch::DropOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pinch::DropOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<DeallocOp>(op, op.getOperand());

    return success();
  }
};

struct CastOpLowering : public OpRewritePattern<pinch::CastOp> {
  using OpRewritePattern<pinch::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pinch::CastOp op,
                                PatternRewriter &rewriter) const final {

    Value in = op.getOperand();
    if (in && in.getType().isa<MemRefType>()) {
      auto indexAttr =
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
      Value iop = rewriter.create<ConstantOp>(op.getLoc(), indexAttr);

      in = rewriter.create<LoadOp>(op.getLoc(), in, iop);
    }

    rewriter.replaceOp(op, in);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PinchToStd RewritePatterns: math operations
//===----------------------------------------------------------------------===//

namespace {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    // Generate an adaptor for the remapped operands of the BinaryOp. This
    // allows for using the nice named accessors that are generated by the
    // ODS.
    typename BinaryOp::OperandAdaptor binaryAdaptor(operands);

    // Generate loads for the element of 'lhs' and 'rhs' at the inner
    // loop.
    auto indexAttr =
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    Value iop = rewriter.create<ConstantOp>(loc, indexAttr);

    Value loadedLhs = binaryAdaptor.lhs();
    if (loadedLhs.getType().isa<MemRefType>())
      loadedLhs = rewriter.create<LoadOp>(loc, binaryAdaptor.lhs(), iop);
    Value loadedRhs = binaryAdaptor.rhs();
    if (loadedRhs.getType().isa<MemRefType>())
      loadedRhs = rewriter.create<LoadOp>(loc, binaryAdaptor.rhs(), iop);

    // Create the binary operation performed on the loaded values.
    Value res = rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);

    auto ptrType = mlir::MemRefType::get(llvm::makeArrayRef<int64_t>(1),
                                         rewriter.getIntegerType(32));
    Value stack = rewriter.create<AllocaOp>(loc, ptrType);
    rewriter.create<StoreOp>(loc, res, stack, iop);

    rewriter.replaceOp(op, stack);
    return success();
  }
};
}
using AddOpLowering = BinaryOpLowering<pinch::AddOp, AddIOp>;
using MulOpLowering = BinaryOpLowering<pinch::MulOp, MulIOp>;

/* -------- pointer lowering ----------- */
struct MoveOpLowering : public ConversionPattern {
  MoveOpLowering(MLIRContext *ctx)
      : ConversionPattern(pinch::MoveOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

struct BorrowOpLowering : public ConversionPattern {
  BorrowOpLowering(MLIRContext *ctx)
      : ConversionPattern(pinch::BorrowOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

struct BorrowMutOpLowering : public ConversionPattern {
  BorrowMutOpLowering(MLIRContext *ctx)
      : ConversionPattern(pinch::BorrowMutOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

struct DerefOpLowering : public ConversionPattern {
  DerefOpLowering(MLIRContext *ctx)
      : ConversionPattern(pinch::DerefOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (operands[0].getType().isa<MemRefType>()) {
      auto indexAttr =
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
      Value iop = rewriter.create<ConstantOp>(op->getLoc(), indexAttr);

      rewriter.replaceOpWithNewOp<LoadOp>(op, operands[0], iop);
    } else {
      rewriter.replaceOp(op, operands[0]);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PinchToStd RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<pinch::ReturnOp> {
  using OpRewritePattern<pinch::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pinch::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // If the op returns an i32, rewrite it to match the funcops u32
    auto fop = op.getParentOfType<FuncOp>();
    if (fop && fop.getCallableResults().size() > 0) {
      auto res = fop.getCallableResults()[0];
      if (res.isUnsignedInteger()) {
        auto argtypes = fop.getType().getInputs();
        std::vector<Type> nargtypes;
        // if any of the types are u32, change to i32
        for (size_t i = 0; i < argtypes.size(); i++) {
          nargtypes.push_back(argtypes.data()[i]);
          if (nargtypes[i].isUnsignedInteger()) {
            nargtypes[i] = rewriter.getIntegerType(32);
          }
        }

        auto nftype = rewriter.getFunctionType(llvm::makeArrayRef(nargtypes),
                                               rewriter.getIntegerType(32));
        fop.setType(nftype);
      }
    }

    // TODO: add results
    // We lower "pinch.return" directly to "std.return".
    if (op.getOperands().size() == 1) {
      Value in = op.getOperand(0);
      if (in
          && in.getType().isa<MemRefType>()
          && fop.getCallableResults().size() == 1
          && fop.getCallableResults()[0].isa<IntegerType>()) {
        auto indexAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
        Value iop = rewriter.create<ConstantOp>(op.getLoc(), indexAttr);

        in = rewriter.create<LoadOp>(op.getLoc(), in, iop);
      }
      rewriter.replaceOpWithNewOp<ReturnOp>(op, in);
    } else {
      rewriter.replaceOpWithNewOp<ReturnOp>(op, op.getOperands());
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// PinchToStdLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct PinchToStdLoweringPass
    : public PassWrapper<PinchToStdLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace

void PinchToStdLoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() == "main") {
    // Verify that the given main has no inputs and results.
    if (function.getNumArguments() || function.getType().getNumResults()) {
      function.emitError("expected 'main' to have 0 inputs and 0 results");
      return signalPassFailure();
    }
  } else {
    return;
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<StandardOpsDialect>();

  // We also define the Pinch dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Pinch operations that don't want
  // to lower, `pinch.print`, as `legal`.
  target.addIllegalDialect<pinch::PinchDialect>();
  target.addLegalOp<pinch::PrintOp>();

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `pinch`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, ConstantOpLowering, MulOpLowering,
                  ReturnOpLowering, MoveOpLowering,
                  BorrowOpLowering, BorrowMutOpLowering,
                  DerefOpLowering, CastOpLowering,
                  BoxOpLowering, DropOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Pinch` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::pinch::createLowerToStdPass() {
  return std::make_unique<PinchToStdLoweringPass>();
}

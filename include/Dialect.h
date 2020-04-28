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
#include "mlir/IR/StandardTypes.h"

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

  /// Parse an instance of a type registered to the toy dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the toy dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

/// Include the auto-generated header file containing the declarations of the
/// pinch operations.
#define GET_OP_CLASSES
#include "Ops.h.inc"

namespace PinchTypes {
enum Kinds {
  Box = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
}

struct BoxTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = mlir::MemRefType;

  /// The following field contains the element types of the struct.
  mlir::MemRefType memref;

  /// A constructor for the type storage instance.
  BoxTypeStorage(mlir::MemRefType memref)
      : memref(memref) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == memref; }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself, see the `StructType::get` method further below.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(mlir::MemRefType memref) {
    return KeyTy(memref);
  }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    /* WARNING: might cause issues? */
    return llvm::hash_value(key.getShape());
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static BoxTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    mlir::MemRefType memref = key;

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<BoxTypeStorage>())
      BoxTypeStorage(memref);
  }

};

class BoxType : public Type::TypeBase<BoxType, Type,
                                      BoxTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PinchTypes::Box; }

  static BoxType get(Location loc)
  {
    auto inttype =  mlir::IntegerType::getChecked(32,
                                                  mlir::IntegerType::SignednessSemantics::Unsigned,
                                                  loc);
    auto *ctx = inttype.getContext();
    auto mt = mlir::MemRefType::get(llvm::makeArrayRef<int64_t>(1), inttype);
    return Base::get(ctx, PinchTypes::Box, mt);
  }

  mlir::Type getElementType() {
    return getImpl()->memref.getElementType();
  }
};

} // end namespace pinch
} // end namespace mlir

#endif // MLIR_TUTORIAL_PINCH_DIALECT_H_

//===- Ops.td - Pinch dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Pinch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PINCH_OPS
#define PINCH_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpBase.td"

// Provide a definition of the 'pinch' dialect in the ODS framework so that we
// can define our operations.
def Pinch_Dialect : Dialect {
  let name = "pinch";
  let cppNamespace = "pinch";
}

// Base class for pinch dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Pinch_Op<string mnemonic, list<OpTrait> traits = []> :
Op<Pinch_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Pinch Types
//===----------------------------------------------------------------------===//

def RefType: MemRefOf<[AnyType]>;
def MutRefType: MemRefOf<[AnyType]>;
def BoxType : Type<CPred<"$_self.isa<BoxType>()">, "box type">;
def AnyRefType: AnyTypeOf<[BoxType, RefType, MutRefType]>;

def DataType: AnyTypeOf<[UI32]>;
def OwnedType: AnyTypeOf<[BoxType, DataType, AnyRefType]>;


//===----------------------------------------------------------------------===//
// Pinch Operations
//===----------------------------------------------------------------------===//

// We define a pinch operation by inheriting from our base 'Pinch_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Pinch_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins UI32Attr:$value, StrAttr:$dst);

  // The constant operation returns a single value of TensorType.
  let results = (outs OwnedType);

  // Specify a parser and printer method.
  let parser = [{ return ::parseConstantOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
    let builders = [
    // Build a constant with a given constant int
    OpBuilder<"Builder *builder, OperationState &state, StringRef dst, unsigned int value">,
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def BoxOp : Pinch_Op<"box"> {
  let summary = "heap allocation";
  let description = [{
      boxed heap allocation.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins UI32Attr:$value, StrAttr:$dst);
  let results = (outs BoxType);

  let builders = [
    // Build a constant with a given constant int
    OpBuilder<"Builder *builder, OperationState &state, StringRef dst, unsigned int value">,
  ];

    // Invoke a static verify method to verify this constant operation.
    let verifier = [{ return ::verify(*this); }];
}

def DropOp : Pinch_Op<"drop"> {
  let summary = "heap deallocation";
  let description = [{
      drop a heap allocated box
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins BoxType:$value);
}

def AddOp : Pinch_Op<"add"> {
  let summary = "addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two ints
  }];

  let arguments = (ins DataType:$lhs, DataType:$rhs, StrAttr:$dst);
  let results = (outs DataType);

  // Specify a parser and printer method.
  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs, StringRef dst">
  ];
}

def MulOp : Pinch_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DataType:$lhs, DataType:$rhs, StrAttr:$dst);
  let results = (outs DataType);

  // Specify a parser and printer method.
  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs, StringRef dst">
  ];
}

def BorrowOp : Pinch_Op<"borrow"> {
  let summary = "borrow a shared reference";
  let description = [{
    Borrow a shared reference to the SSA operand
  }];

  let arguments = (ins OwnedType:$data, StrAttr:$src, StrAttr:$dst);
  let results = (outs RefType);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  let builders = [
      OpBuilder<"Builder *b, OperationState &state, "
          "MemRefType memrefType, Value data, StringRef src, StringRef dst",[{
              state.types.push_back(memrefType);
              state.addOperands(data);
              state.addAttribute("src", b->getStringAttr(src));
              state.addAttribute("dst", b->getStringAttr(dst));
          }]>
  ];
}

def MoveOp : Pinch_Op<"move"> {
  let summary = "Move a value from one variable to another";
  let description = [{
      Move a value from src to dst, and store it in an SSA variable
  }];

  // ins StringAttr:$src, ins StringAttr:$dst
  let arguments = (ins OwnedType:$data, StrAttr:$src, StrAttr:$dst);
  let results = (outs OwnedType);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  let builders = [
      OpBuilder<"Builder *b, OperationState &state, "
          "Value data, StringRef src, StringRef dst",[{
              state.types.push_back(data.getType());
              state.addOperands(data);
              state.addAttribute("src", b->getStringAttr(src));
              state.addAttribute("dst", b->getStringAttr(dst));
          }]>
  ];
}

def BorrowMutOp : Pinch_Op<"borrow_mut"> {
  let summary = "borrow a mutable reference";
  let description = [{
    Borrow a mutable reference to the SSA operand
  }];

  let arguments = (ins OwnedType:$data, StrAttr:$src, StrAttr:$dst);
  let results = (outs RefType);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  let builders = [
      OpBuilder<"Builder *b, OperationState &state, "
          "MemRefType memrefType, Value data, StringRef src, StringRef dst",[{
              state.types.push_back(memrefType);
              state.addOperands(data);
              state.addAttribute("src", b->getStringAttr(src));
              state.addAttribute("dst", b->getStringAttr(dst));
          }]>
  ];
}

def DerefOp : Pinch_Op<"deref"> {
  let summary = "dereference a pointer";
  let description = [{
      Dereferences a borrow
  }];

  let arguments = (ins AnyRefType:$data, StrAttr:$src, StrAttr:$dst);
  let results = (outs OwnedType);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
    //let printer = [{ return ::printBinaryOp(p, *this); }];
}

def CastOp : Pinch_Op<"cast", [NoSideEffect]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins AnyTypeOf<[UI32,MemRefOf<[AnyType]>]>:$input);
  let results = (outs UI32:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  // Set the folder bit so that we can fold redundant cast operations.
  let hasFolder = 1;
}

def GenericCallOp : Pinch_Op<"generic_call",
                             [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = pinch.generic_call @my_func(%1, %3)
           : (u32, u32) : -> u32
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
    // callee, and inputs for the call.
    // TODO: make ref types
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<OwnedType>:$inputs, StrAttr:$dst);

  let results = (outs OwnedType);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments, StringRef dst">
  ];
}

def PrintOp : Pinch_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins AnyTypeOf<[OwnedType, I32]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReturnOp : Pinch_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> u32 {
        ...
        pinch.return %0 : u32
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    // TODO: add ref type
  let arguments = (ins Variadic<OwnedType>:$input, StrAttr:$src);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<
    "Builder *b, OperationState &state", [{ build(b, state, llvm::None, StringRef("")); }]
  >];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

#endif // PINCH_OPS

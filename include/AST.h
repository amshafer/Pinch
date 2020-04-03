//===- AST.h - Node definition for the Pinch AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Pinch language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_PINCH_AST_H_
#define MLIR_TUTORIAL_PINCH_AST_H_

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace pinch {

/// A variable type with shape information.
enum Type {
  null,
  u32,
};

struct VarType {
  Type type;
  bool is_ref;
};

  /// Base class for all expression nodes.
  class ExprAST {
  public:
    enum ExprASTKind {
      Expr_VarDecl,
      Expr_Return,
      Expr_Num,
      Expr_Literal,
      Expr_Var,
      Expr_VarRef,
      Expr_VarMutRef,
      Expr_BinOp,
      Expr_Call,
      Expr_Print,
    };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind; }

    const Location &loc() { return location; }

  private:
    const ExprASTKind kind;
    Location location;
  };

  /// A block-list of expressions.
  using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

  /// Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST {
    double Val;

  public:
  NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), Val(val) {}

    double getValue() { return Val; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
  };

  /// Expression class for a literal value.
  class LiteralExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> values;
    std::vector<int64_t> dims;

  public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

    llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
    llvm::ArrayRef<int64_t> getDims() { return dims; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
  };

  /// Expression class for referencing a variable, like "a".
  class VariableExprAST : public ExprAST {
    std::string name;
    VarType type;

  public:
    VariableExprAST(Location loc, VarType ty, llvm::StringRef name)
        : ExprAST(Expr_Var, loc), name(name) { type = ty; }

    llvm::StringRef getName() { return name; }
    const VarType &getType() { return type; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
  };
  /// Represents an expression taking a shared ref of a variable
  class VariableRefExprAST : public ExprAST {
    std::string name;

  public:
  VariableRefExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_VarRef, loc), name(name) {}

    llvm::StringRef getName() { return name; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarRef; }
  };
  /// Represents an expression taking a mutable ref of a variable
  class VariableMutRefExprAST : public ExprAST {
    std::string name;

  public:
  VariableMutRefExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_VarMutRef, loc), name(name) {}

    llvm::StringRef getName() { return name; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarMutRef; }
  };

  /// Expression class for defining a variable.
  class VarDeclExprAST : public ExprAST {
    std::string name;
    VarType type;
    std::unique_ptr<ExprAST> initVal;

  public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

    llvm::StringRef getName() { return name; }
    ExprAST *getInitVal() { return initVal.get(); }
    const VarType &getType() { return type; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
  };

  /// Expression class for a return operator.
  class ReturnExprAST : public ExprAST {
    llvm::Optional<std::unique_ptr<ExprAST>> expr;

  public:
  ReturnExprAST(Location loc, llvm::Optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

    llvm::Optional<ExprAST *> getExpr() {
      if (expr.hasValue())
        return expr->get();
      return llvm::None;
    }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
  };

  /// Expression class for a binary operator.
  class BinaryExprAST : public ExprAST {
    char op;
    std::unique_ptr<ExprAST> lhs, rhs;

  public:
    char getOp() { return op; }
    ExprAST *getLHS() { return lhs.get(); }
    ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, loc), op(Op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
  };

  /// Expression class for function calls.
  class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;

  public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, loc), callee(callee), args(std::move(args)) {}

    llvm::StringRef getCallee() { return callee; }
    llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
  };

  /// Expression class for builtin print calls.
  class PrintExprAST : public ExprAST {
    std::unique_ptr<ExprAST> arg;

  public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, loc), arg(std::move(arg)) {}

    ExprAST *getArg() { return arg.get(); }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
  };

  /// This class represents the "prototype" for a function, which captures its
  /// name, and its argument names (thus implicitly the number of arguments the
  /// function takes).
  class PrototypeAST {
    Location location;
    VarType return_type;
    std::string name;
    std::vector<std::unique_ptr<VariableExprAST>> args;

  public:
  PrototypeAST(Location location, const std::string &name,
               VarType rt,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(location), name(name), args(std::move(args)) {
    return_type = rt;
  }

    const Location &loc() { return location; }
    llvm::StringRef getName() const { return name; }
    const VarType &getReturnType() { return return_type; }
    llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
  };

  /// This class represents a function definition itself.
  class FunctionAST {
    std::unique_ptr<PrototypeAST> proto;
    std::unique_ptr<ExprASTList> body;

  public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
    PrototypeAST *getProto() { return proto.get(); }
    ExprASTList *getBody() { return body.get(); }
  };

  /// This class represents a list of functions to be processed together
  class ModuleAST {
    std::vector<FunctionAST> functions;

  public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

    auto begin() -> decltype(functions.begin()) { return functions.begin(); }
    auto end() -> decltype(functions.end()) { return functions.end(); }
  };

  void dump(ModuleAST &);

} // namespace pinch

#endif // MLIR_TUTORIAL_PINCH_AST_H_

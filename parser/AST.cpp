//===- AST.cpp - Helper for printing out the Pinch AST ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump for the Pinch language.
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace pinch;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(VariableExprAST *node);
  void dump(VariableRefExprAST *node);
  void dump(VariableMutRefExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
  mlir::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableRefExprAST,
            VariableMutRefExprAST, VariableExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &expr : *exprList)
    dump(expr.get());
  indent();
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *node) {
  INDENT();
  llvm::errs() << "var: " << node->getName() << ": ";
  dump(node->getType());
  llvm::errs() << loc(node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableMutRefExprAST *node) {
  INDENT();
  llvm::errs() << "var ref mut: " << node->getName() << " " << loc(node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableRefExprAST *node) {
  INDENT();
  llvm::errs() << "var ref: " << node->getName() << " " << loc(node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (node->getExpr().hasValue())
    return dump(*node->getExpr());
  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *node) {
  INDENT();
  llvm::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
  for (auto &arg : node->getArgs())
    dump(arg.get());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *node) {
  INDENT();
  llvm::errs() << "Print [ " << loc(node) << "\n";
  dump(node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type) {
  if (type.is_ref)
    llvm::errs() << "&";
  if (type.type == Type::u32)
    llvm::errs() << "u32";
  else if (type.type == Type::null)
    llvm::errs() << "void";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "'\n";
  indent();
  llvm::errs() << "Params: [";
  mlir::interleaveComma(node->getArgs(), llvm::errs(),
                        [](auto &arg) {
                          llvm::errs() << arg->getName() << ": ";
                          auto type = arg->getType();
                          if (type.is_ref)
                            llvm::errs() << "&";
                          if (type.type == Type::u32)
                            llvm::errs() << "u32";
                          else if (type.type == Type::null)
                            llvm::errs() << "void";
                        });
  llvm::errs() << "]'\n";
  llvm::errs() << "Return type: ";
  dump(node->getReturnType());
  llvm::errs() << loc(node) << "\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &f : *node)
    dump(&f);
}

namespace pinch {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace pinch

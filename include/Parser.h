//===- Parser.h - Pinch Language Parser -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Pinch language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_PINCH_PARSER_H
#define MLIR_TUTORIAL_PINCH_PARSER_H

#include "AST.h"
#include "Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

namespace pinch {

  /// This is a simple recursive parser for the Pinch language. It produces a well
  /// formed AST from a stream of Token supplied by the Lexer. No semantic checks
  /// or symbol resolution is performed. For example, variables are referenced by
  /// string and the code could reference an undeclared variable and the parsing
  /// succeeds.
  class Parser {
  public:
    /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

    /// Parse a full Module. A module is a list of function definitions.
    std::unique_ptr<ModuleAST> parseModule() {
      lexer.getNextToken(); // prime the lexer

      // Parse functions one at a time and accumulate in this vector.
      std::vector<FunctionAST> functions;
      while (auto f = parseDefinition()) {
        functions.push_back(std::move(*f));
        if (lexer.getCurToken() == tok_eof)
          break;
      }
      // If we didn't reach EOF, there was an error during parsing
      if (lexer.getCurToken() != tok_eof)
        return parseError<ModuleAST>("nothing", "at end of module");

      return std::make_unique<ModuleAST>(std::move(functions));
    }

  private:
    Lexer &lexer;

    /// Parse a return statement.
    /// return :== return ; | return expr ;
    std::unique_ptr<ReturnExprAST> parseReturn() {
      auto loc = lexer.getLastLocation();
      lexer.consume(tok_return);

      // return takes an optional argument
      llvm::Optional<std::unique_ptr<ExprAST>> expr;
      if (lexer.getCurToken() != ';') {
        expr = parseExpression();
        if (!expr)
          return nullptr;
      }
      return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
    }

    /// Parse a literal number.
    /// numberexpr ::= number
    std::unique_ptr<ExprAST> parseNumberExpr() {
      auto loc = lexer.getLastLocation();
      auto result =
          std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
      lexer.consume(tok_number);
      return std::move(result);
    }

    /// parenexpr ::= '(' expression ')'
    std::unique_ptr<ExprAST> parseParenExpr() {
      lexer.getNextToken(); // eat (.
      auto v = parseExpression();
      if (!v)
        return nullptr;

      if (lexer.getCurToken() != ')')
        return parseError<ExprAST>(")", "to close expression with parentheses");
      lexer.consume(Token(')'));
      return v;
    }

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression ')'
    std::unique_ptr<ExprAST> parseIdentifierExpr() {
      std::string name(lexer.getId());

      auto loc = lexer.getLastLocation();

      //TODO check for reference
      if (lexer.getCurToken() == tok_ref) {
        lexer.getNextToken(); // eat the current token
        return std::make_unique<VariableRefExprAST>(std::move(loc), name);
      }

      if (lexer.getCurToken() == '*') {
        lexer.getNextToken(); // eat the current token
        return std::make_unique<DerefExprAST>(std::move(loc), name);
      }

      if (lexer.getCurToken() == tok_mut) {
        lexer.getNextToken(); // eat the current token
        return std::make_unique<VariableMutRefExprAST>(std::move(loc), name);
      }

      lexer.getNextToken(); // eat identifier.

      // default vartype. We don't care about it since this is
      // just a reference
      // TODO: fixme
      VarType ty;
      ty.type = Type::null;
      ty.is_ref = false;
      if (lexer.getCurToken() != '(') // Simple variable ref.
        return std::make_unique<VariableExprAST>(std::move(loc), ty, name);

      // This is a function call.
      lexer.consume(Token('('));
      std::vector<std::unique_ptr<ExprAST>> args;
      if (lexer.getCurToken() != ')') {
        while (true) {
          if (auto arg = parseExpression())
            args.push_back(std::move(arg));
          else
            return nullptr;

          if (lexer.getCurToken() == ')')
            break;

          if (lexer.getCurToken() != ',')
            return parseError<ExprAST>(", or )", "in argument list");
          lexer.getNextToken();
        }
      }
      lexer.consume(Token(')'));

      // It can be a builtin call to print
      if (name == "print") {
        if (args.size() != 1)
          return parseError<ExprAST>("<single arg>", "as argument to print()");

        return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
      } else if (name == "box") {
        if (args.size() != 1)
          return parseError<ExprAST>("<single arg>", "as argument to box()");

        return std::make_unique<BoxExprAST>(std::move(loc), std::move(args[0]));
      }

      // Call to a user-defined function
      return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= tensorliteral
    std::unique_ptr<ExprAST> parsePrimary() {
      switch (lexer.getCurToken()) {
      default:
        llvm::errs() << "unknown token '" << lexer.getCurToken()
                     << "' when expecting an expression\n";
        return nullptr;
      case tok_identifier:
      case tok_ref:
      case tok_mut:
      case '*':
        return parseIdentifierExpr();
      case tok_number:
        return parseNumberExpr();
      case '(':
        return parseParenExpr();
      case ';':
        return nullptr;
      case '}':
        return nullptr;
      }
    }

    /// Recursively parse the right hand side of a binary expression, the ExprPrec
    /// argument indicates the precedence of the current binary operator.
    ///
    /// binoprhs ::= ('+' primary)*
    std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                           std::unique_ptr<ExprAST> lhs) {
      // If this is a binop, find its precedence.
      while (true) {
        int tokPrec = getTokPrecedence();

        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (tokPrec < exprPrec)
          return lhs;

        // Okay, we know this is a binop.
        int binOp = lexer.getCurToken();
        lexer.consume(Token(binOp));
        auto loc = lexer.getLastLocation();

        // Parse the primary expression after the binary operator.
        auto rhs = parsePrimary();
        if (!rhs)
          return parseError<ExprAST>("expression", "to complete binary operator");

        // If BinOp binds less tightly with rhs than the operator after rhs, let
        // the pending operator take rhs as its lhs.
        int nextPrec = getTokPrecedence();
        if (tokPrec < nextPrec) {
          rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
          if (!rhs)
            return nullptr;
        }

        // Merge lhs/RHS.
        lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                              std::move(lhs), std::move(rhs));
      }
    }

    /// expression::= primary binop rhs
    std::unique_ptr<ExprAST> parseExpression() {
      auto lhs = parsePrimary();
      if (!lhs)
        return nullptr;

      return parseBinOpRHS(0, std::move(lhs));
    }

    /// type ::= < shape_list >
    /// shape_list ::= num | num , shape_list
    std::unique_ptr<VarType> parseType() {
      auto type = std::make_unique<VarType>();

      if (lexer.getCurToken() == tok_u32) {
        type->type = Type::u32;
        type->is_ref = false;
      } else if (lexer.getCurToken() == tok_box) {
        type->type = Type::box;
        type->is_ref = false;
      } else if (lexer.getCurToken() == tok_ref
                 || lexer.getCurToken() == tok_mut) {
        if (lexer.getId() == "u32")
          type->type = Type::u32;
        else
          return parseError<VarType>("u32", "non u32 type");

        type->is_ref = true;
      } else {
        return parseError<VarType>("&/u32", "non u32 type");
      }

      lexer.getNextToken(); // eat >
      return type;
    }

    /// Parse a variable declaration, it starts with a `let` keyword followed by
    /// and identifier and an optional type (shape specification) before the
    /// initializer.
    /// decl ::= let identifier [ type ] = expr
    std::unique_ptr<VarDeclExprAST> parseDeclaration() {
      if (lexer.getCurToken() != tok_let)
        return parseError<VarDeclExprAST>("let", "to begin declaration");
      auto loc = lexer.getLastLocation();
      lexer.getNextToken(); // eat keyword let

      if (lexer.getCurToken() != tok_identifier)
        return parseError<VarDeclExprAST>("identified",
                                          "after 'var' declaration");
      std::string id(lexer.getId());
      lexer.getNextToken(); // eat id

      std::unique_ptr<VarType> type; // Type is optional, it can be inferred
      if (lexer.getCurToken() == ':') {
        type = parseType();
        if (!type)
          return nullptr;
      }

      if (!type)
        type = std::make_unique<VarType>();
      lexer.consume(Token('='));
      auto expr = parseExpression();
      return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                              std::move(*type), std::move(expr));
    }

    /// Parse a block: a list of expression separated by semicolons and wrapped in
    /// curly braces.
    ///
    /// block ::= { expression_list }
    /// expression_list ::= block_expr ; expression_list
    /// block_expr ::= decl | "return" | expr
    std::unique_ptr<ExprASTList> parseBlock() {
      if (lexer.getCurToken() != '{')
        return parseError<ExprASTList>("{", "to begin block");
      lexer.consume(Token('{'));

      auto exprList = std::make_unique<ExprASTList>();

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));

      while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
        if (lexer.getCurToken() == tok_let) {
          // Variable declaration
          auto varDecl = parseDeclaration();
          if (!varDecl)
            return nullptr;
          exprList->push_back(std::move(varDecl));
        } else if (lexer.getCurToken() == tok_return) {
          // Return statement
          auto ret = parseReturn();
          if (!ret)
            return nullptr;
          exprList->push_back(std::move(ret));
        } else {
          // General expression
          auto expr = parseExpression();
          if (!expr)
            return nullptr;
          exprList->push_back(std::move(expr));
        }
        // Ensure that elements are separated by a semicolon.
        if (lexer.getCurToken() != ';')
          return parseError<ExprASTList>(";", "after expression");

        // Ignore empty expressions: swallow sequences of semicolons.
        while (lexer.getCurToken() == ';')
          lexer.consume(Token(';'));
      }

      if (lexer.getCurToken() != '}')
        return parseError<ExprASTList>("}", "to close block");

      lexer.consume(Token('}'));
      return exprList;
    }

    /// prototype ::= def id '(' decl_list ')'
    /// decl_list ::= identifier | identifier, decl_list
    std::unique_ptr<PrototypeAST> parsePrototype() {
      auto loc = lexer.getLastLocation();

      if (lexer.getCurToken() != tok_fn)
        return parseError<PrototypeAST>("fn", "in prototype");
      lexer.consume(tok_fn);

      if (lexer.getCurToken() != tok_identifier)
        return parseError<PrototypeAST>("function name", "in prototype");

      std::string fnName(lexer.getId());
      lexer.consume(tok_identifier);

      if (lexer.getCurToken() != '(')
        return parseError<PrototypeAST>("(", "in prototype");
      lexer.consume(Token('('));

      std::vector<std::unique_ptr<VariableExprAST>> args;
      if (lexer.getCurToken() != ')') {
        do {
          std::string name(lexer.getId());
          auto loc = lexer.getLastLocation();
          lexer.consume(tok_identifier);

          if (lexer.getCurToken() != ':')
            return parseError<PrototypeAST>(
                "identifier", "':' expected, type not specified with");
          lexer.consume(tok_colon);

          VarType type = *parseType();

          auto decl = std::make_unique<VariableExprAST>(std::move(loc),
                                                        type,
                                                        name);
                                                        
          args.push_back(std::move(decl));
          if (lexer.getCurToken() != ',')
            break;
          lexer.consume(Token(','));
          if (lexer.getCurToken() != tok_identifier)
            return parseError<PrototypeAST>(
                "identifier", "after ',' in function parameter list");
        } while (true);
      }
      if (lexer.getCurToken() != ')')
        return parseError<PrototypeAST>("}", "to end function prototype");

      // success.
      lexer.consume(Token(')'));

      VarType type;
      type.type = Type::null;
      type.is_ref = false;

      // find out if there is a return type
      if (lexer.getCurToken() == tok_arrow) {
        lexer.getNextToken(); // eat ->

        type = *parseType();
      }

      return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                            type,
                                            std::move(args));
    }

    /// Parse a function definition, we expect a prototype initiated with the
    /// `def` keyword, followed by a block containing a list of expressions.
    ///
    /// definition ::= prototype block
    std::unique_ptr<FunctionAST> parseDefinition() {
      auto proto = parsePrototype();
      if (!proto)
        return nullptr;

      if (auto block = parseBlock())
        return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
      return nullptr;
    }

    /// Get the precedence of the pending binary operator token.
    int getTokPrecedence() {
      if (!isascii(lexer.getCurToken()))
        return -1;

      // 1 is lowest precedence.
      switch (static_cast<char>(lexer.getCurToken())) {
      case '-':
        return 20;
      case '+':
        return 20;
      case '*':
        return 40;
      default:
        return -1;
      }
    }

    /// Helper function to signal errors while parsing, it takes an argument
    /// indicating the expected token and another argument giving more context.
    /// Location is retrieved from the lexer to enrich the error message.
    template <typename R, typename T, typename U = const char *>
                                                       std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
      auto curToken = lexer.getCurToken();
      llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
      << lexer.getLastLocation().col << "): expected '" << expected
      << "' " << context << " but has Token " << curToken;
      if (isprint(curToken))
        llvm::errs() << " '" << (char)curToken << "'";
      llvm::errs() << "\n";
      return nullptr;
    }
  };

} // namespace pinch

#endif // MLIR_TUTORIAL_PINCH_PARSER_H

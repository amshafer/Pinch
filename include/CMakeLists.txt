set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
add_public_tablegen_target(PinchOpsIncGen)

set(LLVM_TARGET_DEFINITIONS BorrowChecker.td)
mlir_tablegen(BorrowChecker.h.inc -gen-op-interface-decls)
mlir_tablegen(BorrowChecker.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(BorrowCheckOpsIncGen)

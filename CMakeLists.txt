add_custom_target(Pinch)
set_target_properties(Pinch PROPERTIES FOLDER Examples)

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

macro(add_pinch name)
  add_dependencies(Pinch ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_pinch name)

add_pinch(pinch
  pinch.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToLLVM.cpp
  mlir/LowerToStd.cpp
  mlir/BorrowChecker.cpp

  DEPENDS
  PinchOpsIncGen
  BorrowCheckOpsIncGen
  )
  
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(pinch
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffects
    MLIRTargetLLVMIR
    MLIRTransforms
    )

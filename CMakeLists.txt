add_custom_target(Pinch)
set_target_properties(Pinch PROPERTIES FOLDER Examples)

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
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

  DEPENDS
  PinchOpsIncGen
  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(pinch
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffects
    MLIRTransforms)

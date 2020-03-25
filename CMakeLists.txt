add_custom_target(Pinch)
set_target_properties(Pinch PROPERTIES FOLDER Examples)

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
  )
include_directories(include/)
target_link_libraries(pinch
  PRIVATE
    MLIRSupport)

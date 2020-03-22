set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch1
  pinch.cpp
  parser/AST.cpp
  )
include_directories(include/)
target_link_libraries(pinch-ch1
  PRIVATE
    MLIRSupport)

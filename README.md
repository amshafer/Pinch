# Pinch

https://github.com/amshafer/pinch

A simple borrow-checked language built using LLVM's MLIR

Multi-level Intermediate Representation (MLIR) is an LLVM project
designed to provide insfrastructure for developing high level
languages. Languages create a MLIR "dialect" and lower it to LLVM IR
for code generation.

Pinch is a borrow checked language inspired by Rust. It supports very
few operations and isn't intended for any real use. It was completed
as a course project for my code optimization class.

# Running the Pinch Compiler

Once you have compiled LLVM, you should have a directory named
`build`. The `pinch` executable is the newly built compiler:

```
# see pinch -h for the full set of flags
$ ./build/bin/pinch ~/source.pinch -emit=mlir
```

The `-emit` flag controls what pass of the compiler should generate
output. Here are the different passes:
* `-emit=mlir` - generate Pinch MLIR dialect
* `-emit=mlir-borrow-checked` - generate Pinch MLIR, but generate
compiler errors if the borrow checker fails.
* `-emit=mlir-std` - partially lower to the Standard dialect
* `-emit=llvm` - lower to LLVM IR
* `-emit=jit` - execute the program

# Example Programs

Here's a couple example programs and their output. More programs can
be found in the `examples` directory.

## Basic Borrow Checking

```
fn bad_function() -> &u32 {
   let ret = 3;
   return &ret;
}

fn main() {
   let a = 2;
   let b = &mut a;
   let c = &a;

   let d = bad_function();
}
```

The MLIR generated from this is:

```
module {
  func @bad_function() -> memref<1xui32> attributes {src = [], sym_visibility = "private"} {
    %0 = pinch.constant {dst = "ret"} 3
    %1 = "pinch.borrow"(%0) {dst = "return", src = "ret"} : (ui32) -> memref<1xui32>
    pinch.return %1 : memref<1xui32> {src = "return"}
  }
  func @main() attributes {src = []} {
    %0 = pinch.constant {dst = "a"} 2
    %1 = "pinch.borrow_mut"(%0) {dst = "b", src = "a"} : (ui32) -> memref<1xui32>
    %2 = "pinch.borrow"(%0) {dst = "c", src = "a"} : (ui32) -> memref<1xui32>
    %3 = pinch.generic_call @bad_function() {dst = "d"} : () -> ui32
    pinch.return {src = ""}
  }
}
```
If we compile with `-emit=mlir-borrow-checked` we see the following
error:
```
loc("/home/ashafer/bin/test.pinch":5:4): error: returning reference to variable whose lifetime is ending
```

In this example we have a function `bad_function` which returns an
address to a stack variable, which is unsafe. We also have variable `a`
being borrowed invalidly in `main`. If a mutable reference is borrowed
then we cannot create any shared references. The compiler generates an
error in `bad_function` and exits before it can process this second error.


## Boxed Variables
In Rust Box is a special type which is a smart pointer to heap
allocated data. Pinch isn't as functional, but we do support boxes:

```
fn test2(a: Box) -> u32 {
   return 3;
}

fn test(a: Box) -> u32 {
   let b = *a;
   test2(a);
   print(b);
   return b;
}

fn main() {
   let a = box(3);
   test(a);
}
```

In this example we create a heap allocated integer, and move it into
some subroutines. The result is just a printed integer, not very
impressive. The MLIR output however is pretty interesting:

```
module {
  func @test2(%arg0: !pinch.box) -> ui32 attributes {src = ["a"], sym_visibility = "private"} {
    "pinch.drop"(%arg0) : (!pinch.box) -> ()
    %0 = pinch.constant {dst = "return"} 3
    pinch.return %0 : ui32 {src = ""}
  }
  func @test(%arg0: !pinch.box) -> ui32 attributes {src = ["a"], sym_visibility = "private"} {
    %0 = "pinch.deref"(%arg0) {dst = "b", src = "a"} : (!pinch.box) -> ui32
    %1 = "pinch.move"(%arg0) {dst = "", src = "a"} : (!pinch.box) -> !pinch.box
    %2 = pinch.generic_call @test2(%1) {dst = ""} : (!pinch.box) -> ui32
    pinch.print %0 : ui32
    pinch.return %0 : ui32 {src = "b"}
  }
  func @main() attributes {src = []} {
    %0 = "pinch.box"() {dst = "a", value = 3 : ui32} : () -> !pinch.box
    %1 = "pinch.move"(%0) {dst = "", src = "a"} : (!pinch.box) -> !pinch.box
    %2 = pinch.generic_call @test(%1) {dst = ""} : (!pinch.box) -> ui32
    pinch.return {src = ""}
  }
}
```

You can see that the boxed value isn't dropped until `test2` is
called. Based on the borrow checking rules related to moving variables
this is the final place the box is used. Both of the other functions
move the box, which is why they do not generate calls to the
`pinch.drop` operation.

# Compiling

This repository is designed to be dropped into the `examples` folder
in the MLIR project. You will need to grab the latest llvm project,
and add this directory to it.

Steps:
```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project/mlir/examples
$ git clone https://github.com/amshafer/pinch
```

You then need to add the following line to the `CMakeLists.txt` in
`mlir/examples`:
```
add_subdirectory(pinch)
```

At this point you can go ahead and build the llvm-project as
normal. Pinch will be picked up by the build system the same way the
Toy examples are.

See the README in llvm-project for steps to build LLVM.

# Example Output

Below are the expected results for all the examples in `examples` when
executed. If you would like to get the other IR outputs please consult
the flags to `-emit=` mentioned above.

### valid.pinch
```
ashafer@wolfgang:git/llvm-project % ./build/bin/pinch mlir/examples/pinch/examples/valid.pinch -emit=jit 
3 
```

### bad_return.pinch
```
ashafer@wolfgang:git/llvm-project % ./build/bin/pinch mlir/examples/pinch/examples/bad_return.pinch -emit=jit
loc("mlir/examples/pinch/examples/bad_return.pinch":5:4): error: returning reference to variable whose lifetime is ending
loc("mlir/examples/pinch/examples/bad_return.pinch":11:12): error: Cannot borrow a shared reference while a mutable reference is active
```

### box.pinch
```
ashafer@wolfgang:git/llvm-project % ./build/bin/pinch mlir/examples/pinch/examples/box.pinch -emit=jit 
3 
```
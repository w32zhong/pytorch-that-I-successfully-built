## Module dependencies
![](./dep.png)

## Build
```sh
$ git submodule sync
$ git submodule update --init --recursive
 
$ conda create -y -n pytorch-study python=3.9
$ conda activate pytorch-study
 
$ conda install -y cmake ninja
$ pip install -r requirements.txt

$ conda install -y mkl mkl-include
$ conda install -y -c pytorch magma-cuda121

$ conda install -y gcc_linux-64=11.2.0
$ conda install -y gxx_linux-64=11.2.0
$ echo $PATH | tr ":" "\n"
/home/tk/anaconda3/envs/pytorch-study/bin
/usr/bin
...
$ # there is a bug in cmake that calls `gcc' directly.
$ # here we need to make sure the conda-version gcc is used.
$ # because the above output indicates conda-PATH is still
$ # prioritized than the /usr/bin, we can do:

$ cd $CONDA_PREFIX/bin
$ ln -s x86_64-conda-linux-gnu-gcc gcc
$ ln -s x86_64-conda-linux-gnu-g++ g++
$ cd -

$ nvidia-smi | grep CUDA
NVIDIA-SMI 535.54.03
Driver Version: 535.54.03
CUDA Version: 12.2
$ conda install -y -c nvidia/label/cuda-12.1.1 cuda
$ env | grep CUDA
CUDA_PATH=/opt/cuda
$ unset CUDA_PATH

$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
/home/tk/anaconda3/envs/pytorch-study
/home/tk/anaconda3/envs/pytorch-study/x86_64-conda-linux-gnu/sysroot/usr
$ python setup.py develop
```

Optionally, delete the `easy_install` link so that we do not mistake the local torch package:
```sh
$ rm -f $CONDA_PREFIX/lib/python3.9/site-packages/easy-install.pth
```

Show installed Pytorch information:
```sh
$ ls torch/lib
cmake                        libjitbackend_test.so  libtensorpipe.a
libXNNPACK.a                 libkineto.a            libtensorpipe_cuda.a
libasmjit.a                  libnnpack.a            libtensorpipe_uv.a
libbackend_with_compiler.so  libprotobuf-lite.a     libtorch.so
libc10.so                    libprotobuf.a          libtorch_cpu.so
libc10_cuda.so               libprotoc.a            libtorch_cuda.so
libc10d_cuda_test.so         libpthreadpool.a       libtorch_cuda_linalg.so
libcaffe2_nvrtc.so           libpytorch_qnnpack.a   libtorch_global_deps.so
libclog.a                    libqnnpack.a           libtorch_python.so
libcpuinfo.a                 libshm                 libtorchbind_test.so
libdnnl.a                    libshm.so              pkgconfig
libfbgemm.a                  libshm_windows         python3.9
libfmt.a                     libsleef.a

$ python -c 'import torch; print(torch.cuda.is_available())'
True
$ python -c 'import torch; print(torch.version.cuda)'
12.1
$ python -c 'import torch; print(torch.__version__)'
2.2.0a0+git**a5dd6de**
$ git show --quiet HEAD
commit **a5dd6de**9e7c3e7c33887fb8ee845ba97024a0fe7 (HEAD -> main, origin/main, origin/HEAD)
Author: w32zhong <clock126@126.com>
Date:   Wed Nov 15 22:02:24 2023 -0500

    update README
```

Good to know: the setup.py will generate Ninja build files as long as the ninja command is installed
(see [this](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/fec8db5927af25b99da9ddc6a2343f0893ef7bcb/tools/setup_helpers/cmake.py#L31)).

## Quick partial rebuild
At top leve:
```sh
cd torch
ln -sf ../build/lib.linux-x86_64-cpython-39/torch/_C.cpython-39-x86_64-linux-gnu.so .
cd -
cmake --build ./build --target install --config Release
```

## Quick test
For cmake:
```sh
cd hello-world
mkdir -p build
cd build && cmake -G Ninja ..
cmake --build .  --target all # or ninja all
```

For pytorch:
```sh
$ python hello-world.py
tensor([0.8506], grad_fn=<SigmoidBackward0>)
tensor(0.0223, grad_fn=<MseLossBackward0>)
```

## Verbose build
By directly invoke cmake with `--verbose` option:
```sh
cd build
cmake --trace . | tee trace.log
cmake --build . --target install --config Release --verbose | tee build.log
```
you can see all build commands.

See [build-stage1.log](./build-stage1.log) and [build-stage2.log](./build-stage2.log) for my build logs.

You can also extract the building structure:
```sh
python extract_build_structure.py | tee build-struct.log
```
the output is saved in [build-struct.log](./build-struct.log).
Another way is to use the `build/compile_commands.json` generated by the build system.

The output can be used to draw a module dependency graph as shown at the top.

## Redo Python package building
To redo the whole process of Python package building (without compiling dependencies like Caffe2 etc.):
```sh
$ find . -name '_C.*.so' | xargs rm -f
$ python setup.py develop
...
copying functorch/functorch.so -> build/lib.linux-x86_64-cpython-39/functorch/_C.cpython-39-x86_64-linux-gnu.so
building 'torch._C' extension
gcc -c torch/csrc/stub.c -o build/temp.linux-x86_64-cpython-39/torch/csrc/stub.o
gcc build/temp.linux-x86_64-cpython-39/torch/csrc/stub.o -L torch/lib -ltorch_python.so -o build/lib.linux-x86_64-cpython-39/torch/_C.cpython-39-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-cpython-39/torch/_C.cpython-39-x86_64-linux-gnu.so -> torch
copying build/lib.linux-x86_64-cpython-39/functorch/_C.cpython-39-x86_64-linux-gnu.so -> functorch
...
$ find . -name '_C.*.so'
./build/lib.linux-x86_64-cpython-39/functorch/_C.cpython-39-x86_64-linux-gnu.so 
./build/lib.linux-x86_64-cpython-39/torch/_C.cpython-39-x86_64-linux-gnu.so
./torch/_C.cpython-39-x86_64-linux-gnu.so
./functorch/_C.cpython-39-x86_64-linux-gnu.so
```

Now we know `_C.cpython-39-x86_64-linux-gnu.so` is just a stub.o (which calls `initModule`) plus `libtorch_python.so`.
We can verify `libtorch_python.so` has defined the `initModule`:
```sh
$ nm --defined-only build/lib/libtorch_python.so | grep initModule
000000000073cdc0 T _Z20THPEngine_initModuleP7_object
0000000000741040 T _Z22THPFunction_initModuleP7_object
000000000076d720 T _Z22THPVariable_initModuleP7_object
0000000000b6bf70 T _ZN5torch3cpu10initModuleEP7_object
0000000000b8a0d0 T _ZN5torch4cuda10initModuleEP7_object
00000000006afa80 T initModule
```
(`initModule` is defined in `torch/csrc/Module.cpp`)

If compiled with debug information, we can easily find out the source file:
```
$ nm -Cl --defined-only libtorch_cpu.so | grep "at::_ops::empty_memory_format::call"                                                                                           
00000000020950be T at::_ops::empty_memory_format::call(c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)       /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/build/aten/src/ATen/Operators_2.cpp:3231
```

## Debug and trace code
To know which function is defined at which source code file, we need to build [with debug information](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#tips-and-debugging):
```
CUDA_DEVICE_DEBUG=1 DEBUG=1 REL_WITH_DEB_INFO=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_CAFFE2=0 BUILD_TEST=0 USE_NNPACK=0 USE_XNNPACK=0 USE_QNNPACK=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 python setup.py develop --verbose
```
and now you can see debug sections in ELF and see symbol table with source paths:
```
$ readelf -S build/lib/libtorch_python.so | grep debug
  [28] .debug_aranges    PROGBITS         0000000000000000  01a6a2f0
  [29] .debug_info       PROGBITS         0000000000000000  021ca0b0
  [30] .debug_abbrev     PROGBITS         0000000000000000  0ccf5dad
  [31] .debug_line       PROGBITS         0000000000000000  0ce883d4
  [32] .debug_str        PROGBITS         0000000000000000  0e3c1686
  [33] .debug_line_str   PROGBITS         0000000000000000  111d5e0d
  [34] .debug_loclists   PROGBITS         0000000000000000  111e6c43
  [35] .debug_rnglists   PROGBITS         0000000000000000  111f549c

$ nm -C -D -l -g build/lib/libtorch_python.so | grep "initModule"                                                                                                       
00000000008f5e83 T THPEngine_initModule(_object*)       /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/autograd/python_engine.cpp:475
000000000090655a T THPFunction_initModule(_object*)     /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/autograd/python_function.cpp:1600
0000000000943204 T THPVariable_initModule(_object*)     /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/autograd/python_variable.cpp:2197
00000000010d25b8 T torch::cpu::initModule(_object*)     /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/cpu/Module.cpp:8
00000000010f50f1 T torch::cuda::initModule(_object*)    /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/cuda/Module.cpp:1533
00000000007b21c7 T initModule   /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/Module.cpp:1346
```

And to breakpoint at a C level function in gdb:
```sh
$ gdb python
GNU gdb (GDB) 13.1
Copyright (C) 2023 Free Software Foundation, Inc.
...
(gdb) b initModule
Function "initModule" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (initModule) pending.
(gdb) run
Starting program: /home/tk/anaconda3/envs/pytorch-study/bin/python
This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.archlinux.org>
Enable debuginfod for this session? (y or [n]) y
Python 3.9.18 (main, Sep 11 2023, 13:41:44)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Breakpoint 1.1, initModule ()
    at /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/Module.cpp:1346
1346    PyObject* initModule() {
```

Alternatively, in one command:
```sh
gdb -ex "b initModule" -ex run --args python hello-world.py
```

and to add a breakpoint at `file:line`:
```sh
gdb -ex "b library.cpp:228" -ex run python
```

To debug a already running python:
```sh
$ python
Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os; print(os.getpid())
2031925
>>>
```
in another terminal:
```sh
$ sudo bash -c 'echo 0 > /proc/sys/kernel/yama/ptrace_scope'
$ gdb -p 2031925 python
...
(gdb) break THPVariable_tensor
Function "THPVariable_tensor" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (THPVariable_tensor) pending.
(gdb) c
Continuing.
```
then back in the runnig python:
```sh
>>> a=torch.tensor([1])
```
this will triger gdb to show
```
Thread 1 "python" hit Breakpoint 1, torch::autograd::THPVariable_tensor (self=0x0, 
    args=0x7f279229e6a0, kwargs=0x0)
    at /home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/torch/csrc/autograd/python_torch_functions_manual.cpp:248
248         PyObject* kwargs) {
```

## Build internal
After setup, one can use the following command to trace the cmake execution.
```sh
cd build
cmake .. --trace-expand &> trace.log
```
Here are some of the important cmake files:
* [CMakeLists.txt](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/70c404d0a090463e3fac01346dacef18550c40e1/CMakeLists.txt)
  * `add_subdirectory(caffe2)`
  * `include(cmake/public/utils.cmake)`
  * `include(cmake/Dependencies.cmake)`
  * `include_directories(BEFORE ${PROJECT_SOURCE_DIR}/aten/src/)`
* [cmake/public/utils.cmake](https://github.com/pytorch/pytorch/blob/c47d2b80355db2120a591f21df494bdacff5ef30/cmake/public/utils.cmake#L221)
  * `macro(caffe2_interface_library SRC DST)` (think it as `add_dependencies(${DST} ${SRC})`)
* [cmake/Codegen.cmake](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/bc54c0ee378ab481040778d6e11d48afbe714c4b/cmake/Codegen.cmake#L352)
  * `function(append_filelist name outputvar)` that reads source files from [build_variables.bzl](./build_variables.bzl)
* [caffe2/CMakeLists.txt](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/70c404d0a090463e3fac01346dacef18550c40e1/caffe2/CMakeLists.txt)
  * `add_library(torch ${DUMMY_EMPTY_FILE})`
  * `add_subdirectory(../aten aten)`
  * `target_link_libraries(torch PUBLIC torch_cuda_library)`: this is the place most of dependencies (expanded from `torch_cuda_library`) are attached to `libtorch.so`
  * `caffe2_interface_library(torch_cuda torch_cuda_library)`: `torch_cuda_library` => `torch_cuda`
  * `add_subdirectory(../torch torch)`
  * `add_library(torch_cuda ${Caffe2_GPU_SRCS} ${Caffe2_GPU_CU_SRCS})`: `torch_cuda` => a lot of source files.
  * `list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_CPP_SRCS})` and `list(APPEND Caffe2_GPU_CU_SRCS ${ATen_CUDA_CU_SRCS})`
* [aten/CMakeLists.txt](https://github.com/pytorch/pytorch/blob/e7326ec295559c16795088e79a5631e784bb4d61/aten/CMakeLists.txt)
  * `set(ATen_CUDA_CPP_SRCS ${ATen_CUDA_CPP_SRCS} PARENT_SCOPE)`
* [aten/src/ATen/CMakeLists.txt](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/1b34089a4c42f546f6331d30ffad20aa9549a7e7/aten/src/ATen/CMakeLists.txt)
  * `list(APPEND ATen_CUDA_CPP_SRCS ${cuda_cpp} ${native_cuda_cpp} ...)` 
* [torch/CMakeLists.txt](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/70c404d0a090463e3fac01346dacef18550c40e1/torch/CMakeLists.txt)
  * `add_dependencies(torch_python gen_torch_version)`, meaning that `libtorch_python.so` depends on `gen_torch_version`.
  * `add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})`, similarly, `libtorch_python.so` depends on `${TORCH_PYTHON_SRCS}` which can be expanded by a simple Python line (see below).
  * `add_dependencies(torch_python torch_python_stubs)`
  * `add_dependencies(torch_python generate-torch-sources)`
  * `target_link_libraries(torch_python PRIVATE torch_library ${TORCH_PYTHON_LINK_LIBRARIES})` where `${TORCH_PYTHON_LINK_LIBRARIES}` depends on `ATEN_CPU_FILES_GEN_LIB`

For lines like `append_filelist("libtorch_python_core_sources" TORCH_PYTHON_SRCS)`, we can repreduce the variable being set here, i.e., `TORCH_PYTHON_SRCS`:
```sh
$ python
Python 3.8.16 (default, Mar  2 2023, 03:21:46)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> exec(open('build_variables.bzl').read())
>>> for dep in libtorch_python_core_sources:
...     print(dep)
...
torch/csrc/DataLoader.cpp
torch/csrc/Device.cpp
torch/csrc/Dtype.cpp
torch/csrc/DynamicTypes.cpp
torch/csrc/Exceptions.cpp
torch/csrc/Generator.cpp
torch/csrc/Layout.cpp
torch/csrc/MemoryFormat.cpp
torch/csrc/QScheme.cpp
torch/csrc/Module.cpp
torch/csrc/PyInterpreter.cpp
torch/csrc/python_dimname.cpp
torch/csrc/Size.cpp
torch/csrc/Storage.cpp
torch/csrc/StorageMethods.cpp
torch/csrc/StorageSharing.cpp
torch/csrc/Stream.cpp
torch/csrc/TypeInfo.cpp
torch/csrc/api/src/python/init.cpp
torch/csrc/autograd/functions/init.cpp
torch/csrc/autograd/init.cpp
...
```

We can double check under the cmake debug flag:
```sh
cmake -DPRINT_CMAKE_DEBUG_INFO=1 ..
```

By utilizing ninja, we can browse dependency clearly on browser. For build target `torch_python`:
```sh
# assume we are still in ./build here.
ninja -t browse -p 8080 torch_python
```

## Source internal

### Operator registration
Taken `tensor.empty()` operator as an example here.
```c
// torch/library.h
#define TORCH_LIBRARY(ns, m)                                                   \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);                        \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF,                                                     \
      &TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      c10::nullopt,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)

#define TORCH_LIBRARY_IMPL(ns, k, m, uid)                                 \
  static void C10_CONCATENATE(                                            \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);       \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(           \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      torch::Library::IMPL,                                               \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)       \
           ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid) \
           : [](torch::Library&) -> void {}),                             \
      #ns,                                                                \
      c10::make_optional(c10::DispatchKey::k),                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void C10_CONCATENATE(                                                   \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

// build/aten/src/ATen/RegisterSchema.cpp
TORCH_LIBRARY(aten, m) {
  m.def("empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor", {at::Tag::core, at::Tag::pt2_compliant_tag});
}

// ./build/aten/src/ATen/RegisterCPU.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m, 123) {
  m.impl("empty.memory_format", TORCH_FN(wrapper_CPU_memory_format_empty));
}
// similarly in ./build/aten/src/ATen/RegisterCUDA.cpp
// similarly in ./build/aten/src/ATen/RegisterMkldnnCPU.cpp
// fallback in aten/src/ATen/ConjugateFallback.cpp, for example:
TORCH_LIBRARY_IMPL(aten, Conjugate, m, 124) {
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough());
}
```

After macro expansion:
```c
// build/aten/src/ATen/RegisterSchema.cpp
static void TORCH_LIBRARY_init_aten(torch::Library&);
static const torch::detail::TorchLibraryInit
    TORCH_LIBRARY_static_init_aten(
        torch::Library::DEF,
        &TORCH_LIBRARY_init_aten,
        "aten",
        c10::nullopt,
        "filename.cpp",
        4321
    );
void TORCH_LIBRARY_init_aten(torch::Library& m) {
    m.def("empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor", {at::Tag::core, at::Tag::pt2_compliant_tag});
}

// taken ./build/aten/src/ATen/RegisterCPU.cpp as an example:
static void TORCH_LIBRARY_IMPL_init_aten_Conjugate_123(torch::Library&);
static const torch::detail::TorchLibraryInit
    TORCH_LIBRARY_IMPL_static_init_aten_Conjugate_123(
        torch::Library::IMPL,
        (
            c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::CPU)?
            &TORCH_LIBRARY_IMPL_init_aten_Conjugate_123 : [](torch::Library&) -> void {}
        ),
        "aten",
        c10::make_optional(c10::DispatchKey::CPU),
        "filename.cpp",
        1234
    );
void TORCH_LIBRARY_IMPL_init_aten_Conjugate_123(torch::Library &m) {
    m.impl("empty.memory_format", torch::CppFunction::makeFallthrough());
}
```

The torch::detail::TorchLibraryInit and Library classes:
```c
// torch/library.h
namespace detail {
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:

  // the constructor initializes the member `lib_` and call `fn` right away.
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      c10::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};
} // namespace detail

class TORCH_API Library final {
 public:
  enum Kind {
    DEF,
    IMPL,
    FRAGMENT,
  }

 // ...

 private:
  Kind kind_;
  c10::optional<std::string> ns_;
  c10::optional<c10::DispatchKey> dispatch_key_;
  const char* file_;
  uint32_t line_;
};

// aten/src/ATen/core/library.cpp 
Library::Library(Kind kind, std::string ns, c10::optional<c10::DispatchKey> k, const char* file, uint32_t line)
  : kind_(kind)
  , ns_(ns == "_" ? c10::nullopt : c10::make_optional(std::move(ns)))
  , dispatch_key_(k.value_or(CatchAll) == CatchAll ? c10::nullopt : k)
  , file_(file)
  , line_(line)
  {
    switch (kind_) {
      case DEF:
        // Only DEFs require library uniqueness; fragments
        // don't register a library
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerLibrary(
            *ns_, debugString(file_, line_)
          )
        );
        [[fallthrough]];
      case IMPL:
        // Nothing to do, everything is OK
        break;
    }
  }
```
Basically, for
* DEF: register a new `lib_` in Dispatcher::singleton(), and call `TORCH_LIBRARY_init_aten(lib_)` in which `m.def(...)` is called.
* IMPL: only call `TORCH_LIBRARY_IMPL_init_aten_Conjugate_123(lib_)` in which `m.impl(...)` is called.

For `m.def()`, it parses the schema and call `_def` to set `table[op] = schema`:
```c
// torch/library.h
class TORCH_API Library final {
  // ...
  template <typename Schema>
  Library& def(
      Schema&& raw_schema,
      const std::vector<at::Tag>& tags = {},
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema)); // step 1
    return _def(std::move(s), nullptr, tags, rv);
  }
};

inline c10::FunctionSchema schema(const char* str) {
  c10::FunctionSchema s = torch::jit::parseSchema(str); // step 2 (parse)
  s.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return s;
}

// aten/src/ATen/core/library.cpp
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name, const std::vector<at::Tag>& tags, _RegisterOrVerify rv) & {
  auto ns_opt = schema.getNamespace();
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema), // step 3
          debugString(file_, line_),
          tags
        )
      );
      break;
  }
  return *this;
}

// ./aten/src/ATen/core/dispatch/Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  // step 4 (actual register)
  // think this as table[op_name] = schema
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  listeners_->callOnOperatorRegistered(op);

  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;
  //...
}
```

For `m.impl()`, recall one of its implementation on CPU backend is
```c
m.impl("empty.memory_format", TORCH_FN(wrapper_CPU_memory_format_empty));
```
and `m.impl()` is just a wrapper on `_impl`, and the latter calls `registerKernel(...)`:
```c
// torch/library.h
class TORCH_API Library final {
  template <typename Name, typename Func>
  Library& impl(
      Name name,
      Func&& raw_f,
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    CppFunction f(std::forward<Func>(raw_f));
    return _impl(name, std::move(f), rv);
  }
};

// aten/src/ATen/core/library.cpp
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName op_name = _parseNameForLib(name_str);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(op_name),
          dispatch_key,
          std::move(f.func_),
          f.cpp_signature_,
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
  }
  return *this;
}

// aten/src/ATen/core/dispatch/Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;
  // ...
}
```

### Tensor allocation
For a tensor allocation like:
```py
a=torch.tensor([1])
```

The code to run:
```c
// torch/csrc/Module.cpp
PyObject* initModule() {
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()
  };
  module = PyModule_Create(&torchmodule);
  ASSERT_TRUE(THPVariable_initModule(module));
}

// ./torch/csrc/autograd/python_variable.cpp
bool THPVariable_initModule(PyObject* module) {
  torch::autograd::initTorchFunctions(module);
}

// ./torch/csrc/autograd/python_torch_functions_manual.cpp
void initTorchFunctions(PyObject* module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions); // gather common functions like tensor to torch_functions
  THPVariableFunctions.tp_methods = torch_functions.data();
  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule);
}
```
where `_C._VariableFunctions` are extracted to `torch.*` in `torch/__init__.py`:
```py
# torch/__init__.py
for name in dir(_C._VariableFunctions):
    obj = getattr(_C._VariableFunctions, name)
    obj.__module__ = 'torch'
    if not name.startswith("_"):
        __all__.append(name)
```

And in `gatherTorchFunctions`:
```c
// ./torch/csrc/autograd/python_torch_functions_manual.cpp
static PyMethodDef torch_functions_manual[] = {
    {"asarray",
     castPyCFunctionWithKeywords(THPVariable_asarray),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
     // ...
    {"tensor",
     castPyCFunctionWithKeywords(THPVariable_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
};
void gatherTorchFunctions(std::vector<PyMethodDef>& torch_functions) {
  constexpr size_t num_functions =
      sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  torch_functions.assign(
      torch_functions_manual, torch_functions_manual + num_functions);
}

// ./torch/csrc/autograd/python_torch_functions_manual.cpp
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {

  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
}

// ./torch/csrc/utils/tensor_new.cpp
Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {

    PyObject* data = r.pyobject(0);
    bool type_inference = r.isNone(1);
    bool pin_memory = r.toBool(3);
    bool args_requires_grad = r.toBool(4);
    auto new_tensor = internal_new_from_data(
        typeIdWithDefault(r, 2, dispatch_key),
        r.scalartypeWithDefault(1, scalar_type),
        r.deviceOptional(2),
        data,
        /*copy_variables=*/true,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference,
        pin_memory);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
}

Tensor internal_new_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<Device> device_opt,
    PyObject* data,
    bool copy_variables,
    bool copy_numpy,
    bool type_inference,
    bool pin_memory = false) {

  auto device = device_opt.has_value() ? *device_opt : options.device();
  auto sizes = compute_sizes(data, scalar_type);
  ScalarType inferred_scalar_type =
      type_inference ? infer_scalar_type(data) : scalar_type;
  Tensor tensor;
  {
    tensor = at::empty(sizes, opts.pinned_memory(pin_memory));
    recursive_store(
      (char*)tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      0,
      inferred_scalar_type,
      tensor.dtype().itemsize(),
      data);
    tensor = tensor.to(device, inferred_scalar_type);
  }

  return at::lift_fresh(tensor);
}

void recursive_store(
    char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize,
    PyObject* obj) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  bool is_symfloat = torch::is_symfloat(obj);
  bool is_symint = torch::is_symint(obj);
  if (dim == ndim) {
    if (is_symfloat) {
      auto new_obj = py::reinterpret_borrow<py::object>(obj);
      auto val = new_obj.cast<c10::SymFloat>();
      switch (elementSize) {
        case 8:
          *reinterpret_cast<double*>(data) = val;
          break;
        case 4:
          *reinterpret_cast<float*>(data) = static_cast<float>(val);
          break;
      }
      return;
    }
    if (is_symint) {
      // ...
    }
    torch::utils::store_scalar(data, scalarType, obj);
    return;
  }

  auto n = sizes[dim];
  auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
  PyObject** items = PySequence_Fast_ITEMS(seq.get());
  for (const auto i : c10::irange(n)) {
    recursive_store(
        data, sizes, strides, dim + 1, scalarType, elementSize, items[i]);
    data += strides[dim] * elementSize;
  }
}

// ./torch/csrc/utils/python_scalars.h
inline void store_scalar(void* data, at::ScalarType scalarType, PyObject* obj) {
  switch (scalarType) {
    case at::kByte:
      *(uint8_t*)data = unpackIntegral<uint8_t>(obj, "uint8");
      break;
    case at::kInt:
      *(int32_t*)data = unpackIntegral<int32_t>(obj, "int32");
      break;
    case at::kHalf:
      *(at::Half*)data =
          at::convert<at::Half, double>(THPUtils_unpackDouble(obj));
      break;
    case at::kFloat:
      *(float*)data = (float)THPUtils_unpackDouble(obj);
      break;
    case at::kDouble:
      *(double*)data = THPUtils_unpackDouble(obj);
      break;
    // ...
  }
}
```

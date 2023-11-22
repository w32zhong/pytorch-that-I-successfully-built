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

Good to know: the setup.py will generate Ninja build files as long as the command is installed
(see [this](https://github.com/w32zhong/pytorch-that-I-successfully-built/blob/fec8db5927af25b99da9ddc6a2343f0893ef7bcb/tools/setup_helpers/cmake.py#L31)).

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
$ nm -D --defined-only build/lib/libtorch_python.so | grep initModule
000000000073cdc0 T _Z20THPEngine_initModuleP7_object
0000000000741040 T _Z22THPFunction_initModuleP7_object
000000000076d720 T _Z22THPVariable_initModuleP7_object
0000000000b6bf70 T _ZN5torch3cpu10initModuleEP7_object
0000000000b8a0d0 T _ZN5torch4cuda10initModuleEP7_object
00000000006afa80 T initModule
```

The `initModule` actually is defined in `torch/csrc/Module.cpp`.

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
but since now t

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
>>> for dep in libtorch_python_core_sources[:50]:
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
torch/csrc/autograd/profiler_python.cpp
torch/csrc/autograd/python_anomaly_mode.cpp
torch/csrc/autograd/python_saved_variable_hooks.cpp
torch/csrc/autograd/python_cpp_function.cpp
torch/csrc/autograd/python_engine.cpp
torch/csrc/autograd/python_function.cpp
torch/csrc/autograd/python_hook.cpp
torch/csrc/autograd/python_legacy_variable.cpp
torch/csrc/autograd/python_nested_functions_manual.cpp
torch/csrc/autograd/python_torch_functions_manual.cpp
torch/csrc/autograd/python_variable.cpp
torch/csrc/autograd/python_variable_indexing.cpp
torch/csrc/dynamo/python_compiled_autograd.cpp
torch/csrc/dynamo/cpp_shim.cpp
torch/csrc/dynamo/cpython_defs.c
torch/csrc/dynamo/eval_frame.c
torch/csrc/dynamo/guards.cpp
torch/csrc/dynamo/init.cpp
torch/csrc/functorch/init.cpp
torch/csrc/mps/Module.cpp
torch/csrc/jit/backends/backend_init.cpp
torch/csrc/jit/python/init.cpp
torch/csrc/jit/passes/onnx.cpp
torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp
torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp
torch/csrc/jit/passes/onnx/eval_peephole.cpp
torch/csrc/jit/passes/onnx/constant_fold.cpp
torch/csrc/jit/passes/onnx/constant_map.cpp
torch/csrc/jit/passes/onnx/eliminate_unused_items.cpp
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

## Install
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
$ ls torch
CMakeLists.txt                     _decomp                 _namedtensor_internals.py  _vmap_internals.py          export         mps               share
README.txt                         _deploy.py              _numpy                     _weights_only_unpickler.py  extension.h    multiprocessing   signal
_C                                 _dispatch               _ops.py                    abi-check.cpp               fft            nested            sparse
_C.cpython-39-x86_64-linux-gnu.so  _dynamo                 _prims                     amp                         func           nn                special
_C_flatbuffer                      _export                 _prims_common              ao                          functional.py  onnx              storage.py
_VF.py                             _functorch              _python_dispatcher.py      autograd                    futures        optim             test
_VF.pyi                            _guards.py              _refs                      backends                    fx             overrides.py      testing
__config__.py                      _higher_order_ops       _sources.py                bin                         hub.py         package           torch_version.py
__future__.py                      _inductor               _storage_docs.py           compiler                    include        profiler          types.py
__init__.py                        _jit_internal.py        _streambase.py             contrib                     jit            py.typed          utils
__pycache__                        _lazy                   _subclasses                cpu                         legacy         quantization      version.py
_appdirs.py                        _library                _tensor.py                 csrc                        lib            quasirandom.py    version.py.tpl
_awaits                            _linalg_utils.py        _tensor_docs.py            cuda                        library.h      random.py
_classes.py                        _lobpcg.py              _tensor_str.py             custom_class.h              library.py     return_types.py
_compile.py                        _logging                _torch_docs.py             custom_class_detail.h       linalg         return_types.pyi
_custom_op                         _lowrank.py             _utils.py                  distributed                 masked         script.h
_custom_ops.py                     _meta_registrations.py  _utils_internal.py         distributions               monitor        serialization.py

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

## Quick Start
```sh
$ python hello-world.py
tensor([0.8506], grad_fn=<SigmoidBackward0>)
tensor(0.0223, grad_fn=<MseLossBackward0>)
```

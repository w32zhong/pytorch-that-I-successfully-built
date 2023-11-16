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

```sh
$ echo $PATH | tr ":" "\n"
/home/tk/anaconda3/envs/pytorch-src/bin
/usr/bin
...
```

```sh
$ env | grep CUDA
CUDA_PATH=/opt/cuda
```

```sh
$ nvidia-smi  | grep CUDA
NVIDIA-SMI 535.54.03
Driver Version: 535.54.03
CUDA Version: 12.2
```

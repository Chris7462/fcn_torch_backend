# FCN Segmentation Torch Backend

## Preparation

### Clone PyTorch and TorchVision from GitHub
```
cd ~/thirdparty/
git clone git@github.com:pytorch/pytorch.git --recurse-submodules
git clone git@github.com:pytorch/vision.git --recurse-submodules
```

### Build LibTorch
```bash
cd ~/thirdparty/pytorch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_CUDSS=ON \
         -DUSE_CUFILE=ON \
         -DUSE_CUSPARSELT=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorch
cmake --build . -j8
cmake --install .
```
If you encountered issue, try to add the following in your `~/.bashrc` before build the libtorch
```bash
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda/include:$CPATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

## Point to your custom libtorch installation
Add the following to your `~/.bashrc` file
```bash
export Torch_DIR="$HOME/thirdparty/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorch/lib:$LD_LIBRARY_PATH"
```

### Build LibTorchVision
```bash
cd ~/thirdparty/vision
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DWITH_CUDA=ON \
         -DCMAKE_PREFIX_PATH=$HOME/thirdparty/libtorch \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorchvision
cmake --build . -j8
cmake --install .
```

## Point to your custom libtorchvision installation
Add the following to your `~/.bashrc` file
```bash
export TorchVision_DIR="$HOME/thirdparty/libtorchvision/share/cmake/TorchVision"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorchvision/lib:$LD_LIBRARY_PATH"
```

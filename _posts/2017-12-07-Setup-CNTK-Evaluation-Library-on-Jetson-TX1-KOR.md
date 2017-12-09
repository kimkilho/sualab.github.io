---
layout: post
title: "Jetson TX1에 CNTK Evaluation 라이브러리 설치하기"
date: 2017-12-07 16:00:00 +0900
author: namgoo_lee
categories: [high-performance-computing]
tags: [embedded, tx1, cntk]
comments: true
name: setup-cntk-evaluation-library-on-jetson-tx1
---

이 포스트에서는 Jetson TX1에 CNTK를 이용한 모델 evaluation에 필요한 라이브러리들을 설치하는 방법을 설명하겠습니다. 전체 소요 시간은 약 2시간입니다. 이 포스트는 공식 CNTK 사이트의 [Setup CNTK on Linux](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux) 문서를 참조하였습니다.

CNTK 프로젝트 팀은 리눅스 환경에서 실행할 수 있는 도커 이미지를 제공하고 있습니다. 그런데 현재 [도커 허브](https://hub.docker.com/r/microsoft/cntk/)에 제공되어 있는 이미지들은 GPU 기능이 활성화된 버전을 이용하고 싶을 경우 [nvidia-docker](https://github.com/nvidia/nvidia-docker)를 이용해야 한다고 안내되어 있습니다. 문제는 이 nvidia-docker가 [테그라 플랫폼(Tegra platform)을 지원하지 않는다는 데에 있습니다.](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support) 따라서 TX1에서 CNTK를 돌리기 위해서는 어쩔 수 없이 CNTK 소스코드를 받아서 직접 빌드해야 합니다.

# 빌드대상

이 포스트에서는 다른 기기에서 학습된 CNTK 모델을 TX1에서 evaluation만 하는 상황을 가정하겠습니다. 따라서 CNTK 소스코드에 있는 모든 라이브러리와 실행파일을 빌드하지 않고 다음 목록에 있는 것만 빌드하도록 하겠습니다.

* libCntk.Core-2.1.so
* libCntk.Math-2.1.so
* libCntk.PerformanceProfiler-2.1.so

다음의 실행파일도 빌드할 것인데 학습된 모델을 evaluation 하는 기능을 갖춘 예제코드입니다. CNTK Evaluation 라이브러리가 제대로 빌드 되었는지 테스팅하는 용도로 사용할 것입니다.

* CNTKLibraryCPPEvalExamples

# 사전작업

1. **TX1에 [JetPack 3.1 설치](https://developer.nvidia.com/embedded/jetpack)**: 포스트 작성 시점에서 최신의 JetPack 버전인 3.1 버전이 설치되어 있다고 가정하겠습니다.

2. **TX1 개발용 키트에 추가 저장 장치 설치**: Jetson TX1 개발용 키트를 구매하면 기본 저장장치로 eMMC가 설치되어 있는데 용량이 16GB에 불과합니다. JetPack을 설치하면 저장 공간 부족 문제로 CNTK를 빌드할 수 없습니다. 저는 [이 가이드](http://www.jetsonhacks.com/2017/01/28/install-samsung-ssd-on-nvidia-jetson-tx1/)를 참고로 해서 256GB SSD를 설치하였고 TX1이 SSD에서 부팅하도록 하였습니다.

TX1에서 개발을 진행할 때는 `jetson_clocks.sh` 스크립트를 실행하는 것이 좋습니다. 이 스크립트는 TX1의 CPU, GPU 및 EMC(메모리 컨트롤러) 클럭을 최대로 설정해줍니다. 이 스크립트는 JetPack을 설치하면 홈 디렉토리에 복사되어 있습니다.
```console
$ sudo ~/jetson_clocks.sh
```

# 빌드

CNTK 설치를 위한 사전작업으로 다음의 라이브러리들을 설치하겠습니다.

* OpenBLAS 0.2.20
* libzip 1.1.2
* Boost 1.58.0
* CUB 1.4.1
* protobuf 3.1.0

그리고 CNTK `configure` 스크립트가 위 라이브러리들을 잘 찾을 수 있도록 시스템 폴더들에 살짝 손을 대는 과정을 거치도록 하겠습니다.

여기까지 끝나면 CNTK를 빌드할 수 있습니다. GPU 기능이 활성화된 2.1 버전을 빌드하도록 하겠습니다.

* CNTK 2.1 (GPU enabled)

## OpenBLAS v0.2.20

[CNTK에 사용되는 기본 수학 라이브러리는 Intel Math Kernel Library (Intel MKL) 입니다.](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#mkl) 그러나 이 라이브러리는 [여기](https://software.intel.com/en-us/mkl)에 명시되어 있듯이 Intel 계열 프로세서만 지원합니다. TX1은 aarch64 기반이므로 우리는 다른 수학 라이브러리를 이용해야 합니다. CNTK는 현재 Intel MKL 말고 OpenBLAS를 지원하고 있는 것으로 보이므로 ([참고1](https://github.com/Microsoft/CNTK/issues/2198), [참고2](https://github.com/Microsoft/CNTK/blob/v2.1/configure#L33)) 우리는 OpenBLAS를 사용하도록 하겠습니다.

OpenBLAS를 빌드하기 전에 gfortran 라이브러리를 설치해야 합니다. 그래야 [OpenBLAS에 LAPACKE 라이브러리가 포함되서 빌드가 되는데](https://github.com/Microsoft/CNTK/issues/1424), 이렇게 하지 않으면 나중에 CNTK 빌드시에 링킹 에러가 발생합니다.

```console
nvidia@tegra-ubuntu:~$ git clone https://github.com/xianyi/OpenBLAS.git
nvidia@tegra-ubuntu:~$ cd OpenBLAS
nvidia@tegra-ubuntu:~/OpenBLAS$ git checkout v0.2.20
nvidia@tegra-ubuntu:~/OpenBLAS$ sudo apt install gfortran
nvidia@tegra-ubuntu:~/OpenBLAS$ make -j3
```

빌드가 완료되면 다음과 같이 LAPACK 및 LAPACKE 라이브러리까지 빌드되었다고 나와야 합니다.

```console
 OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)

  OS               ... Linux             
  Architecture     ... arm64               
  BINARY           ... 64bit                 
  C compiler       ... GCC  (command line : gcc)
  Fortran compiler ... GFORTRAN  (command line : gfortran)
  Library Name     ... libopenblas_cortexa57p-r0.2.20.a (Multi threaded; Max num-threads is 4)
```

이제 빌드된 라이브러리를 다음과 같이 설치하도록 하겠습니다.

```console
nvidia@tegra-ubuntu:~/OpenBLAS$ sudo make install PREFIX=/usr/local/OpenBLAS
```

## libzip 1.1.2

JetPack 3.1은 Ubuntu 16.04를 포함하고 있습니다. Ubuntu 16.04에서 제공하는 libzip 버전은 1.0.1입니다([참고](https://launchpad.net/ubuntu/xenial/+source/libzip)). 그러나 CNTK 문서에서 [보다 최신 버전을 설치할 것을 강하게 권고](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#libzip)하고 있기 때문에 우리도 문서를 따라 1.1.2 버전을 설치하도록 하겠습니다.
```console
nvidia@tegra-ubuntu:~$ wget http://nih.at/libzip/libzip-1.1.2.tar.gz
nvidia@tegra-ubuntu:~$ tar xzf ./libzip-1.1.2.tar.gz
nvidia@tegra-ubuntu:~$ cd libzip-1.1.2
nvidia@tegra-ubuntu:~/libzip-1.1.2$ ./configure
nvidia@tegra-ubuntu:~/libzip-1.1.2$ make -j3
nvidia@tegra-ubuntu:~/libzip-1.1.2$ sudo make install
```

## Boost 1.58.0

Ubuntu 16.04에서 기본으로 제공하는 버전은 1.58.0입니다([참고](https://launchpad.net/ubuntu/xenial/+package/libboost-all-dev)). CNTK 문서에서 [예시로 든 버전은 1.60.0 버전](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#boost-library)이지만 TX1에서 Boost 소스를 빌드하기에는 시간도 오래 걸리고, 우리는 그냥 1.58.0 버전을 이용하도록 하겠습니다.
```console
nvidia@tegra-ubuntu:~$ sudo apt install libboost-all-dev
```

## CUB 1.4.1

```console
nvidia@tegra-ubuntu:~$ wget https://github.com/NVlabs/cub/archive/1.4.1.zip
nvidia@tegra-ubuntu:~$ unzip ./1.4.1.zip
nvidia@tegra-ubuntu:~$ sudo cp -r cub-1.4.1 /usr/local
```

## Protobuf 3.1.0
Ubuntu 16.04에서 기본으로 제공하는 버전은 2.6.1입니다([참고](https://launchpad.net/ubuntu/xenial/+source/protobuf)). 이 버전을 이용해서 CNTK를 빌드하려고 했으나 proto2가 아닌 proto3를 요구하면서 빌드에 실패하였습니다. 따라서 [CNTK 문서에 따라](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#protobuf) 3.1.0 버전을 받아 설치하겠습니다.
```console
nvidia@tegra-ubuntu:~$ wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz
nvidia@tegra-ubuntu:~$ tar xzf v3.1.0.tar.gz
nvidia@tegra-ubuntu:~$ cd protobuf-3.1.0
nvidia@tegra-ubuntu:~/protobuf-3.1.0$ sudo apt install curl
nvidia@tegra-ubuntu:~/protobuf-3.1.0$ ./autogen.sh
nvidia@tegra-ubuntu:~/protobuf-3.1.0$ ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/protobuf-3.1.0
nvidia@tegra-ubuntu:~/protobuf-3.1.0$ make -j3
nvidia@tegra-ubuntu:~/protobuf-3.1.0$ sudo make install
```

## 추가 설정

CNTK `configure` 스크립트가 우리가 설치한 라이브러리들을 알아볼 수 있도록 심볼릭 링크(symbolic link)를 생성하도록 하겠습니다.

```shell
# make symbolic links for CUDNN
$ sudo mkdir /usr/local/cudnn-6.0/cuda/include -p
$ sudo ln -s /usr/lib/aarch64-linux-gnu /usr/local/cudnn-6.0/lib
$ sudo ln -s /usr/include/aarch64-linux-gnu/cudnn_v6.h /usr/local/cudnn-6.0/cuda/include/cudnn.h

# make symbolic link for CUDA
$ sudo ln -s /usr/local/cuda-8.0/targets/aarch64-linux/include/nvml.h /usr/local/include

# make symbolic link for OpenMPI
$ sudo mkdir /usr/lib/openmpi/bin
$ sudo ln -s /usr/bin/mpic++ /usr/lib/openmpi/bin/mpic++

# make symbolic links for Boost
$ sudo mkdir /usr/local/boost-1.58.0
$ sudo ln -s /usr/lib/aarch64-linux-gnu /usr/local/boost-1.58.0/lib
$ sudo ln -s /usr/include /usr/local/boost-1.58.0/include
```

## CNTK 2.1

이제 본격적으로 CNTK를 빌드하겠습니다.

```console
nvidia@tegra-ubuntu:~$ git clone https://github.com/Microsoft/CNTK.git
nvidia@tegra-ubuntu:~$ cd CNTK
nvidia@tegra-ubuntu:~/CNTK$ git checkout v2.1
nvidia@tegra-ubuntu:~/CNTK$ mkdir build/release -p
nvidia@tegra-ubuntu:~/CNTK$ cd build/release
nvidia@tegra-ubuntu:~/CNTK/build/release$ ../../configure --asgd=no                                 \
                                                          --cuda=yes                                \
                                                          --with-openblas=/usr/local/OpenBLAS       \
                                                          --with-boost=/usr/local/boost-1.58.0      \
                                                          --with-cudnn=/usr/local/cudnn-6.0         \
                                                          --with-protobuf=/usr/local/protobuf-3.1.0 \
                                                          --with-mpi=/usr/lib/openmpi               \
                                                          --with-gdk-include=/usr/local/include     \
                                                          --with-gdk-nvml-lib=/usr/local/cuda-8.0/targets/aarch64-linux/lib/stubs
nvidia@tegra-ubuntu:~/CNTK/build/release$ make -C ../../                                                                                                                            \
                                               BUILD_TOP=$PWD                                                                                                                       \
                                               SSE_FLAGS=''                                                                                                                         \
                                               GENCODE_FLAGS='-gencode arch=compute_53,code=\"sm_53,compute_53\"'                                                                   \
                                               CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC='$(PWD)/../../Examples/Evaluation/CNTKLibraryCPPEvalGPUExamples/CNTKLibraryCPPEvalGPUExamples.cpp' \
                                               CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC+='$(PWD)/../../Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/EvalMultithreads.cpp'         \
                                               $PWD/lib/libCntk.Core-2.1.so                                                                                                         \
                                               $PWD/lib/libCntk.Math-2.1.so                                                                                                         \
                                               $PWD/lib/libCntk.PerformanceProfiler-2.1.so                                                                                          \
                                               $PWD/bin/CNTKLibraryCPPEvalExamples
```

빌드 도중 `g++: internal compiler error: Killed (program cc1plus)` 에러가 발생한다면 메모리 부족 문제일 가능성이 큽니다. `make`시에 `-j` 옵션은 되도록 주지 않는 것이 좋고, 웹브라우저같은 다른 어플리케이션은 종료시킨 상태로 컴파일을 하는 것이 좋습니다.

빌드가 끝나면 하기 경로에 다음과 같이 3개의 라이브러리가 빌드되었음을 볼 수 있습니다.

```console
nvidia@tegra-ubuntu:~/CNTK/build/release$ ls lib
libCntk.Core-2.1.so  libCntk.Math-2.1.so  libCntk.PerformanceProfiler-2.1.so
```

또한 하기 경로에는 테스트용 실행파일이 빌드되었음을 볼 수 있습니다.

```console
nvidia@tegra-ubuntu:~/CNTK/build/release$ ls bin
CNTKLibraryCPPEvalExamples
```

# 테스트

테스트를 위해서 `01_OneHidden.model` 파일이 필요합니다. 이 파일은 [여기](https://github.com/nglee/CNTK/tree/master/Examples/Image/GettingStarted)에 있는 설명을 따라 `01_OneHidden.cntk`를 학습시키면 얻을 수 있습니다. 저는 윈도우 머신에서 이 모델을 학습시켰고 그것을 TX1으로 복사해서 돌려보았습니다. `01_OneHidden.model`은 `CNTKLibraryCPPEvalExamples`와 같은 디렉토리에 위치해야 합니다.

```console
nvidia@tegra-ubuntu:~/CNTK/build/release$ cd bin
nvidia@tegra-ubuntu:~/CNTK/build/release/bin$ sudo ln -s /usr/local/cuda-8.0/targets/aarch64-linux/lib/stubs/libnvidia-ml.so /usr/local/cuda-8.0/targets/aarch64-linux/lib/stubs/libnvidia-ml.so.1
nvidia@tegra-ubuntu:~/CNTK/build/release/bin$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/targets/aarch64-linux/lib/stubs:../lib:$LD_LIBRARY_PATH
nvidia@tegra-ubuntu:~/CNTK/build/release/bin$ ./CNTKLibraryCPPEvalExamples
```

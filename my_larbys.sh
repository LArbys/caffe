#!/usr/bin/env bash

# ROOT
source /home/taritree/software/root/bin/thisroot.sh
export ROOT_INCDIR=`$ROOTSYS/bin/root-config --incdir`
export ROOT_LIBDIR=`$ROOTSYS/bin/root-config --libdir`
export ROOT_BINDIR=`$ROOTSYS/bin/root-config --bindir`

# BOOST
export BOOST_INCDIR=/usr/local/include
export BOOST_LIBDIR=/usr/lib/x86_64-linux-gnu

# OpenCV
export OPENCV_INCDIR=/usr/local/include
export OPENCV_LIBDIR=/usr/local/lib

# Qt5
export QT_INCDIR=/usr/include/qt5
export QT_LIBDIR=/usr/lib/x86_64-linux-gnu/qt5

# OpenBLAS
export OPENBLAS_INCDIR=/usr/include
export OPENBLAS_LIBDIR=/usr/lib

# protobuf
export PROTOBUF_INCDIR=/usr/include/google
export PROTOBUF_LIBDIR=/usr/lib/x86_64-linux-gnu

# glog
export GLOG_INCDIR=/usr/include/glog
export GLOG_LIBDIR=/usr/lib/x86_64-linux-gnu

# gflags
export GFLAGS_INCDIR=/usr/include/gflags
export GFLAGS_LIBDIR=/usr/lib/x86_64-linux-gnu

# hdf5
export HDF5_INCDIR=/usr/include/hdf5/serial
export HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu
export HDF5_BINDIR=/usr/bin

# leveldb
export LEVELDB_INCDIR=/usr/include
export LEVELDB_LIBDIR=/usr/lib/x86_64-linux-gnu

# lmdb
export LMDB_INCDIR=/usr/include
export LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu

# cuda
export CUDA_INCDIR=/usr/local/cuda-8.0/targets/x86_64-linux/include
export CUDA_LIBDIR=/usr/local/cuda-8.0/targets/x86_64-linux/lib

# caffe: setme!
#export CAFFE_DIR=
#export CAFFE_LIBDIR=${CAFFE_DIR}/build/lib
#export CAFFE_INCDIR=${CAFFE_DIR}/build/include
#export CAFFE_BINDIR=${CAFFE_DIR}/build/tools

# Set PATH and PYTHONPATH

#if [[ -z $NUDOT_LARBYS_CONFIG ]]; then
#export PATH=$CAFFE_BINDIR:$ROOT_BINDIR:$HDF5_BINDIR:$PATH
#export PYTHONPATH=$ROOT_LIBDIR:${CAFFE_LIBDIR}/python:$PYTHONPATH
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$CAFFE_LIBDIR:$ROOT_LIBDIR:$OPENCV_LIBDIR:$QT_LIBDIR:$CUDA_LIBDIR:$LD_LIBRARY_PATH
#export NUDOT_LARBYS_CONFIG=1
#fi

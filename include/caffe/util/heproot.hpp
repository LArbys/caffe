#ifndef CAFFE_UTIL_ROOT_H_
#define CAFFE_UTIL_ROOT_H_

#include <string>

//Caffe
#include "caffe/blob.hpp"

namespace caffe {

  struct root_helper  {

    //::larcv::IOManager* iom;
    std::string _filler_name;

  };
  
  template <typename Dtype>
  void root_load_data(root_helper& rh, Blob<Dtype>* data_blob, Blob<Dtype>* label_blob);

  template <typename Dtype>
  void root_load_data(root_helper& rh, Blob<Dtype>* data_blob, Blob<Dtype>* label_blob, Blob<Dtype>* weight_blob);
  
}  // namespace caffe

#endif   // CAFFE_UTIL_ROOT_H_

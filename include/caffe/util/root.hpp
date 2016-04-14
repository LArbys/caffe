#ifndef CAFFE_UTIL_ROOT_H_
#define CAFFE_UTIL_ROOT_H_

#include <string>

//LArCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

//Caffe
#include "caffe/blob.hpp"

namespace caffe {

  
  template <typename Dtype>

  void root_load_data<Dtype>(::larcv::IOManager* iom, 
			     std::string producer,
			     Blob<float>* data_blob,
			     Blob<float>* label_blob);

  
  
}  // namespace caffe

#endif   // CAFFE_UTIL_ROOT_H_

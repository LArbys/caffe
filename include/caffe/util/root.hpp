#ifndef CAFFE_UTIL_ROOT_H_
#define CAFFE_UTIL_ROOT_H_

#include <string>

//LArCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"

//Caffe
#include "caffe/blob.hpp"

namespace caffe {

  template <typename Dtype>
  void root_load_nd_dataset_helper(::larcv::EventBase* ev_data, 
				   int nentries, 
				   int min_dim, 
				   int max_dim,
				   Blob<Dtype>* blob);

  template <typename Dtype>
  void root_load_nd_dataset(::larcv::IOManager* iom, 
			    const char* dataset_name_, 
			    int min_dim,
			    int max_dim,
			    Blob<Dtype>* blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_ROOT_H_

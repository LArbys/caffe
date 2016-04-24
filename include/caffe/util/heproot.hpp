#ifndef CAFFE_UTIL_ROOT_H_
#define CAFFE_UTIL_ROOT_H_

#include <string>

//Caffe
#include "caffe/blob.hpp"

namespace caffe {


  struct root_helper  {

    //::larcv::IOManager* iom;
    std::string filename;
    std::string image_producer;
    std::string roi_producer;
    int nentries;
    
    std::vector<float> img_means;
    std::vector<std::vector<float> > mean_imgs;
    std::vector<float> imin_v;
    std::vector<float> imax_v;
  };
  
  template <typename Dtype>
  void root_load_data(root_helper& rh,
		      Blob<Dtype>* data_blob,
		      Blob<Dtype>* label_blob);

  
  
}  // namespace caffe

#endif   // CAFFE_UTIL_ROOT_H_

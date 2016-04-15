#ifndef CAFFE_UTIL_ROOT_H_
#define CAFFE_UTIL_ROOT_H_

#include <string>

//Caffe
#include "caffe/blob.hpp"

namespace caffe {


  struct root_helper  {

    //::larcv::IOManager* iom;
    std::string filename;
    std::string producer;
    bool background;
    int nentries;
    
    std::vector<float> img_means;
    std::vector<std::vector<float> > mean_imgs;
    float imin;
    float imax;

  };
  
  template <typename Dtype>
  void root_load_data(root_helper& rh,
		      Blob<Dtype>* data_blob,
		      Blob<Dtype>* label_blob);

  
  
}  // namespace caffe

#endif   // CAFFE_UTIL_ROOT_H_

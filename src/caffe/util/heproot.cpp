#include "caffe/util/heproot.hpp"
#include <string>
#include <vector>
//LArCV
#include "APICaffe/ThreadDatumFiller.h"
#include "APICaffe/ThreadFillerFactory.h"

namespace caffe {

  template <>
  void root_load_data<float>(root_helper& rh, Blob<float>* data_blob, Blob<float>* label_blob)
  {
    auto& filler = ::larcv::ThreadFillerFactory::get_filler(rh._filler_name);
    size_t wait_counter=0;
    while(filler.thread_running()) {
      usleep(200);
      ++wait_counter;
      if(wait_counter%5000==0)
        LOG(INFO) << "Queuing data... (" << wait_counter/5000 << " sec.)" << std::endl;
    }

    //
    // Define blob dimension
    //
    auto const& data_dims = filler.dim();
    auto const& data = filler.data();

    // changing for segmentation
    //std::vector<int> label_dims(1);
    //auto const& label_dims = filler.label_dim();

    auto const& label = filler.labels();
    
    //label_dims[0] = data_dims[0];

    data_blob->Reshape(data_dims);  
    label_blob->Reshape(data_dims);

    /*
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
    	        << " with memory size " << data.size() * sizeof(float)  << "\n";
    */
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    /*
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
    	        << " with memory size " << label.size() * sizeof(float)  << "\n";
    */
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(float) );    
  }

  template <>
  void root_load_data<double>(root_helper& rh, Blob<double>* data_blob, Blob<double>* label_blob)
  {
    LOG(ERROR) << "Not implemented!" << std::endl;
    throw ::larcv::larbys();
  }

}  // namespace caffe

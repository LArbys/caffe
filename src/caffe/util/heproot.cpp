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
    //LOG(INFO) << "Start " << __FUNCTION__ << std::endl;
    auto& filler = ::larcv::ThreadFillerFactory::get_filler(rh._filler_name);
    size_t wait_counter=0;
    while(filler.thread_running()) {
      usleep(200);
      ++wait_counter;
      if(wait_counter%5000==0)
        LOG(INFO) << "Queuing data... (" << wait_counter/5000 << " sec.)" << std::endl;
    }
    //LOG(INFO) << "IO Thread wait time: " << wait_counter * 200 << " [usec]" << std::endl;
    //
    // Define blob dimension
    //
    auto const& image_dims = filler.dim();
    auto const& image_data = filler.data();

    auto const& label_dims = filler.dim(false);
    auto const& label_data = filler.labels();

    data_blob->Reshape(image_dims);  
    label_blob->Reshape(label_dims);

    /*
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
    	        << " with memory size " << data.size() * sizeof(float)  << "\n";
    */
    memcpy(data_blob->mutable_cpu_data(),image_data.data(),image_data.size() * sizeof(float) );
    /*
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
    	        << " with memory size " << label.size() * sizeof(float)  << "\n";
    */
    memcpy(label_blob->mutable_cpu_data(),label_data.data(),label_data.size() * sizeof(float) );    
    //LOG(INFO) << "Finished " << __FUNCTION__ << std::endl;
  }

  template <>
  void root_load_data<double>(root_helper& rh, Blob<double>* data_blob, Blob<double>* label_blob)
  {
    LOG(ERROR) << "Not implemented!" << std::endl;
    throw ::larcv::larbys();
  }

  template <>
  void root_load_data<float>(root_helper& rh, Blob<float>* data_blob, Blob<float>* label_blob, Blob<float>* weight_blob)
  {
    //LOG(INFO) << "Start " << __FUNCTION__ << std::endl;
    auto& filler = ::larcv::ThreadFillerFactory::get_filler(rh._filler_name);
    size_t wait_counter=0;
    while(filler.thread_running()) {
      usleep(200);
      ++wait_counter;
      if(wait_counter%5000==0)
        LOG(INFO) << "Queuing data... (" << wait_counter/5000 << " sec.)" << std::endl;
    }
    //LOG(INFO) << "IO Thread wait time: " << wait_counter * 200 << " [usec]" << std::endl;
    //
    // Define blob dimension
    //
    auto const& image_dims = filler.dim();
    auto const& image_data = filler.data();

    auto const& label_dims = filler.dim(false);
    auto const& label_data = filler.labels();

    auto const& weight_data = filler.weights();

    data_blob->Reshape(image_dims);  
    label_blob->Reshape(label_dims);
    weight_blob->Reshape(image_dims);

    /*
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
    	        << " with memory size " << data.size() * sizeof(float)  << "\n";
    */
    memcpy(data_blob->mutable_cpu_data(),image_data.data(),image_data.size() * sizeof(float) );
    /*
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
    	        << " with memory size " << label.size() * sizeof(float)  << "\n";
    */
    memcpy(label_blob->mutable_cpu_data(),label_data.data(),label_data.size() * sizeof(float) );
    /*
    LOG(INFO) << "\t>> memcpy with weight.size() " << weight.size() 
    	        << " with memory size " << weight.size() * sizeof(float)  << "\n";
    */
    memcpy(weight_blob->mutable_cpu_data(),weight_data.data(),weight_data.size() * sizeof(float) );
    //LOG(INFO) << "Finished " << __FUNCTION__ << std::endl;
  }

  template <>
  void root_load_data<double>(root_helper& rh, Blob<double>* data_blob, Blob<double>* label_blob, Blob<double>* weight_blob)
  {
    LOG(ERROR) << "Not implemented!" << std::endl;
    throw ::larcv::larbys();
  }

}  // namespace caffe

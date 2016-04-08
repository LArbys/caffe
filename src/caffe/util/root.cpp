#include "caffe/util/root.hpp"

#include <string>
#include <vector>

namespace caffe {

  // Verifies format of data stored in ROOT file and reshapes blob accordingly.
  template <typename Dtype>
  void root_load_nd_dataset_helper(::larcv::EventBase* ev_data,
				   const char* dataset_name_, 
				   int min_dim, 
				   int max_dim,
				   Blob<Dtype>* blob) {
    

    const auto& meta = ((::larcv::EventImage2D*)(ev_data))->Image2DArray()[0].meta();
    
    //1 image... in first position batching for now
    //int* dim = { 1,3, meta.rows(), meta.cols() };
    //std::vector<int> dims = { 1, 3, meta.rows(), meta.cols() };
    std::vector<int> dims(4);
    dims[0]=1; dims[1]=3; dims[2]=meta.rows(); dims[3]=meta.cols();

    vector<int> blob_dims(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      blob_dims[i] = dims[i];
    }

    blob->Reshape(blob_dims);
  }

  template <>
  void root_load_nd_dataset<float>(::larcv::IOManager* iom, 
				   const char* dataset_name_,
				   int min_dim, 
				   int max_dim, 
				   Blob<float>* blob) {
    
    iom->read_entry(0);
    ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    
    //load data dimensions and reshape the blob
    root_load_nd_dataset_helper(ev_data, dataset_name_, min_dim, max_dim, blob);
    
    auto const& imgs = ev_data->Image2DArray();

    std::vector<float> data; data.reserve( 3 * imgs[0].as_vector().size() );
    
    for(size_t i=0;i<3;++i) {
      
      const std::vector<float> & v = imgs[i].as_vector();
      size_t len = v.size();

      for(size_t j=0;j<len;++j) 

	data[i*len+j] = v[j];
      
    }

    memcpy(blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
  }

  template <>
  void root_load_nd_dataset<double>(::larcv::IOManager* iom, 
				    const char* dataset_name_,
				    int min_dim, 
				    int max_dim, 
				    Blob<double>* blob) {
    
    iom->read_entry(0);
    ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    
    //load data dimensions and reshape the blob
    root_load_nd_dataset_helper(ev_data, dataset_name_, min_dim, max_dim, blob);
    
    auto const& imgs = ev_data->Image2DArray();

    std::vector<float> data; data.reserve( 3 * imgs[0].as_vector().size() );
    
    for(size_t i=0;i<3;++i) {
      
      const std::vector<float> & v = imgs[i].as_vector();
      size_t len = v.size();

      for(size_t j=0;j<len;++j) 

	data[i*len+j] = v[j];
      
    }

    memcpy(blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
  }


}  // namespace caffe

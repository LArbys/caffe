#include "caffe/util/heproot.hpp"

#include <string>
#include <vector>

namespace caffe {
  
  template <>
  void root_load_data<float>(root_helper& rh,
			     Blob<float>* data_blob,
			     Blob<float>* label_blob) {
    
    auto& iom = rh.iom;

    iom->read_entry(0);
    
    ::larcv::EventImage2D* ev_data =
	(::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,rh.producer));

    bool background = rh.background;
    
    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = rh.nentries;

    const auto& im = ev_data->Image2DArray();
    const auto& meta = im[0].meta();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; 
    data_dims[1]  = 3; 
    data_dims[2]  = meta.rows(); 
    data_dims[3]  = meta.cols();
    
    label_dims[0] = nentries;
   
    data_blob->Reshape(data_dims);   
    label_blob->Reshape(label_dims);

    int nchannels = im.size();

    std::vector<float> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );

    std::vector<float> label; 
    label.resize( nentries );

    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(entry);
      ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,rh.producer));
      
      label[entry] = background ? 0 : 1; // class
      
      auto const& imgs = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	const std::vector<float> & v = imgs[ ch ].as_vector();
	size_t len = v.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  data[ idx ] = v[j] - rh.img_means[ch];

	  if ( data[ idx ] < rh.imin ) data[ idx ] = 0;
	  if ( data[ idx ] > rh.imax ) data[ idx ] = rh.imax;
	  
	}
      }
    }
    
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(float)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(float)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(float) );
    
  }

  
}  // namespace caffe

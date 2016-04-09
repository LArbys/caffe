#include "caffe/util/root.hpp"

#include <string>
#include <vector>

namespace caffe {

  // Verifies format of data stored in ROOT file and reshapes blob accordingly.
  template <typename Dtype>
  void root_load_nd_dataset_helper(::larcv::EventImage2D* ev_data,
				   int nentries,
				   int min_dim, 
				   int max_dim,
				   Blob<Dtype>* blob) {
    
    const auto& meta = ev_data->Image2DArray()[0].meta();
    
    
    std::vector<int> dims(4);
    dims[0]=nentries; dims[1]=3; dims[2]=meta.rows(); dims[3]=meta.cols();

    vector<int> blob_dims(dims.size());

    for (int i = 0; i < dims.size(); ++i) 

      blob_dims[i] = dims[i];
    

    blob->Reshape(blob_dims);
  }

  template <>
  void root_load_nd_dataset<float>(::larcv::IOManager* iom, 
				   const char* dataset_name_,
				   int min_dim, 
				   int max_dim, 
				   Blob<float>* blob) {
    
    

    std::cout << "\t>> get first entry\n";
    iom->read_entry(0);
    
    std::cout << "\t>> get ev_data\n";
    ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    
    std::cout << "\t>> load dn_data_set_helper\n";
    //load data dimensions and reshape the blob

    //should be long?
    int nentries = iom->get_n_entries();
      
    root_load_nd_dataset_helper(ev_data, nentries, min_dim, max_dim, blob);

    auto& im = ev_data->Image2DArray();

    std::vector<float> data; data.resize( nentries * 3 * im[0].as_vector().size() );
    std::cout << "\t>> just made data... data.size()" << data.size() << "\n";
    int nchannels = 3;

    std::cout << ">> \t got nentries(): " << nentries << "\n";
    
    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(entry);
      ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
      
      std::cout << "\t>> ENTRY: " << entry << "\n";
      
      auto const& imgs = ev_data->Image2DArray();
    
      for(size_t ch=0;ch<nchannels;++ch) {
	std::cout << "CH: " << ch << "\n";
	
	const std::vector<float> & v = imgs[ ch ].as_vector();
	size_t len = v.size();

	for(size_t j=0;j<len;++j)  {
	  data[ ( entry * nchannels + ch)*len + j ] = v[j];
	  std::cout << v[j] << " ";

	  if ( j%10 == 0 && j != 0) std::cout << "\n";

	}

	std::cout << "\n";
      }

    }

    std::cout << "\t>> memcpy with data.size() " << data.size() << " with memory size" << data.size() * sizeof(float)  << "\n";
    memcpy(blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
  }

    template <>
  void root_load_nd_dataset<double>(::larcv::IOManager* iom, 
				   const char* dataset_name_,
				   int min_dim, 
				   int max_dim, 
				   Blob<double>* blob) {
    
    

    std::cout << "\t>> get first entry\n";
    iom->read_entry(0);
    
    std::cout << "\t>> get ev_data\n";
    ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    
    std::cout << "\t>> load dn_data_set_helper\n";
    //load data dimensions and reshape the blob

    //should be long?
    int nentries = iom->get_n_entries();
      
    root_load_nd_dataset_helper(ev_data, nentries, min_dim, max_dim, blob);

    auto& im = ev_data->Image2DArray();

    std::vector<double> data; data.resize( nentries * 3 * im[0].as_vector().size() );
    std::cout << "\t>> just made data... data.size()" << data.size() << "\n";
    int nchannels = 3;

    std::cout << ">> \t got nentries(): " << nentries << "\n";
    
    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(entry);
      ::larcv::EventImage2D* ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
      
      std::cout << "\t>> ENTRY: " << entry << "\n";
      
      auto const& imgs = ev_data->Image2DArray();
    
      for(size_t ch=0;ch<nchannels;++ch) {
	std::cout << "CH: " << ch << "\n";
	
	const std::vector<float> & v = imgs[ ch ].as_vector();
	size_t len = v.size();

	for(size_t j=0;j<len;++j)  {
	  data[ ( entry * nchannels + ch)*len + j ] = v[j];
	  std::cout << v[j] << " ";

	  if ( j%10 == 0 && j != 0) std::cout << "\n";

	}

	std::cout << "\n";
      }

    }

    std::cout << "\t>> memcpy with data.size() " << data.size() << " with memory size" << data.size() * sizeof(double)  << "\n";
    memcpy(blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );
    
  }



}  // namespace caffe

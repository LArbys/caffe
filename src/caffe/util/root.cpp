#include "caffe/util/root.hpp"

#include <string>
#include <vector>

namespace caffe {
  
  template <>
  void root_load_data<float>(::larcv::IOManager* iom, 
			     Blob<float>* data_blob,
			     Blob<float>* label_blob) {
    
    std::cout << "\t>> loading data\n";
    
    std::cout << "\t>> get first entry\n";
    iom->read_entry(0);
    
    std::cout << "\t>> get ev_data\n";
    ::larcv::EventImage2D* ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    ::larcv::EventROI*     roi_data = (::larcv::EventROI*)    (iom->get_data(::larcv::kProductROI    ,"event_roi"));
    
    std::cout << "\t>> load dn_data_set_helper\n";
    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = iom->get_n_entries();

    const auto& im = ev_data->Image2DArray();
    const auto& meta = im[0].meta();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; data_dims[1]  = 3; data_dims[2]  = meta.rows(); data_dims[3]  = meta.cols();
    label_dims[0] = nentries;
   
    data_blob->Reshape (data_dims);   
    label_blob->Reshape(label_dims);


    std::vector<float> data; data.resize( nentries * 3 * im[0].as_vector().size() );
    std::vector<float> label; label.resize( nentries );

    std::cout << "\t>> just made data...  data.size()" << data.size() << "\n";
    std::cout << "\t>> just made label... label.size()" << label.size() << "\n";

    int nchannels = 3;

    std::cout << ">> \t got nentries(): " << nentries << "\n";
    
    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(entry);
      ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
      roi_data = (::larcv::EventROI*)    (iom->get_data(::larcv::kProductROI    ,"event_roi"));
      
      std::cout << "\t>> ENTRY: " << entry << "\n";

      label[ entry ] = ( float ) roi_data->at(0).Type(); //float is o.k. ?
      
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
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );

    std::cout << "\t>> memcpy with label.size() " << label.size() << " with memory size" << label.size() * sizeof(float)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(float) );
    
  }
  template <>
  void root_load_data<double>(::larcv::IOManager* iom, 
			     Blob<double>* data_blob,
			     Blob<double>* label_blob) {
    
    std::cout << "\t>> loading data\n";
    
    std::cout << "\t>> get first entry\n";
    iom->read_entry(0);
    
    std::cout << "\t>> get ev_data\n";
    ::larcv::EventImage2D* ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
    ::larcv::EventROI*     roi_data = (::larcv::EventROI*)    (iom->get_data(::larcv::kProductROI    ,"event_roi"));
    
    std::cout << "\t>> load dn_data_set_helper\n";
    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = iom->get_n_entries();

    const auto& im = ev_data->Image2DArray();
    const auto& meta = im[0].meta();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; data_dims[1]  = 3; data_dims[2]  = meta.rows(); data_dims[3]  = meta.cols();
    label_dims[0] = nentries;
   
    data_blob->Reshape (data_dims);   
    label_blob->Reshape(label_dims);


    std::vector<double> data; data.resize( nentries * 3 * im[0].as_vector().size() );
    std::vector<double> label; label.resize( nentries );

    std::cout << "\t>> just made data...  data.size()" << data.size() << "\n";
    std::cout << "\t>> just made label... label.size()" << label.size() << "\n";

    int nchannels = 3;

    std::cout << ">> \t got nentries(): " << nentries << "\n";
    
    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(entry);
      ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,"event_image"));
      roi_data = (::larcv::EventROI*)    (iom->get_data(::larcv::kProductROI    ,"event_roi"));
      
      std::cout << "\t>> ENTRY: " << entry << "\n";


      label[ entry ] = ( double ) roi_data->at(0).Type(); //float is o.k. ?
      
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
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );

    std::cout << "\t>> memcpy with label.size() " << label.size() << " with memory size" << label.size() * sizeof(double)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(double) );
    
  }



}  // namespace caffe

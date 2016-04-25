#include "caffe/util/heproot.hpp"
#include "caffe/util/larcv_iom_singleton.hpp"
#include <string>
#include <vector>
//LArCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace caffe {

  size_t random_entry(size_t nmax)
  {
    static const double rmax = (double)RAND_MAX;
    return (size_t)(( (double)(rand()) / rmax ) * nmax);
  }

  template <>
  void root_load_data<float>(root_helper& rh,
			     Blob<float>* data_blob,
			     Blob<float>* label_blob) {
    LOG(INFO) << "Getting singleton";
    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);

    const ::larcv::ProducerID_t image_producer_id = iom.producer_id(::larcv::kProductImage2D,rh.image_producer);
    const ::larcv::ProducerID_t roi_producer_id = iom.producer_id(::larcv::kProductROI,rh.roi_producer);
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));

    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = rh.nentries;

    const auto& im = ev_data->Image2DArray();
    const auto& meta = im[0].meta();

    const int nchannels = im.size();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; 
    data_dims[1]  = nchannels;
    data_dims[2]  = meta.rows(); 
    data_dims[3]  = meta.cols();
    
    label_dims[0] = nentries;
    LOG(INFO) << "Reshape data";    
    data_blob->Reshape(data_dims);  
    LOG(INFO) << "Reshape label";     
    label_blob->Reshape(label_dims);

    LOG(INFO) << "Resize data";    
    std::vector<float> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );
    LOG(INFO) << "Resize label";    
    std::vector<float> label; 
    label.resize( nentries );

    size_t cosmic_count=0;
    size_t nu_count=0;

    bool use_flat_mean = rh.mean_imgs.empty();
    LOG(INFO) << "Reading nentries: " << nentries;
    for(int entry = 0; entry < nentries; ++entry ) {

      iom.read_entry(random_entry(iom.get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));
      label[entry] = 1; // Neutrino by default
      ++nu_count;
      auto event_roi = (larcv::EventROI*)(iom.get_data(roi_producer_id));
      for(auto const& roi : event_roi->ROIArray()) {
	if(roi.Type() == ::larcv::kROICosmic) {
	  label[entry] = 0;
	  --nu_count;
	  ++cosmic_count;
	  break;
	}
      }
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ch].as_vector();
	auto const& mean_img  = rh.mean_imgs[ch];

	size_t len = input_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  if(!use_flat_mean)
	    data[idx] = input_img[j] - mean_img[j] - rh.imin_v[ch];
	  else
	    data[idx] = input_img[j] - rh.img_means[ch] - rh.imin_v[ch];

	  if ( data[idx] < 0             ) data[idx] = 0;
	  if ( data[idx] > rh.imax_v[ch] ) data[idx] = rh.imax_v[ch];
	  
	}
      }
    }

    LOG(INFO) << "\t>> loading " << nu_count << " neutrinos, " << cosmic_count << " cosmics...\n";
    
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
    	      << " with memory size " << data.size() * sizeof(float)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
    	      << " with memory size " << label.size() * sizeof(float)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(float) );    
  }

  template <>
  void root_load_data<double>(root_helper& rh,
			     Blob<double>* data_blob,
			     Blob<double>* label_blob) {

    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);

    const ::larcv::ProducerID_t image_producer_id = iom.producer_id(::larcv::kProductImage2D,rh.image_producer);
    const ::larcv::ProducerID_t roi_producer_id = iom.producer_id(::larcv::kProductROI,rh.roi_producer);
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));

    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = rh.nentries;

    const auto& im = ev_data->Image2DArray();
    const auto& meta = im[0].meta();

    const int nchannels = im.size();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; 
    data_dims[1]  = nchannels;
    data_dims[2]  = meta.rows(); 
    data_dims[3]  = meta.cols();
    
    label_dims[0] = nentries;
   
    data_blob->Reshape(data_dims);   
    label_blob->Reshape(label_dims);

    std::vector<double> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );

    std::vector<double> label; 
    label.resize( nentries );

    bool use_flat_mean = rh.mean_imgs.empty();

    size_t cosmic_count=0;
    size_t nu_count=0;

    for(int entry = 0; entry < nentries; ++entry ) {

      iom.read_entry(random_entry(iom.get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));
      label[entry] = 1; // Neutrino by default
      ++nu_count;
      auto event_roi = (larcv::EventROI*)(iom.get_data(roi_producer_id));
      for(auto const& roi : event_roi->ROIArray()) {
	if(roi.Type() == ::larcv::kROICosmic) {
	  label[entry] = 0;
	  --nu_count;
	  ++cosmic_count;
	  break;
	}
      }
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ ch ].as_vector();
	auto const& mean_img  = rh.mean_imgs[ ch ];

	size_t len = input_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  if(!use_flat_mean)
	    data[ idx ] = input_img[j] - mean_img[j] - rh.imin_v[ch];
	  else
	    data[ idx ] = input_img[j] - rh.img_means[ch] - rh.imin_v[ch];

	  if ( data[ idx ] < 0             ) data[ idx ] = 0.;
	  if ( data[ idx ] > rh.imax_v[ch] ) data[ idx ] = rh.imax_v[ch];
	  
	}
      }
    }

    LOG(INFO) << "\t>> loading " << nu_count << " neutrinos, " << cosmic_count << " cosmics...\n";
    
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(double)  << " (double)\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(double)  << " (double)\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(double) );
    
  }

}  // namespace caffe

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
  /*
  template <>
  void root_load_data<float>(root_helper& rh,
			     Blob<float>* data_blob,
			     Blob<float>* label_blob) {
    //auto& iom = rh.iom;
    //::larcv::IOManager iom(::larcv::IOManager::kREAD,"IOData");
    //iom.add_in_file(rh.filename);
    //iom.initialize();
    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);
    
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));

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

    std::vector<uint8_t> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );

    std::vector<uint8_t> label; 
    label.resize( nentries );

    bool use_flat_mean = rh.mean_imgs.empty();
    for(int entry = 0; entry < nentries; ++entry ) {

      iom.read_entry(random_entry(iom.get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));
      label[entry] = (((::larcv::EventROI*)(iom.get_data(::larcv::kProductROI,rh.producer)))->ROIArray().size() > 0 ? 1 : 0);
      //label[entry] = background ? 0 : 1; // class
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ch].as_vector();
	auto const& mean_img  = rh.mean_imgs[ch];

	size_t len = input_img.size();

	float val = 0;

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;

	  if(!use_flat_mean)
	    val = (input_img[j] - mean_img[j] - rh.imin)/rh.imax;
	  else
	    val = (input_img[j] - rh.img_means[ch] - rh.imin)/rh.imax;

	  if( val<0 ) val = 0;
	  if ( val > rh.imax ) data[idx] = 1.0;
	  
	  data[idx] = (uint8_t)((int)(255. * val + 0.5));
	}
      }
    }
    iom.finalize();
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(uint8_t)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(uint8_t) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(uint8_t)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(uint8_t) );
    
  }
  */
  template <>
  void root_load_data<float>(root_helper& rh,
			     Blob<float>* data_blob,
			     Blob<float>* label_blob) {
    //auto& iom = rh.iom;
    //::larcv::IOManager iom(::larcv::IOManager::kREAD,"IOData");
    //iom.add_in_file(rh.filename);
    //iom.initialize();
    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);
    
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));

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

    bool use_flat_mean = rh.mean_imgs.empty();

    for(int entry = 0; entry < nentries; ++entry ) {

      iom.read_entry(random_entry(iom.get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));
      label[entry] = (((::larcv::EventROI*)(iom.get_data(::larcv::kProductROI,rh.producer)))->ROIArray().size() > 0 ? 1 : 0);
      //label[entry] = background ? 0 : 1; // class
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ch].as_vector();
	auto const& mean_img  = rh.mean_imgs[ch];

	size_t len = input_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  if(!use_flat_mean)
	    data[idx] = input_img[j] - mean_img[j] - rh.imin;
	  else
	    data[idx] = input_img[j] - rh.img_means[ch] - rh.imin;

	  if ( data[idx] < 0       ) data[idx] = 0;
	  if ( data[idx] > rh.imax ) data[idx] = rh.imax;
	  
	}
      }
    }
    //iom.finalize();
    /*
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(float)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(float) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(float)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(float) );
    */    
  }

  template <>
  void root_load_data<double>(root_helper& rh,
			     Blob<double>* data_blob,
			     Blob<double>* label_blob) {
    //auto& iom = rh.iom;
    //::larcv::IOManager iom(::larcv::IOManager::kREAD,"IOData");
    //iom.add_in_file(rh.filename);
    //iom.initialize();
    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);
    
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));

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

    std::vector<double> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );

    std::vector<double> label; 
    label.resize( nentries );

    bool use_flat_mean = rh.mean_imgs.empty();

    for(int entry = 0; entry < nentries; ++entry ) {

      iom.read_entry(random_entry(iom.get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom.get_data(::larcv::kProductImage2D,rh.producer));
      label[entry] = (((::larcv::EventROI*)(iom.get_data(::larcv::kProductROI,rh.producer)))->ROIArray().size() > 0 ? 1 : 0);
      //label[entry] = background ? 0 : 1; // class
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ ch ].as_vector();
	auto const& mean_img  = rh.mean_imgs[ ch ];

	size_t len = input_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  if(!use_flat_mean)
	    data[ idx ] = input_img[j] - mean_img[j] - rh.imin;
	  else
	    data[ idx ] = input_img[j] - rh.img_means[ch] - rh.imin;

	  if ( data[ idx ] < 0       ) data[ idx ] = 0.;
	  if ( data[ idx ] > rh.imax ) data[ idx ] = rh.imax;
	  
	}
      }
    }
    iom.finalize();
    /*
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(double)  << "\n";
    */
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );
    /*
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(double)  << "\n";
    */
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(double) );
    
  }

  /*
  template <>
  void root_load_data<double>(root_helper& rh,
			      Blob<double>* data_blob,
			      Blob<double>* label_blob) {
    
    auto& iom = rh.iom;

    iom->read_entry(0);
    
    auto ev_data = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,rh.producer));

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

    std::vector<double> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );

    std::vector<double> label; 
    label.resize( nentries );

    bool use_flat_mean = rh.mean_imgs.empty();

    for(int entry = 0; entry < nentries; ++entry ) {

      iom->read_entry(random_entry(iom->get_n_entries()));
      ev_data  = (::larcv::EventImage2D*)(iom->get_data(::larcv::kProductImage2D,rh.producer));
      label[entry] = (((::larcv::EventROI*)(iom->get_data(::larcv::kProductROI,rh.producer)))->ROIArray().size() > 0 ? 1 : 0);
      //label[entry] = background ? 0 : 1; // class
      
      auto const& input_img_v = ev_data->Image2DArray();
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto const& input_img = input_img_v[ ch ].as_vector();
	auto const& mean_img  = rh.mean_imgs[ ch ];

	size_t len = input_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch)*len + j;
	  
	  if(!use_flat_mean)
	    data[ idx ] = input_img[j] - mean_img[j];
	  else
	    data[ idx ] = input_img[j] - rh.img_means[ch];

	  if ( data[ idx ] < rh.imin ) data[ idx ] = 0;
	  if ( data[ idx ] > rh.imax ) data[ idx ] = rh.imax;
	  
	}
      }
    }
    
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
	      << " with memory size " << data.size() * sizeof(double)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
	      << " with memory size " << label.size() * sizeof(double)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(double) );
    
  }

  */
}  // namespace caffe

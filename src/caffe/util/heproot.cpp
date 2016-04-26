#include "caffe/util/heproot.hpp"
#include "caffe/util/larcv_iom_singleton.hpp"
#include <string>
#include <vector>
//LArCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace caffe {

  size_t random_size(size_t nmax)
  {
    static const double rmax = (double)RAND_MAX;
    return (size_t)(( (double)(rand()) / rmax ) * nmax);
  }

  void fill_random_gaus(double mean, double sigma, std::vector<double>& pool)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean,sigma);
    for(size_t i=0; i<pool.size(); ++i) pool[i] = d(gen);
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

    auto const& im = ev_data->Image2DArray();
    auto const& meta = im[0].meta();

    auto const col_pad = rh.random_col_pad;
    auto const row_pad = rh.random_row_pad;

    const int nchannels = im.size();

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; 
    data_dims[1]  = nchannels;
    data_dims[2]  = meta.rows() + row_pad; 
    data_dims[3]  = meta.cols() + col_pad;

    auto canvas_meta = larcv::ImageMeta(meta.width() + col_pad * meta.pixel_width(),
					meta.height() + row_pad * meta.pixel_height(),
					data_dims[2], data_dims[3],
					meta.min_x(), meta.max_y());
    auto canvas = larcv::Image2D(canvas_meta);

    label_dims[0] = nentries;
    LOG(INFO) << "Reshape data";    
    data_blob->Reshape(data_dims);  
    LOG(INFO) << "Reshape label";     
    label_blob->Reshape(label_dims);

    LOG(INFO) << "Resize data";    
    std::vector<float> data; 
    data.resize(nentries * nchannels * data_dims[2] * data_dims[3]);
    LOG(INFO) << "Resize label";    
    std::vector<float> label; 
    label.resize( nentries );

    size_t cosmic_count=0;
    size_t nu_count=0;

    std::vector<double> adc_scale_v(nentries,1.0);
    if( rh.random_adc_scale_sigma > 0.0 ) 

      fill_random_gaus(rh.random_adc_scale_mean, rh.random_adc_scale_sigma, adc_scale_v);

    bool use_flat_mean = rh.mean_imgs.empty();
    LOG(INFO) << "Reading nentries: " << nentries;
    for(int entry = 0; entry < nentries; ++entry ) {
      float adc_scale_factor = (float)(adc_scale_v[entry]);
      iom.read_entry(random_size(iom.get_n_entries()));

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

      ev_data  = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));
      std::vector<larcv::Image2D> input_img_v;
      ev_data->Move(input_img_v);

      size_t row_shift = ( row_pad ? random_size(row_pad) : 0 );
      size_t col_shift = ( col_pad ? random_size(col_pad) : 0 );
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto& input_img = input_img_v[ch];
	auto const& mean_img  = rh.mean_imgs[ch];

	if(!use_flat_mean)
	  input_img -= mean_img;
	else
	  input_img -= rh.img_means[ch];

	input_img -= rh.imin_v[ch];

	canvas.reset_origin(input_img.meta().min_x(), input_img.meta().max_y());
	canvas.paint(0.);

	if(row_shift || col_shift) {
	  canvas.reset_origin(input_img.meta().min_x() - col_shift * input_img.meta().pixel_width(),
			      input_img.meta().max_y() + row_shift * input_img.meta().pixel_height());

	}
	canvas.overlay(input_img);
	
	auto crop_meta = larcv::ImageMeta(input_img.meta().width(), input_img.meta().height(),
					  input_img.meta().rows(),  input_img.meta().cols(),
					  canvas.meta().min_x(),    canvas.meta().max_y(),
					  input_img.meta().plane());
	auto cropped = canvas.crop(crop_meta);
	auto const& canvas_img = cropped.as_vector();

	size_t len = canvas_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch ) * len + j;
	  
	  if ( data[idx] < 0             ) data[idx] = 0;

	  data[idx] *= adc_scale_factor;

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
    LOG(INFO) << "Getting singleton";
    auto& iom = ::larcv::SingleIOManager::get(rh.filename).manager;
    iom.read_entry(0);

    const ::larcv::ProducerID_t image_producer_id = iom.producer_id(::larcv::kProductImage2D,rh.image_producer);
    const ::larcv::ProducerID_t roi_producer_id = iom.producer_id(::larcv::kProductROI,rh.roi_producer);
    auto ev_data = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));

    //load data dimensions and reshape the data_blob

    //should be long?
    int nentries = rh.nentries;

    auto const& im = ev_data->Image2DArray();
    auto const& meta = im[0].meta();

    auto const col_pad = rh.random_col_pad;
    auto const row_pad = rh.random_row_pad;

    const int nchannels = im.size();

    if(nchannels != rh.imin_v.size()) throw ::larcv::larbys("# channels do not match with imin parameter length!");

    std::vector<int> data_dims(4), label_dims(1);
    data_dims[0]  = nentries; 
    data_dims[1]  = nchannels;
    data_dims[2]  = meta.rows() + row_pad; 
    data_dims[3]  = meta.cols() + col_pad;

    auto canvas_meta = larcv::ImageMeta(meta.width() + col_pad * meta.pixel_width(),
					meta.height() + row_pad * meta.pixel_height(),
					data_dims[2], data_dims[3],
					meta.min_x(), meta.max_y());
    auto canvas = larcv::Image2D(canvas_meta);

    label_dims[0] = nentries;
    LOG(INFO) << "Reshape data";    
    data_blob->Reshape(data_dims);  
    LOG(INFO) << "Reshape label";     
    label_blob->Reshape(label_dims);

    LOG(INFO) << "Resize data";    
    std::vector<double> data; 
    data.resize(nentries * nchannels * im[0].as_vector().size() );
    LOG(INFO) << "Resize label";    
    std::vector<double> label; 
    label.resize( nentries );

    size_t cosmic_count=0;
    size_t nu_count=0;

    std::vector<double> adc_scale_v(nentries,1.0);
    if( rh.random_adc_scale_sigma > 0.0 ) 

      fill_random_gaus(rh.random_adc_scale_mean, rh.random_adc_scale_sigma, adc_scale_v);

    bool use_flat_mean = rh.mean_imgs.empty();
    LOG(INFO) << "Reading nentries: " << nentries;
    for(int entry = 0; entry < nentries; ++entry ) {
      double adc_scale_factor = (double)(adc_scale_v[entry]);
      iom.read_entry(random_size(iom.get_n_entries()));

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

      ev_data  = (::larcv::EventImage2D*)(iom.get_data(image_producer_id));
      std::vector<larcv::Image2D> input_img_v;
      ev_data->Move(input_img_v);
      
      for(size_t ch=0;ch<nchannels;++ch) {
	
	auto& input_img = input_img_v[ch];
	auto const& mean_img  = rh.mean_imgs[ch];

	if(!use_flat_mean)
	  input_img -= mean_img;
	else
	  input_img -= rh.img_means[ch];

	input_img -= rh.imin_v[ch];

	canvas.reset_origin(input_img.meta().min_x(), input_img.meta().max_y());
	canvas.paint(0.);

	if(canvas.meta().rows() != input_img.meta().rows() ||
	   canvas.meta().cols() != input_img.meta().cols() ) {
	  
	  size_t row_shift = random_size(row_pad);
	  size_t col_shift = random_size(col_pad);

	  canvas.reset_origin(input_img.meta().min_x() - col_shift * input_img.meta().pixel_width(),
			      input_img.meta().max_y() + row_shift * input_img.meta().pixel_height());
	}
	canvas.overlay(input_img);
	auto const& canvas_img = canvas.as_vector();

	size_t len = canvas_img.size();

	for(size_t j=0;j<len;++j)  {

	  auto idx =  ( entry * nchannels + ch ) * len + j;
	  
	  if ( data[idx] < 0             ) data[idx] = 0;

	  data[idx] *= adc_scale_factor;

	  if ( data[idx] > rh.imax_v[ch] ) data[idx] = rh.imax_v[ch];
	  
	}
      }
    }

    LOG(INFO) << "\t>> loading " << nu_count << " neutrinos, " << cosmic_count << " cosmics...\n";
    
    LOG(INFO) << "\t>> memcpy with data.size() " << data.size() 
    	      << " with memory size " << data.size() * sizeof(double)  << "\n";
    memcpy(data_blob->mutable_cpu_data(),data.data(),data.size() * sizeof(double) );
    
    LOG(INFO) << "\t>> memcpy with label.size() " << label.size() 
    	      << " with memory size " << label.size() * sizeof(double)  << "\n";
    memcpy(label_blob->mutable_cpu_data(),label.data(),label.size() * sizeof(double) );    
  }

}  // namespace caffe

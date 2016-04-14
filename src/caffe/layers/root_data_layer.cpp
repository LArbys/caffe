/*
  TODO:
  - do everything
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "stdint.h"

// LArCV
#include "DataFormat/IOManager.h"

//
#include "caffe/layers/root_data_layer.hpp"
#include "caffe/util/heproot.hpp"

namespace caffe {
  
  template <typename Dtype>
  ROOTDataLayer<Dtype>::~ROOTDataLayer<Dtype>() { }
  
  // Load data and label from ROOT filename into the class property blobs.
  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LoadROOTFileData(std::pair<std::string,std::string>& file_producer) {

    auto& filename = file_producer.first;
    auto& producer = file_producer.second;
    LOG(INFO) << "Loading ROOT file: " << filename << " with producer: " << producer << "\n";
    
    // _iom.set_verbosity(::larcv::msg::kDEBUG);

    _iom.add_in_file(filename);
    _iom.initialize();

    root_helper rh;
    
    int top_size = this->layer_param_.top_size();
    root_blobs_.resize(top_size);

    // should only be size 2: data and label, but user 
    // could put them in any order...
    for (int i = 0; i < top_size; ++i) 
      
      root_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

    rh.iom = & _iom;
    rh.producer   = producer;
    rh.background = producer == "data" ? true : false;
    
    rh.nentries = this->layer_param_.root_data_param().nentries();
    
    rh.imin = this->layer_param_.root_data_param().imin();
    rh.imax = this->layer_param_.root_data_param().imax();
    
    std::vector<float> immeans = { this->layer_param_.root_data_param().ch0_mean(),
				   this->layer_param_.root_data_param().ch1_mean(),
				   this->layer_param_.root_data_param().ch2_mean() };
    
    rh.img_means = immeans;

    root_load_data(rh,
		   root_blobs_[0].get(),
		   root_blobs_[1].get());
    
    _iom.finalize();

    // MinTopBlobs==1 guarantees at least one top blob
    CHECK_GE(root_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
    const int num = root_blobs_[0]->shape(0);
    for (int i = 1; i < top_size; ++i) {
      CHECK_EQ(root_blobs_[i]->shape(0), num);
    }
    // Default to identity permutation.
    data_permutation_.clear();
    data_permutation_.resize(root_blobs_[0]->shape(0));
    for (int i = 0; i < root_blobs_[0]->shape(0); i++)
      data_permutation_[i] = i;

    // Shuffle if needed.
    if (this->layer_param_.root_data_param().shuffle()) {
      std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
      DLOG(INFO) << "Successully loaded " << root_blobs_[0]->shape(0)
		 << " rows (shuffled)";
    } else {
      DLOG(INFO) << "Successully loaded " << root_blobs_[0]->shape(0) << " rows";
    }
  }

  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
    // Refuse transformation parameters since ROOT is totally generic.
    CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
    // Read the source to parse the filenames.
    const string& source = this->layer_param_.root_data_param().source();

    root_filenames_.clear();

    LOG(INFO) << "Loading ROOT file list and producers" << source;
    std::ifstream source_file(source.c_str());

    if (source_file.is_open()) {
      std::string word1,word2;
      std::string line;
      int c = -1;
      while (source_file >> line) {
	c+=1;
	if (!c) word1 = line;
	else    word2 = line;
	
	if ( c == 1 ) 
	  { root_filenames_.emplace_back(word1,word2); c=-1; }
	
	LOG(INFO) << "Got something... " << line << "\n";
      }
    } else {
      LOG(FATAL) << "Failed to open source file: " << source;

    }
    source_file.close();

    num_files_ = root_filenames_.size();
    current_file_ = 0;
    LOG(INFO) << "Number of ROOT files:     " << num_files_;

    CHECK_GE(num_files_, 1) << "Must have at least 1 ROOT filename listed in "
    			    << source;
    
    file_permutation_.clear();
    file_permutation_.resize(num_files_);
    // Default to identity permutation.
    for (int i = 0; i < num_files_; i++)
      file_permutation_[i] = i;
    

    // Shuffle if needed.
    if (this->layer_param_.root_data_param().shuffle()) {
      std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
    }

    // Load the first ROOT file and initialize the line counter.
    LoadROOTFileData(root_filenames_[file_permutation_[current_file_]]);
    current_row_ = 0;

    // Reshape blobs.
    const int batch_size = this->layer_param_.root_data_param().batch_size();
    const int top_size = this->layer_param_.top_size();
    vector<int> top_shape;
    for (int i = 0; i < top_size; ++i) {
      top_shape.resize(root_blobs_[i]->num_axes());
      top_shape[0] = batch_size;
      for (int j = 1; j < top_shape.size(); ++j) {
	top_shape[j] = root_blobs_[i]->shape(j);
      }
      top[i]->Reshape(top_shape);
    }
  }

  template <typename Dtype>
  void ROOTDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					 const vector<Blob<Dtype>*>& top) {
    const int batch_size = this->layer_param_.root_data_param().batch_size();
    for (int i = 0; i < batch_size; ++i, ++current_row_) {
      if (current_row_ == root_blobs_[0]->shape(0)) {
	
	if (num_files_ > 1) {
	  ++current_file_;
	  if (current_file_ == num_files_) {
	    current_file_ = 0;
	    if (this->layer_param_.root_data_param().shuffle()) {
	      std::random_shuffle(file_permutation_.begin(),
				  file_permutation_.end());
	    }
	    DLOG(INFO) << "Looping around to first file.";
	  }
	  LoadROOTFileData(root_filenames_[file_permutation_[current_file_]]);
	}
	
	
	current_row_ = 0;
	if (this->layer_param_.root_data_param().shuffle())
	  std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
	
      }
      for (int j = 0; j < this->layer_param_.top_size(); ++j) {
	int data_dim = top[j]->count() / top[j]->shape(0);
	caffe_copy(data_dim,
		   &root_blobs_[j]->cpu_data()[data_permutation_[current_row_]
					       * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
      }
    }
  }

#ifdef CPU_ONLY
  STUB_GPU_FORWARD(ROOTDataLayer, Forward);
#endif

  INSTANTIATE_CLASS(ROOTDataLayer);
  REGISTER_LAYER_CLASS(ROOTData);

}  // namespace caffe

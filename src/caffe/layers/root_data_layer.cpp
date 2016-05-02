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
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"
#include "Base/Parser.h"
//
#include "caffe/layers/root_data_layer.hpp"
#include "caffe/util/heproot.hpp"

namespace caffe {
  
  template <typename Dtype>
  ROOTDataLayer<Dtype>::~ROOTDataLayer<Dtype>() {}//{ _iom.finalize(); }
  
  // Load data and label from ROOT filename into the class property blobs.
  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LoadROOTFileData(const std::string& filename) {

    //Instantiate ThreadDatumFiller only once
    static size_t id = ::larcv::kINVALID_SIZE;
    if(id == ::larcv::kINVALID_SIZE) {

    }

    root_helper rh;

    int top_size = this->layer_param_.top_size();
    root_blobs_.resize(top_size);

    // should only be size 2: data and label, but user 
    // could put them in any order...
    for (int i = 0; i < top_size; ++i) 
      
      root_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

    root_load_data(rh,
		   root_blobs_[0].get(),
		   root_blobs_[1].get());
    
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
      /*
      DLOG(INFO) << "Successully loaded " << root_blobs_[0]->shape(0)
		 << " rows (shuffled)";
      */
    } 
    //else { DLOG(INFO) << "Successully loaded " << root_blobs_[0]->shape(0) << " rows"; }
  }

  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {
    // Refuse transformation parameters since ROOT is totally generic.
    CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";

    // Load the first ROOT file and initialize the line counter.
    LoadROOTFileData();

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
    LoadROOTFileData();
    for (int i = 0; i < batch_size; ++i, ++current_row_) {

      for (int j = 0; j < this->layer_param_.top_size(); ++j) {
	int data_dim = top[j]->count() / top[j]->shape(0);
	caffe_copy(data_dim,
		   &root_blobs_[j]->cpu_data()[i * data_dim], 
		   &top[j]->mutable_cpu_data()[i * data_dim]);
					       
      }
    }
  }

  /*
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
  */
#ifdef CPU_ONLY
  STUB_GPU_FORWARD(ROOTDataLayer, Forward);
#endif

  INSTANTIATE_CLASS(ROOTDataLayer);
  REGISTER_LAYER_CLASS(ROOTData);

}  // namespace caffe

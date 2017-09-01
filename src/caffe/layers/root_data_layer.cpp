/*
  TODO:
  - do everything
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "stdint.h"

// LArCV
#include "APICaffe/ThreadDatumFiller.h"
//
#include "caffe/layers/root_data_layer.hpp"
#include "caffe/util/heproot.hpp"
#include "APICaffe/ThreadDatumFiller.h"
#include "APICaffe/ThreadFillerFactory.h"

namespace caffe {
  
  template <typename Dtype>
  ROOTDataLayer<Dtype>::~ROOTDataLayer<Dtype>() { 
    if(th_.joinable()) th_.join(); 
  }
  
  // Load data and label from ROOT filename into the class property blobs.
  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LoadROOTFileData() {
    io_thread_timer_.start();
    //LOG(INFO) << "Start " << __FUNCTION__ << std::endl;

    size_t batch_size = this->layer_param_.root_data_param().batch_size();
    std::string name  = this->layer_param_.root_data_param().filler_name();
    
    //Instantiate ThreadDatumFiller only once
    bool first_time = !(::larcv::ThreadFillerFactory::exist_filler(name));
    auto& filler = ::larcv::ThreadFillerFactory::get_filler(name);
    if(first_time) {
      filler.configure(this->layer_param_.root_data_param().filler_config());
      // Start read thread
      filler.batch_process(batch_size);
    }
    else if(!filler.thread_config())
      filler.batch_process(batch_size);

    root_helper rh;
    rh._filler_name = name;

    int top_size = this->layer_param_.top_size();
    //LOG(INFO) << "Resizing blob" << std::endl;
    if(!use_blob_buffer_) {
      root_blobs_.resize(top_size);
      
      // should only be size 2: data and label, but user 
      // could put them in any order...
      //LOG(INFO) << "Filling empty blob" << std::endl;
      if(top_size != 2 && top_size != 3) {
	LOG(ERROR) << "Top size " << top_size << "is invalid!" << std::endl;
	throw std::exception();
      }

      for (int i = 0; i < top_size; ++i) 
	
	root_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

      //LOG(INFO) << "Calling root_load_data" << std::endl;
      if(time_report_) {
	io_thread_copy_timer_.start();
	if(top_size==2)
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get());
	else
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get(), root_blobs_[2].get());
	io_thread_copy_time_ += io_thread_copy_timer_.wall_time();
      }else{
	if(top_size==2)
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get());
	else
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get(), root_blobs_[2].get());
      }

      // MinTopBlobs==1 guarantees at least one top blob
      CHECK_GE(root_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
      const int num = root_blobs_[0]->shape(0);
      for (int i = 1; i < top_size; ++i) {
	CHECK_EQ(root_blobs_[i]->shape(0), num);
      }

    }else{

      //for(size_t i=0; i<cbuffer_.size(); ++i)
      //LOG(INFO) << "buffer " << i << " @ state " << cbuffer_.state(i) << std::endl;

      // first of all, if current buffer is valid, start clear thread
      if(cbuffer_.state(current_buffer_id_) == caffe::heproot::kReady)
	cbuffer_.flush(current_buffer_id_);

      // make sure the next buffer is ready
      size_t sleep_ctr=0;
      while(cbuffer_.busy(next_buffer_id_)) {
	usleep(100);
	sleep_ctr += 100;
	if(sleep_ctr && sleep_ctr%1000000 == 0)
	  LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to allocate blob buffer..." << std::endl;
      }

      if(cbuffer_.get(next_buffer_id_).size() != top_size) {

	LOG(WARNING) << "Top shape changed... (" << cbuffer_.get(next_buffer_id_).size() 
		     << " => " << top_size << ") reallocating Blobs..." << std::endl;
	if(cbuffer_.state(next_buffer_id_) == caffe::heproot::kReady)
	  cbuffer_.flush(next_buffer_id_);

	while(cbuffer_.busy(next_buffer_id_)) {
	  usleep(100);
	  sleep_ctr += 100;
	  if(sleep_ctr && sleep_ctr%1000000 == 0)
	    LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to clean blob buffer..." << std::endl;
	}

	cbuffer_.init(next_buffer_id_, top_size);

	while(cbuffer_.busy(next_buffer_id_)) {
	  usleep(100);
	  sleep_ctr += 100;
	  if(sleep_ctr && sleep_ctr%1000000 == 0)
	    LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to allocate blob buffer..." << std::endl;
	}
      }

      auto& root_blobs = cbuffer_.get(next_buffer_id_);
      //LOG(INFO) << "Calling root_load_data" << std::endl;
      if(time_report_) {
	io_thread_copy_timer_.start();
	if(top_size==2)
	  root_load_data(rh, root_blobs[0].get(), root_blobs[1].get());
	else
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get(), root_blobs_[2].get());
	io_thread_copy_time_ += io_thread_copy_timer_.wall_time();
      }else{
	if(top_size==2)
	  root_load_data(rh, root_blobs[0].get(), root_blobs[1].get());
	else
	  root_load_data(rh, root_blobs_[0].get(), root_blobs_[1].get(), root_blobs_[2].get());
      }
      // MinTopBlobs==1 guarantees at least one top blob
      CHECK_GE(root_blobs[0]->num_axes(), 1) << "Input must have at least 1 axis.";
      const int num = root_blobs[0]->shape(0);
      for (int i = 1; i < top_size; ++i) {
	CHECK_EQ(root_blobs[i]->shape(0), num);
      }

      //for(size_t i=0; i<cbuffer_.size(); ++i)
      //LOG(INFO) << "buffer " << i << " @ state " << cbuffer_.state(i) << std::endl;

      current_buffer_id_ = next_buffer_id_;
      next_buffer_id_ = (next_buffer_id_ ? 0 : 1);
      //LOG(INFO) << "current " << current_buffer_id_ << " next " << next_buffer_id_ << std::endl;
      while(cbuffer_.busy(next_buffer_id_)) {
	usleep(100);
	sleep_ctr += 100;
	if(sleep_ctr && sleep_ctr%1000000 == 0)
	  LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to be cleared for the next cycle..." << std::endl;
      }
      cbuffer_.init(next_buffer_id_,top_size);
    }

    if(filler.thread_config())
      filler.batch_process(batch_size);

    if(time_report_)
      io_thread_time_ += io_thread_timer_.wall_time();
    
    //LOG(INFO) << "Finished " << __FUNCTION__ << std::endl;
    //else { DLOG(INFO) << "Successully loaded " << root_blobs_[0]->shape(0) << " rows"; }
  }

  template <typename Dtype>
  void ROOTDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					const vector<Blob<Dtype>*>& top) {

    // Refuse transformation parameters since ROOT is totally generic.
    CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";

    time_report_ = this->layer_param_.root_data_param().report_time();
    use_blob_buffer_ = this->layer_param_.root_data_param().use_circular_buffer();
    use_thread_ = this->layer_param_.root_data_param().use_thread();
    next_buffer_id_ = 0;
    current_buffer_id_ = 0;
    if(use_blob_buffer_) {
      size_t sleep_ctr=0;
      cbuffer_.init(current_buffer_id_, this->layer_param_.top_size());
      while(cbuffer_.busy(current_buffer_id_)) {
	usleep(100);
	sleep_ctr += 100;
	if(sleep_ctr && sleep_ctr%1000000 == 0)
	  LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to allocate blob buffer..." << std::endl;
      }
    }

    // Load the first ROOT file and initialize the line counter.
    LoadROOTFileData();

    // reset timer for LoadROOTFileData
    io_thread_time_ = io_thread_copy_time_ = 0.;

    // Reshape blobs.
    const int batch_size = this->layer_param_.root_data_param().batch_size();
    const int top_size = this->layer_param_.top_size();
    vector<int> top_shape;
    for (int i = 0; i < top_size; ++i) {
      if(!use_blob_buffer_) {
	top_shape.resize(root_blobs_[i]->num_axes());
	top_shape[0] = batch_size;
	for (int j = 1; j < top_shape.size(); ++j) {
	  top_shape[j] = root_blobs_[i]->shape(j);
	}
	top[i]->Reshape(top_shape);
      }else{
	auto& root_blobs = cbuffer_.get(current_buffer_id_);
	top_shape.resize(root_blobs[i]->num_axes());
	top_shape[0] = batch_size;
	for (int j = 1; j < top_shape.size(); ++j) {
	  top_shape[j] = root_blobs[i]->shape(j);
	}
	top[i]->Reshape(top_shape);
      }
    }

    // now reset buffer
    if(use_blob_buffer_) {
      for(size_t i=0; i<cbuffer_.size(); ++i) {
	size_t sleep_ctr=0;
	cbuffer_.flush(i);
	while(cbuffer_.busy(i)) {
	  usleep(100);
	  sleep_ctr += 100;
	  if(sleep_ctr && sleep_ctr%1000000 == 0)
	    LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to re-initialize blob buffer..." << std::endl;
	}
	LOG(INFO) << "buffer " << i << " @ state " << cbuffer_.state(i) << std::endl;
      }

      current_buffer_id_ = 0;
      next_buffer_id_ = 0;

      size_t sleep_ctr=0;
      cbuffer_.init(current_buffer_id_, this->layer_param_.top_size());
      while(cbuffer_.busy(current_buffer_id_)) {
	usleep(100);
	sleep_ctr += 100;
	if(sleep_ctr && sleep_ctr%1000000 == 0)
	  LOG(INFO) << "Waiting for " << sleep_ctr/1000000 << " [s] to allocate blob buffer..." << std::endl;
      }
    }

  }

  template <typename Dtype>
  void ROOTDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					 const vector<Blob<Dtype>*>& top) {

    const int batch_size = this->layer_param_.root_data_param().batch_size();
    if(time_report_>0) io_wait_timer_.start();
    if(use_thread_) {
      if(!th_.joinable()) {
	std::thread t(&ROOTDataLayer<Dtype>::LoadROOTFileData, this);
	th_ = std::move(t);
      }
      th_.join();
    }else{
      LoadROOTFileData();
    }
    if(time_report_>0) {
      io_wait_time_ += io_wait_timer_.wall_time();
      io_wait_timer_.start();
    }
    for (int i = 0; i < batch_size; ++i) {
      
      for (int j = 0; j < this->layer_param_.top_size(); ++j) {
	int data_dim = top[j]->count() / top[j]->shape(0);
	if(use_blob_buffer_) {
	  auto& root_blobs = cbuffer_.get(current_buffer_id_);
	  caffe_copy(data_dim,
		     &root_blobs[j]->cpu_data()[i * data_dim], 
		     &top[j]->mutable_cpu_data()[i * data_dim]);
	}else{
	  caffe_copy(data_dim,
		     &root_blobs_[j]->cpu_data()[i * data_dim], 
		     &top[j]->mutable_cpu_data()[i * data_dim]);
	}
      }
    }
    ++io_counter_;
    if(time_report_>0) {
      caffe_copy_time_ += io_wait_timer_.wall_time();
      
      if(io_counter_%time_report_ == 0) {
	LOG(INFO) << "Time report (" << io_counter_ << " IO done): "  << std::endl;
	LOG(INFO) << "    Total lifetime .....: " << life_timer_.wall_time() << " [s]" << std::endl;
	LOG(INFO) << "    IO wait ............: " << io_wait_time_
		  << " [s] ... average: " << io_wait_time_ / ((double)(io_counter_)) << std::endl;
	LOG(INFO) << "    IO thread ..........: " << io_thread_time_
		  << " [s] ... average: " << io_thread_time_ / ((double)(io_counter_)) << std::endl;
	LOG(INFO) << "    IO thread (copy) ...: " << io_thread_copy_time_
		  << " [s] ... average: " << io_thread_copy_time_ / ((double)(io_counter_)) << std::endl;
	LOG(INFO) << "    GPU caffe copy .....: " << caffe_copy_time_
		  << " [s] ... average: " << caffe_copy_time_ / ((double)(io_counter_)) << std::endl;
      }	
    }
    if(use_thread_) {
      std::thread t(&ROOTDataLayer<Dtype>::LoadROOTFileData, this);
      th_ = std::move(t);
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

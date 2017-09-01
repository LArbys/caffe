/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "caffe/layers/root_data_layer.hpp"
namespace caffe {
template <typename Dtype>
void ROOTDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
		   &top[j]->mutable_gpu_data()[i * data_dim]);
      }else{
	caffe_copy(data_dim,
		   &root_blobs_[j]->cpu_data()[i * data_dim], 
		   &top[j]->mutable_gpu_data()[i * data_dim]);
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
INSTANTIATE_LAYER_GPU_FUNCS(ROOTDataLayer);

}  // namespace caffe

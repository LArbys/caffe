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
}

INSTANTIATE_LAYER_GPU_FUNCS(ROOTDataLayer);

}  // namespace caffe

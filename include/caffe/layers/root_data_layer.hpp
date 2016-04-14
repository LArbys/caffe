#ifndef CAFFE_ROOT_DATA_LAYER_HPP_
#define CAFFE_ROOT_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

#include "DataFormat/IOManager.h"

namespace caffe {

  /**
   * @brief Provides data to the Net from ROOT files.
   *
   * TODO(dox): thorough documentation for Forward and proto params.
   */
  template <typename Dtype>
  class ROOTDataLayer : public Layer<Dtype> {

  public:
    explicit ROOTDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) , _iom() {}

    virtual ~ROOTDataLayer();

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			    const vector<Blob<Dtype>*>& top);

    // Data layers should be shared by multiple solvers in parallel
    virtual inline bool ShareInParallel() const { return true; }

    // Data layers have no bottoms, so reshaping is trivial.
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			 const vector<Blob<Dtype>*>& top) {}

    virtual inline const char* type() const { return "ROOTData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int MinTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
    virtual void LoadROOTFileData(std::pair<std::string,std::string>& file_producer);
    
    std::vector<std::pair<std::string,std::string> > root_filenames_;
    
    unsigned int num_files_;
    unsigned int current_file_;
    unsigned int current_row_;

    std::vector<shared_ptr<Blob<Dtype> > > root_blobs_;
    std::vector<unsigned int> data_permutation_;
    std::vector<unsigned int> file_permutation_;
    
    ::larcv::IOManager _iom;
    
  };

}  // namespace caffe

#endif  // CAFFE_ROOT_DATA_LAYER_HPP_

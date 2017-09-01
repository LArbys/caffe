#ifndef CAFFE_ROOT_DATA_LAYER_HPP_
#define CAFFE_ROOT_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/heproot_circular_buffer.hpp"
#include "caffe/util/heproot_circular_buffer.imp.hpp"
#include "caffe/util/heproot_timer.hpp"
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
    //: Layer<Dtype>(param) , _iom(::larcv::IOManager::kREAD,"IOData") , _mean_imgs() {}
      : Layer<Dtype>(param)
    {
      use_thread_  = false;
      time_report_ = 0;
      life_timer_.start();
      io_thread_time_ = io_wait_time_ = io_thread_copy_time_ = caffe_copy_time_ = 0.;
      io_counter_ = 0;
    }

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
    virtual void LoadROOTFileData();
    
    std::vector<shared_ptr<Blob<Dtype> > > root_blobs_;

    heproot::circular_buffer<Dtype> cbuffer_;
    
    size_t next_buffer_id_;
    size_t current_buffer_id_;
    bool   use_blob_buffer_;

    int time_report_;
    std::thread th_;
    bool use_thread_;
    size_t io_counter_;     ///< counter of IO operations

    heproot::timer life_timer_;

    heproot::timer io_wait_timer_;        ///< timer to measure wait time of io thread in the main thread
    heproot::timer io_thread_timer_;      ///< timer to measure io thread duration
    heproot::timer io_thread_copy_timer_; ///< timer to measure io thread duration fraction for preparing blob and copying data
    double io_wait_time_;                 ///< time spent in main thred to wait for io thread to return
    double io_thread_time_;               ///< time spent within io thread
    double io_thread_copy_time_;          ///< time spent within io thread for preparing blob and copying data
    double caffe_copy_time_;              ///< time spent to copy cpu data into gpu memory
  };

}  // namespace caffe

#endif  // CAFFE_ROOT_DATA_LAYER_HPP_

#ifndef HEPROOT_CIRCULAR_BUFFER_H
#define HEPROOT_CIRCULAR_BUFFER_H

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include "caffe/blob.hpp"

namespace caffe {

  namespace heproot {

    enum BufferState_t {
      kEmpty,
      kInit,
      kReady,
      kClean,
      kUndefined
    };
    
    /**
       \class heproot_circular_buffer
    */
    template <class Dtype>
    class circular_buffer {
      
    public:
      
      /// Default constructor
      circular_buffer(size_t num_buf=2);
      
      /// Default destructor
      virtual ~circular_buffer();
      
      size_t size() const { return _buffer_v.size(); }

      BufferState_t state(const size_t id) const;
      
      bool flush(const size_t id);
      
      bool init(const size_t id, const size_t num_blob);
      
      bool busy(const size_t id, const bool raise=false) const;
      
      std::vector<std::shared_ptr<caffe::Blob<Dtype> > >& get(const size_t id);
      
      void lock(const size_t id);
      
      void unlock(const size_t id);
      
    protected:
      
      std::vector<std::vector<std::shared_ptr<caffe::Blob<Dtype> > > > _buffer_v;
      
    private:
      
      void check_id(const size_t id) const;
      
      void fill_buffer(const size_t id, const size_t num_blob);
      
      void clean_buffer(const size_t id);
      
      std::vector<caffe::heproot::BufferState_t> _state_v;
      std::vector<bool> _lock_v;
      std::vector<std::thread> _thread_v;
      
    };
  }
}

#endif
/** @} */ // end of doxygen group 


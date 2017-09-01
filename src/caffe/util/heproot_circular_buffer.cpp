#ifndef HEPROOT_CIRCULAR_BUFFER_CXX
#define HEPROOT_CIRCULAR_BUFFER_CXX

#include "caffe/util/heproot_circular_buffer.hpp"

namespace caffe {
  namespace heproot {
    
    template <class Dtype>
    circular_buffer<Dtype>::circular_buffer(size_t num_buf)
      : _buffer_v ( num_buf          )
      , _state_v  ( num_buf, kEmpty  )
      , _lock_v   ( num_buf, false   )
      , _thread_v ( num_buf          )
    {}
    
    template <class Dtype>
    circular_buffer<Dtype>::~circular_buffer()
    {
      // loop over buffers
      for(size_t i=0; i<_buffer_v.size(); ++i) {
	// if thread is running, wait
	if(_thread_v[i].joinable())
	  _thread_v[i].join();
	// then flush
	if( _state_v[i] == kReady)
	  flush(i);
      }
      // make sure destructor holds till flushing is done
      for(size_t i=0; i<_buffer_v.size(); ++i)
	if(_thread_v[i].joinable()) _thread_v[i].join();
      
      /*
	for(size_t i=0; i<_thread_v.size(); ++i) {
	std::cout<<i<<" ... state " << _state_v[i]<< " ... "<<(_thread_v[i].joinable() ? "joinable" : "not joinable")<<std::endl;
	}
      */
    }
    
    template <class Dtype>
    void circular_buffer<Dtype>::lock(const size_t id)
    {
      busy(id,true);
      _lock_v[id] = true;
    }
    
    template <class Dtype>
    void circular_buffer<Dtype>::unlock(const size_t id)
    {
      busy(id,true);
      _lock_v[id] = true;
    }
    
    template <class Dtype>
    BufferState_t circular_buffer<Dtype>::state(const size_t id) const
    {
      check_id(id);
      return _state_v[id];
    }
    
    template <class Dtype>
    bool circular_buffer<Dtype>::flush(const size_t id)
    {
      busy(id);
      if(_thread_v[id].joinable()) _thread_v[id].join();
      std::thread t(&circular_buffer<Dtype>::clean_buffer, this, id);
      _thread_v[id] = std::move(t);
      usleep(500);
      return true;
    }
    
    template <class Dtype>
    bool circular_buffer<Dtype>::init(const size_t id, const size_t num_blob)
    {
      busy(id,true);
      if(_thread_v[id].joinable()) _thread_v[id].join();
      std::thread t(&circular_buffer<Dtype>::fill_buffer, this, id, num_blob);
      _thread_v[id] = std::move(t);
      usleep(500);
      return true;
    }
    
    template <class Dtype>
    bool circular_buffer<Dtype>::busy(const size_t id, const bool raise) const
    {
      check_id(id);
      bool state = (_state_v[id] == kClean || _state_v[id] == kInit);
      
      if(state && raise) {
	std::cerr << "Buffer ID " << id << " is in busy state..." << std::endl;
	throw std::exception();
      }
      if(!state && _lock_v[id]) {
	state = true;
	if(raise) {
	  std::cerr << "Buffer ID " << id << " is in locked state..." << std::endl;
	  throw std::exception();
	}
      }
      
      return state;
    }
    
    template <class Dtype>
    std::vector<std::shared_ptr<caffe::Blob<Dtype> > >& circular_buffer<Dtype>::get(const size_t id)
    {
      check_id(id);
      if(busy(id)) {
	std::cerr << "Buffer " << id << " is busy!" << std::endl;
	throw std::exception();
      }
      return _buffer_v[id];
    }
    
    template <class Dtype>
    void circular_buffer<Dtype>::check_id(const size_t id) const
    { if( id < _state_v.size() ) return;
      std::cerr << "Invalid buffer id requested!" << std::endl;
      throw std::exception();
    }
    
    template <class Dtype>
    void circular_buffer<Dtype>::fill_buffer(const size_t id, const size_t num_blob)
    {
      if(_state_v[id] != kEmpty) {
	std::cerr << "Buffer id " << id << " @ state " << _state_v[id] << " (not kEmpty)!" << std::endl;
	throw std::exception();
      }
      _state_v  [id] = kInit;
      auto& vec = _buffer_v[id];
      vec.resize(num_blob);
      for(size_t i=0; i<num_blob; ++i)
	vec[i] = std::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>);
      _state_v  [id] = kReady;
    }
    
    template <class Dtype>
    void circular_buffer<Dtype>::clean_buffer(const size_t id)
    {
      if(_state_v[id] != kReady) {
	std::cerr << "Buffer id " << id << " is not in kReady state!" << std::endl;
	throw std::exception();
      }
      _state_v[id]  = kClean;
      _buffer_v[id].clear();
      _state_v[id]  = kEmpty;
    }
  }
}

#include "caffe/util/heproot_circular_buffer.imp.hpp"

#endif

#ifndef HEPROOT_TIMER_CPP
#define HEPROOT_TIMER_CPP

#include "caffe/util/heproot_timer.hpp"

namespace caffe {
  namespace heproot {

    void timer::start() 
    {
      struct timeval current_time;
      gettimeofday(&current_time,NULL);
      _wall_time_start = (double)current_time.tv_sec + (double)current_time.tv_usec * 1.e-6;
      // Get current cpu time
      _cpu_time_start = (double)(clock());
    }

    double timer::wall_time() const
    {
      // Get current wall time
      struct timeval current_time;
      gettimeofday(&current_time,NULL);
      double now = (double)current_time.tv_sec + (double)current_time.tv_usec * 1.e-6;
      // Return diff
      return (now - _wall_time_start);
    }

    double timer::cpu_time() const
    {
      // Get cpu time
      double now = (double)(clock());
      // Return diff
      return (now - _cpu_time_start)/CLOCKS_PER_SEC;
    }
  }
}
#endif

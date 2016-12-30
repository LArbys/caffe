#ifndef HEPROOT_TIMER_HPP
#define HEPROOT_TIMER_HPP

#include <sys/time.h>
#include <time.h>

namespace caffe {
  namespace heproot {

    class timer {
    public:
      timer() {}
      ~timer() {}

      void start(); 

      double wall_time() const;

      double cpu_time() const;
    private:
      double _cpu_time_start;
      double _wall_time_start;
    };
  }
}
#endif

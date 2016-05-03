
#include "caffe/util/larcv_iom_singleton.hpp"

namespace larcv {

  std::map<std::string,larcv::ThreadDatumFiller*> ThreadFillerFactory::_filler_m;

}




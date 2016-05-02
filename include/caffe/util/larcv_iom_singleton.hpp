#include "APICaffe/ThreadDatumFiller.h"

namespace larcv {
  class ThreadFillerFactory {
    
  private:
    ThreadFillerFactory() {}

  public:
    ~ThreadFillerFactory(){ _filler_v.clear(); }
    
    ThreadDatumFiller& get_filler(size_t id) {
      if(id >= _filler_v.size()) 
        throw larbys("Invalid filler id requested!");

      return _filler_v[id];
    }

    size_t create_filler() {
      size_t id = _filler_v.size();
      _filler_v.push_back(::larcv::ThreadDatumFiller());
      return id;
    }

    static ThreadFillerFactory& get() {
      if(!_me) _me = new ThreadFillerFactory;
      return *_me;
    }

  private:

    static ThreadFillerFactory* _me;
    std::vector<larcv::ThreadDatumFiller> _filler_v;

  };
}



#include "APICaffe/ThreadDatumFiller.h"

namespace larcv {
  class SingleIOManager {
    
  private:
    SingleIOManager(size_t id) : data_holder(), _id(id)
    {}

  public:
    ~SingleIOManager(){ data_holder.reset(); }
    
    static SingleIOManager& get(size_t id) {
      return (*(_siom_v.at(id)));
    }

    static SingleIOManager& get() {
      _siom_v.push_back(new SingleIOManager(_siom_v.size()));
      return (*(_siom_v.back()));
    }

    size_t id() const { return _id; }
    
    ThreadDatumFiller data_holder;

  private:
    size_t _id; 

  private:
    static std::vector<larcv::SingleIOManager*> _siom_v;

  };
}



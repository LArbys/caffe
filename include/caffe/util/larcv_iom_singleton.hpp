#include "DataFormat/IOManager.h"

namespace larcv {
  class SingleIOManager {
    
  private:
    SingleIOManager(std::string fname) : manager(larcv::IOManager::kREAD,"IOData")
    { manager.add_in_file(fname); manager.initialize(); }
  public:
    ~SingleIOManager(){ manager.finalize(); }
    
    static SingleIOManager& get(const std::string& fname) { 
      if(!_me) _me = new SingleIOManager(fname);
      return *_me;
    }

    IOManager manager;

  private:
    static SingleIOManager* _me;

  };
}



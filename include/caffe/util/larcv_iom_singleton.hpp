#include "DataFormat/IOManager.h"

namespace larcv {
  class SingleIOManager {
    
  private:
    SingleIOManager(std::string fname) : manager(larcv::IOManager::kREAD,"IOData")
    { manager.add_in_file(fname); manager.initialize(); }
  public:
    ~SingleIOManager(){ manager.finalize(); }
    
    static SingleIOManager& get(const std::string& fname) { 
      auto iter = _siom_m.find(fname);
      if(iter != _siom_m.end()) return (*((*iter).second));
      _siom_m[fname] = new SingleIOManager(fname);
      return (*(_siom_m[fname]));
    }
    
    IOManager manager;

  private:
    static std::map<std::string,larcv::SingleIOManager*> _siom_m;

  };
}



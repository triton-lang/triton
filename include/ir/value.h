#ifndef TDL_INCLUDE_IR_VALUE_H
#define TDL_INCLUDE_IR_VALUE_H

#include <string>

namespace tdl{
namespace ir{

class type;

/* Value */
class value {
public:
  void set_name(const std::string &name);
  type* get_type();

private:
  std::string name_;
};

}
}

#endif

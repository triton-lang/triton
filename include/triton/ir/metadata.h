#pragma once

#ifndef _TRITON_IR_METADATA_H_
#define _TRITON_IR_METADATA_H_

#include <vector>

namespace triton{
namespace ir{


/* Metadata */
class metadata{
public:
  enum kind_t{
    multiple_of,
    max_contiguous
  };

private:
  metadata(kind_t kind, std::vector<unsigned> value);

public:
  static metadata* get(kind_t kind, std::vector<unsigned> value);

private:
  kind_t kind_;
  std::vector<unsigned> value_;
};

}
}

#endif

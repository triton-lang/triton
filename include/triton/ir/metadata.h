#pragma once

#ifndef _TRITON_IR_METADATA_H_
#define _TRITON_IR_METADATA_H_

namespace triton{
namespace ir{


/* Metadata */
class metadata{
public:
  enum kind_t{
    multiple_of
  };

private:
  metadata(kind_t kind, unsigned value);

public:
  static metadata* get(kind_t kind, unsigned value);

private:
  kind_t kind_;
  unsigned value_;
};

}
}

#endif

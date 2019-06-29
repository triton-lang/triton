#ifndef TDL_INCLUDE_IR_METADATA_H
#define TDL_INCLUDE_IR_METADATA_H

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

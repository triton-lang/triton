#ifndef _TRITON_CODEGEN_ANALYSIS_GRID_H_
#define _TRITON_CODEGEN_ANALYSIS_GRID_H_

#include <map>
#include <set>
#include <vector>
#include <memory>
#include "triton/tools/graph.h"

namespace triton{

namespace ir{
  class value;
  class type;
  class module;
  class instruction;
  class phi_node;
}

namespace codegen{
namespace analysis{

class axes;
class align;

enum layout_type_t {
  HMMA_884,
  SCANLINE,
  SHARED
};

struct double_buffer_info_t {
  ir::value* first;
  ir::value* latch;
  ir::phi_node* phi;
};

class layout_visitor;
class layout_t;
class layout_hmma_884_t;
class layout_scanline_t;
class layout_shared_t;


class layout_visitor {
public:
  virtual void visit_layout(layout_t *);
  virtual void visit_layout_hmma_884(layout_hmma_884_t*) = 0;
  virtual void visit_layout_scanline(layout_scanline_t*) = 0;
  virtual void visit_layout_shared(layout_shared_t*) = 0;
};

struct layout_t {
  layout_t(layout_type_t _type,
           const std::vector<int>& _axes,
           const std::vector<unsigned> &_shapes,
           const std::vector<ir::value *> &_values,
           ir::type *_ty,
           size_t _id,
           analysis::align* align);

  virtual void accept(layout_visitor* vst) = 0;

  layout_type_t type;
  std::vector<int> axes;
  std::vector<unsigned> shapes;
  std::vector<ir::value*> values;
  std::vector<int> order;
  size_t id;
  size_t size;
  std::shared_ptr<double_buffer_info_t> double_buffer;
  ir::type *ty;
  size_t pad;
  std::vector<int> mts;
  std::vector<int> nts;
  std::vector<int> fpw;
  std::vector<int> wpt;
};

struct layout_hmma_884_t: public layout_t {
  layout_hmma_884_t(size_t num_warps,
                    const std::vector<int>& _axes,
                    const std::vector<unsigned>& _shapes,
                    const std::vector<ir::value *> &_values,
                    ir::type *_ty,
                    size_t _id,
                    analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_hmma_884(this); }
};

struct layout_scanline_t: public layout_t {
  layout_scanline_t(size_t num_warps,
                    const std::vector<int>& _axes,
                    const std::vector<unsigned>& _shapes,
                    const std::vector<ir::value *> &values,
                    ir::type *_ty,
                    size_t _id,
                    analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_scanline(this); }
};

struct layout_shared_t: public layout_t {
  layout_shared_t(const layout_t *arg,
                    const std::vector<int>& _axes,
                    const std::vector<unsigned>& _shapes,
                    const std::vector<ir::value *> &values,
                    ir::type *ty,
                    size_t _id,
                    analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_shared(this); }
};



class layout {
  typedef ir::value* node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  // graph creation
  void connect(ir::value *x, ir::value *y);
  void make_graph(ir::instruction *i);

  void init_hmma_tile(layout_t& layout);
  void init_scanline_tile(layout_t &layout);

  void create(size_t id, const std::vector<ir::value*>& values);

//  size_t shared_tmp_req(ir::instruction* i) {
//    switch(i->get_id()) {
//      case ir::INST_REDUCE: {
//        ir::reduce_inst *red = (ir::reduce_inst*)i;
//        ir::type *ty = red->get_type();


//      }
//      default: return 0;
//    }
//  }

public:
  // constructor
  layout(analysis::axes *axes, analysis::align *align, size_t num_warps);

  // accessors
  unsigned layout_of(ir::value *value) const;
  const std::vector<ir::value*>& values_of(unsigned id) const;
  size_t num_layouts() const;
  const layout_t* get(size_t id) const;
  const layout_t* get(ir::value *v) const;
  std::map<size_t, layout_t*> &get_all();
  size_t tmp(ir::instruction* i);

  // execution
  void run(ir::module &mod);

private:
  analysis::axes* axes_;
  analysis::align* align_;
  size_t num_warps_;
  tools::graph<ir::value*> graph_;
  std::map<ir::value*, size_t> groups_;
  std::map<size_t, std::vector<ir::value*>> values_;
  std::map<size_t, layout_t*> layouts_;
  std::map<ir::value*, size_t> tmp_;
};

}
}

}

#endif

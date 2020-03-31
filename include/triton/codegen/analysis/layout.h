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
class layout_visitor;
class data_layout;
class mma884_layout;
class scanline_layout;
class shared_layout;


class layout_visitor {
public:
  virtual void visit_layout(data_layout *);
  virtual void visit_layout_hmma_884(mma884_layout*) = 0;
  virtual void visit_layout_scanline(scanline_layout*) = 0;
  virtual void visit_layout_shared(shared_layout*) = 0;
};

class data_layout {
protected:
  enum id_t {
    HMMA_884,
    SCANLINE,
    SHARED
  };

  typedef std::vector<int> axes_t;
  typedef std::vector<unsigned> shape_t;
  typedef std::vector<int> order_t;
  typedef std::vector<ir::value*> values_t;

private:
  template<typename T>
  T* downcast(id_t id) {
    if(id_ == id)
      return static_cast<T*>(this);
    return nullptr;
  }

public:
  data_layout(id_t id,
             const std::vector<int>& axes,
             const std::vector<unsigned> &shape,
             const std::vector<ir::value *> &values,
             analysis::align* align);
  // visitor
  virtual void accept(layout_visitor* vst) = 0;
  // downcast
  mma884_layout* to_mma884()          { return downcast<mma884_layout>(HMMA_884); }
  scanline_layout* to_scanline()      { return downcast<scanline_layout>(SCANLINE); }
  shared_layout* to_shared()          { return downcast<shared_layout>(SHARED); }
  // accessors
  size_t get_rank()                   { return shape_.size(); }
  const shape_t& get_shape() const    { return shape_; }
  const order_t& get_order() const    { return order_; }
  const values_t& get_values() const  { return values_;}
  int get_axis(size_t k) const        { return axes_.at(k); }
  const int get_order(size_t k) const { return order_.at(k); }
  // find the position of given axis
  size_t find_axis(int to_find) const;


private:
  id_t id_;
  axes_t axes_;
  values_t values_;

protected:
  order_t order_;
  shape_t shape_;
};

class mma884_layout: public data_layout {
public:
  mma884_layout(size_t num_warps,
                const std::vector<int>& axes,
                const std::vector<unsigned>& shapes,
                const std::vector<ir::value *> &values,
                analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_hmma_884(this); }
  // accessor
  int fpw(size_t k) { return fpw_.at(k); }
  int wpt(size_t k) { return wpt_.at(k); }

private:
  std::vector<int> fpw_;
  std::vector<int> wpt_;
};

struct scanline_layout: public data_layout {
  scanline_layout(size_t num_warps,
                    const std::vector<int>& axes,
                    const std::vector<unsigned>& shape,
                    const std::vector<ir::value *> &values,
                    analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_scanline(this); }
  // accessor
  int mts(size_t k) { return mts_.at(k); }
  int nts(size_t k) { return nts_.at(k); }

public:
  std::vector<int> mts_;
  std::vector<int> nts_;
};

struct double_buffer_info_t {
  ir::value* first;
  ir::value* latch;
  ir::phi_node* phi;
};

class shared_layout: public data_layout {
private:
  static bool is_loop_latch(ir::phi_node *phi, ir::instruction *terminator);
  static void extract_double_bufferable(ir::value *v, std::shared_ptr<double_buffer_info_t>& res);

public:
  shared_layout(const data_layout *arg,
                const std::vector<int>& axes,
                const std::vector<unsigned>& shapes,
                const std::vector<ir::value *> &values_,
                ir::type *ty,
                analysis::align* align);
  void accept(layout_visitor* vst) { vst->visit_layout_shared(this); }
  // accessors
  size_t get_size()                         { return size_; }
  ir::type* get_type()                      { return ty_; }
  double_buffer_info_t* get_double_buffer() { return double_buffer_.get(); }

private:
  size_t size_;
  ir::type *ty_;
  std::shared_ptr<double_buffer_info_t> double_buffer_;
};



class layouts {
  typedef ir::value* node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  // graph creation
  void connect(ir::value *x, ir::value *y);
  void make_graph(ir::instruction *i);

  void init_hmma_tile(data_layout& layouts);
  void init_scanline_tile(data_layout &layouts);

  void create(size_t id, const std::vector<ir::value*>& values);

public:
  // constructor
  layouts(analysis::axes *axes, analysis::align *align, size_t num_warps);

  // accessors
  unsigned layout_of(ir::value *value) const                  { return groups_.at(value); }
  const std::vector<ir::value*>& values_of(unsigned id) const { return values_.at(id); }
  size_t num_layouts() const                                  { return values_.size();}
  data_layout* get(size_t id)                                 { return layouts_.at(id); }
  data_layout* get(ir::value *v)                              { return get(layout_of(v));}
  std::map<size_t, data_layout*> &get_all()                   { return layouts_; }
  size_t tmp(ir::instruction* i)                              { return tmp_.at((ir::value*)i);}

  // execution
  void run(ir::module &mod);

private:
  analysis::axes* axes_;
  analysis::align* align_;
  size_t num_warps_;
  tools::graph<ir::value*> graph_;
  std::map<ir::value*, size_t> groups_;
  std::map<size_t, std::vector<ir::value*>> values_;
  std::map<size_t, data_layout*> layouts_;
  std::map<ir::value*, size_t> tmp_;
};

}
}

}

#endif

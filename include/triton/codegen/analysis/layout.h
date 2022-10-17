#ifndef _TRITON_CODEGEN_ANALYSIS_GRID_H_
#define _TRITON_CODEGEN_ANALYSIS_GRID_H_

#include <map>
#include <set>
#include <vector>
#include <memory>
#include "triton/tools/graph.h"
#include "triton/codegen/target.h"

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
class mma_layout;
class scanline_layout;
class shared_layout;


class layout_visitor {
public:
  virtual void visit_layout(data_layout *);
  virtual void visit_layout_mma(mma_layout*) = 0;
  virtual void visit_layout_scanline(scanline_layout*) = 0;
  virtual void visit_layout_shared(shared_layout*) = 0;
};

class data_layout {
protected:
  enum id_t {
    MMA,
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
  mma_layout* to_mma()          { return downcast<mma_layout>(MMA); }
  scanline_layout* to_scanline()      { return downcast<scanline_layout>(SCANLINE); }
  shared_layout* to_shared()          { return downcast<shared_layout>(SHARED); }
  // accessors
  size_t get_rank()                   { return shape_.size(); }
  const shape_t& get_shape() const    { return shape_; }
  const order_t& get_order() const    { return order_; }
  const values_t& get_values() const  { return values_;}
  int get_axis(size_t k) const        { return axes_.at(k); }
  std::vector<int> get_axes() const		{ return axes_; }
  const int get_order(size_t k) const { return order_.at(k); }
  // find the position of given axis
  int find_axis(int to_find) const;


private:
  id_t id_;
  axes_t axes_;
  values_t values_;

protected:
  order_t order_;
  shape_t shape_;
};

class distributed_layout: public data_layout{
public:
  distributed_layout(id_t id,
                     const std::vector<int>& axes,
                     const std::vector<unsigned>& shape,
                     const std::vector<ir::value*>& values,
                     analysis::align* align);

  int shape_per_cta(size_t k) { return shape_per_cta_.at(k); }
  int rep_per_cta(size_t k) { return shape_[k] / shape_per_cta_[k]; }
  virtual int contig_per_thread(size_t k) = 0;

protected:
  std::vector<int> shape_per_cta_;
};

class mma_layout: public distributed_layout {
public:
  enum TensorCoreType : uint8_t {
    // floating-point tensor core instr
    FP32_FP16_FP16_FP32 = 0, // default
    FP32_BF16_BF16_FP32,
    FP32_TF32_TF32_FP32,
    // integer tensor core instr
    INT32_INT1_INT1_INT32, // Not implemented
    INT32_INT4_INT4_INT32, // Not implemented
    INT32_INT8_INT8_INT32, // Not implemented
    //
    NOT_APPLICABLE,    
  };

  // Used on nvidia GPUs with sm >= 80
  inline static const std::map<TensorCoreType, std::vector<int>> mma_instr_shape_ = {
    {FP32_FP16_FP16_FP32, {16, 8, 16}}, 
    {FP32_BF16_BF16_FP32, {16, 8, 16}},
    {FP32_TF32_TF32_FP32, {16, 8, 8}},

    {INT32_INT1_INT1_INT32, {16, 8, 256}},
    {INT32_INT4_INT4_INT32, {16, 8, 64}},
    {INT32_INT8_INT8_INT32, {16, 8, 32}},
  };

  // shape of matrices loaded by ldmatrix (m-n-k, for mxk & kxn matrices)
  inline static const std::map<TensorCoreType, std::vector<int>> mma_mat_shape_ = {
    {FP32_FP16_FP16_FP32, {8, 8, 8}}, 
    {FP32_BF16_BF16_FP32, {8, 8, 8}},
    {FP32_TF32_TF32_FP32, {8, 8, 4}},

    {INT32_INT1_INT1_INT32, {8, 8, 64}},
    {INT32_INT4_INT4_INT32, {8, 8, 32}},
    {INT32_INT8_INT8_INT32, {8, 8, 16}},
  };

  inline static const std::map<TensorCoreType, std::string> mma_instr_ptx_ = {
    {FP32_FP16_FP16_FP32, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"}, 
    {FP32_BF16_BF16_FP32, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
    {FP32_TF32_TF32_FP32, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},

    {INT32_INT1_INT1_INT32, "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
    {INT32_INT4_INT4_INT32, "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
    {INT32_INT8_INT8_INT32, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},
  };

  // vector length per ldmatrix (16*8/elelment_size_in_bits)
  inline static const std::map<TensorCoreType, int> mma_instr_vec_ = {
    {FP32_FP16_FP16_FP32, 8},
    {FP32_BF16_BF16_FP32, 8},
    {FP32_TF32_TF32_FP32, 4},

    {INT32_INT1_INT1_INT32, 128},
    {INT32_INT4_INT4_INT32, 32},
    {INT32_INT8_INT8_INT32, 16},
  };

public:
  mma_layout(size_t num_warps,
                const std::vector<int>& axes,
                const std::vector<unsigned>& shapes,
                const std::vector<ir::value *> &values,
                analysis::align* align, target *tgt,
             shared_layout* layout_a,
             shared_layout* layout_b,
             ir::value *dot);
  void accept(layout_visitor* vst) { vst->visit_layout_mma(this); }
  // accessor
  int fpw(size_t k) { return fpw_.at(k); }
  int wpt(size_t k) { return wpt_.at(k); }
  int spw(size_t k) { return spw_.at(k); }
  int rep(size_t k) { return rep_.at(k); }
  int contig_per_thread(size_t k) { return contig_per_thread_.at(k); }

  // helpers for generator.cc
  std::string get_ptx_instr() const { return mma_instr_ptx_.at(tensor_core_type_); }
  std::vector<int> get_mma_instr_shape() const { return mma_instr_shape_.at(tensor_core_type_); }
  std::vector<int> get_mma_mat_shape() const { return mma_mat_shape_.at(tensor_core_type_); }
  int get_vec_a() const { return mma_instr_vec_.at(tensor_core_type_); }
  int get_vec_b() const { return mma_instr_vec_.at(tensor_core_type_); }

  // setter
  void set_tensor_core_type(TensorCoreType type) { tensor_core_type_ = type; }

private:
  // fragment per warp
  std::vector<int> fpw_;
  // shape per warp
  std::vector<int> spw_;
  // warp per tile
  std::vector<int> wpt_;
  // shape per tile
  std::vector<int> spt_;
  // repetitions
  std::vector<int> rep_;
  // contiguous per thread
  std::vector<int> contig_per_thread_;

  TensorCoreType tensor_core_type_ = FP32_FP16_FP16_FP32;
};

class scanline_layout: public distributed_layout {
public:
  scanline_layout(size_t num_warps,
                    const std::vector<int>& axes,
                    const std::vector<unsigned>& shape,
                    const std::vector<ir::value *> &values,
                    analysis::align* align,
                    target* tgt);
  void accept(layout_visitor* vst) { vst->visit_layout_scanline(this); }
  // accessor
  int mts(size_t k) { return mts_.at(k); }
  int nts(size_t k) { return nts_.at(k); }
  int contig_per_thread(size_t k) { return nts_.at(k); }

  int per_thread(size_t k) { return contig_per_thread(k) * shape_[k] / shape_per_cta(k);}
private:
  // micro tile size. The size of a tile held by a thread block.
  std::vector<int> mts_;
  // nano tile size. The size of a tile held by a thread.
  std::vector<int> nts_;
};

struct double_buffer_info_t {
  ir::value* first;
  ir::value* latch;
  ir::phi_node* phi;
};

struct N_buffer_info_t {
  std::vector<ir::value*> firsts; // not necessarily ordered as input order
  ir::value* latch;
  ir::phi_node* phi;
  std::map<ir::value*, int> firsts_idx;
};

// abstract for dot and corresponding smem values
class shared_layout: public data_layout {
private:
  static bool is_loop_latch(ir::phi_node *phi, ir::instruction *terminator);
  static void extract_double_bufferable(ir::value *v, std::shared_ptr<double_buffer_info_t>& res);
  static void extract_N_bufferable(ir::value *v, std::shared_ptr<N_buffer_info_t>& res, int &prev_stages);

public:
  shared_layout(data_layout *arg,
                const std::vector<int>& axes,
                const std::vector<unsigned>& shapes,
                const std::vector<ir::value *> &values_,
                ir::type *ty,
                analysis::align* align, target *tgt,
                bool is_tmp = false);
  void accept(layout_visitor* vst) { vst->visit_layout_shared(this); }
  // accessors
  size_t get_size()                         { return size_; }
  ir::type* get_type()                      { return ty_; }
  double_buffer_info_t* get_double_buffer() { return double_buffer_.get(); }
  N_buffer_info_t* get_N_buffer()           { return N_buffer_.get(); }
  int get_num_stages() const;
  size_t get_per_stage_size() const         { return size_ / get_num_stages(); }
  size_t get_per_stage_elements() const;
  size_t get_num_per_phase()                { return num_per_phase_; }
  ir::value* hmma_dot_a()                      { return hmma_dot_a_; }
  ir::value* hmma_dot_b()                      { return hmma_dot_b_; }
  void set_mma_vec(int mma_vec)             { mma_vec_ = mma_vec; }
  int  get_mma_vec()                        { return mma_vec_;}
  int  get_mma_strided()                    { return mma_strided_; }
  bool allow_swizzle() const                { return allow_swizzle_; }
  data_layout* get_arg_layout()             { return arg_layout_; }
  bool is_tmp() const                       { return is_tmp_; }

private:
  size_t size_;
  ir::type *ty_;
  std::shared_ptr<double_buffer_info_t> double_buffer_;
  std::shared_ptr<N_buffer_info_t>      N_buffer_;
  size_t num_per_phase_;
  ir::value* hmma_dot_a_;
  ir::value* hmma_dot_b_;
  data_layout* arg_layout_;
  int mma_vec_;
  int mma_strided_;
  bool allow_swizzle_ = true;
  target *tgt_;
  bool is_tmp_;
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

  void create_tmp_layout(size_t id, data_layout* arg,
                         const std::vector<int>& axes,
                         const std::vector<unsigned>& shape,
                         ir::instruction* i,
                         bool is_index = false);

 public:
  // constructor
  layouts(analysis::axes *axes, analysis::align *align, size_t num_warps, target* tgt);

  // accessors
  unsigned layout_of(ir::value *value) const                  { return groups_.at(value); }
  bool has(ir::value* value) const { return groups_.find(value) != groups_.end(); }
  bool has(size_t id)                                         { return layouts_.find(id) != layouts_.end(); }
  const std::vector<ir::value*>& values_of(unsigned id) const { return values_.at(id); }
  size_t num_layouts() const                                  { return values_.size();}
  data_layout* get(size_t id)                                 { return layouts_.at(id); }
  data_layout* get(ir::value *v)                              { return get(layout_of(v));}
  std::map<size_t, data_layout*> &get_all()                   { return layouts_; }
  bool has_tmp(ir::value* i)                                  { return tmp_.find(i) != tmp_.end(); }
  int tmp(ir::value* i)                                       { return tmp_.at(i);}
  int has_tmp_index(ir::value* i)                             { return tmp_index_.find(i) != tmp_index_.end(); }
  int tmp_index(ir::value* i)                                 { return tmp_index_.at(i);}
  void copy(ir::value* dst, ir::value* src)                   { groups_[dst] = groups_[src]; }

  // layout checkers
  bool is_scanline(ir::instruction* i);

  bool is_coalesced_scanline(ir::instruction* i);

  bool is_mma(ir::instruction* i);

  bool is_a100_mma(ir::instruction* i);

  // execution
  void run(ir::module &mod);

private:
  analysis::axes* axes_;
  analysis::align* align_;
  size_t num_warps_;
  target* tgt_;
  tools::graph<ir::value*> graph_;
  std::map<ir::value*, size_t> groups_;
  std::map<size_t, std::vector<ir::value*>> values_;
  std::map<size_t, data_layout*> layouts_;
  std::map<ir::value*, size_t> tmp_;
  std::map<ir::value*, size_t> tmp_index_;
};

}
}

}

#endif

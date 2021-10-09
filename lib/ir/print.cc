#include <iostream>
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/ir/type.h"
#include "triton/ir/value.h"
#include "triton/ir/constant.h"
#include "triton/ir/function.h"
#include "triton/ir/instructions.h"
#include "triton/ir/print.h"

#include <map>
#include <iomanip>

namespace triton{
namespace ir{

namespace {
class SlotTracker {
  // A mapping of values to slot numbers.
  using value_map = std::map<const value*, unsigned>;

  // The module for which we are holding slot numbers.
  const module *mod_;
  bool module_processed = false;

  // The function for which we are holding slot numbers.
  const function *func_ = nullptr;
  bool function_processed = false;

  // m_map - The slot map for the module level data.
  value_map m_map;
  unsigned m_next = 0;

  // f_map - The slot map for the function level data.
  value_map f_map;
  unsigned f_next = 0;

public:
  // Construct from a module
  explicit SlotTracker(const module *mod) : mod_(mod) {}

  // Construct from a function
  explicit SlotTracker(const function *f) 
    : mod_(f? f->get_parent() : nullptr), func_(f) {}

  // Return the slot number of the specified value. If something is not in
  // the SlotTracker, return -1
  int get_local_slot(const value *v);

  void initialize_if_needed();

  // If you'd like to deal with a function instead of just a module, use
  // this method to get its data into the SlotTracker
  void incorporate_function(const function *f) {
    func_ = f;
    function_processed = false;
  }

private:
  // Add all of the module level global variables (and their initializers)
  // and function declarations, but not contents of those functions.
  void process_module();

  // Add all of the functions arguments, basic blocks, and instructions.
  void process_function();

  // Insert specified value* into the slot table
  void create_function_slot(const value *v);
};

class AssemblyWriter {
  std::ostream &os;
  SlotTracker &slot_tracker;

public:
  AssemblyWriter(std::ostream &os, SlotTracker &slot_tracker)
    : os(os), slot_tracker(slot_tracker) {}

  void print_module(const module *mod);
  void print_function(const function *f);
  void print_argument(const argument *arg);
  void print_basic_block(const basic_block *bb);
  void print_instruction(const instruction *instr);
  void print_value(const value *v);

  void write_operand(const value *op, bool print_type = false);
};
} // anonymous namespace

//-------------------------
// SlotTracker
//-------------------------
void SlotTracker::process_module() {
  // Nothing to do at the moment.
  // Create slots for global variable & unamed functions & ...
  module_processed = true;
}

void SlotTracker::process_function() {
  f_next = 0;

  // Add all the function arguments with no names.
  for (const argument *arg : func_->args())
    if (!arg->has_name())
      create_function_slot(arg);

  // Add all of the basic blocks and instructions with no names.
  for (const basic_block *bb : func_->blocks()) {
    if (!bb->has_name())
      create_function_slot(bb);

    for (const instruction *instr : bb->get_inst_list()) {
      if (!instr->get_type()->is_void_ty() && !instr->has_name())
        create_function_slot(instr);
    }
  }

  function_processed = true;
}

void SlotTracker::create_function_slot(const value *v) {
  assert(!v->get_type()->is_void_ty() && !v->has_name() && "Doesn't need a slot");

  unsigned dst_slot = f_next++;
  f_map[v] = dst_slot;
}

int SlotTracker::get_local_slot(const value *v) {
  assert(dynamic_cast<const constant*>(v) == nullptr && "Can't get a constant slot");

  // Check for uninitialized state and do lazy initialization.
  initialize_if_needed();

  value_map::iterator f_iter = f_map.find(v);
  return f_iter == f_map.end() ? -1 : (int)f_iter->second;
}

void SlotTracker::initialize_if_needed() {
  if (mod_ && !module_processed)
    process_module();

  if (func_ && !function_processed)
    process_function();
}


//-------------------------------
// AssemblyWriter
//-------------------------------
void AssemblyWriter::write_operand(const value *operand, bool print_type) {
  if (!operand) {
    os << "<null operand!>";
    return;
  }

  if (auto *c = dynamic_cast<const ir::constant*>(operand)) {
    os << c->repr();
    return;
  }

  if (operand->has_name()) {
    os << operand->get_name();
    return;
  }

  // Print the normal way
  int slot_num = slot_tracker.get_local_slot(operand);

  if (slot_num != -1)
    os << "%" << slot_num;
  else
    os << "<badref>";
}

void AssemblyWriter::print_module(const module *mod) {
  slot_tracker.initialize_if_needed();
  // ;ModuleID = ...
  // source_filename = ...

  // Print all of the functions.
  for (function *f : mod->get_function_list()) {
    os << "\n";
    print_function(f);
  }
}

void AssemblyWriter::print_function(const function *f) {
  // Annotation & Attributes

  slot_tracker.incorporate_function(f);

  os << "def ";
  ir::type *rt_type = f->get_fn_type()->get_return_ty();
  // Functions must have names.
  os << rt_type->repr() << " " << f->get_name() << "(";
  // Print arguments
  for (ir::argument *arg : f->args()) {
    if (arg->get_arg_no() > 0)
      os << ", ";
    print_argument(arg);
  }
  os << ")";

  // Print function body
  os << "{";
  for (const basic_block *bb : f->blocks()) 
    print_basic_block(bb);
  os << "}\n";
}

void AssemblyWriter::print_argument(const argument *arg) {
  // Print type
  os << arg->get_type()->repr();

  // Print name, if available.
  if (arg->has_name())
    os << " " << arg->get_name();
  else {
    int slot_num = slot_tracker.get_local_slot(arg);
    assert(slot_num != -1 && "expect argument in function here");
    os << " %" << slot_num;
  }

  // Print attributes
  std::set<attribute> attrs = arg->get_parent()->get_attributes(arg);
  for (attribute attr : attrs)
    os << " " << attr.repr();
}

void AssemblyWriter::print_basic_block(const basic_block *bb) {
  // bb label
  if (bb->has_name()) {
    os << "\n";
    os << bb->get_name() << ":";
  } else {
    os << "\n";
    int slot_num = slot_tracker.get_local_slot(bb);
    if (slot_num != -1)
      os << slot_num << ":";
    else
      os << "<badref>:";
  }

  // Print predecessors for the block
  auto const &predecessors = bb->get_predecessors();
  if (!predecessors.empty()) {
    os << std::setw(50) << std::setfill(' ')
       << "; preds = ";
    for (size_t i=0; i<predecessors.size(); ++i) {
      if (i)
        os << ", ";
      write_operand(predecessors[i]);
    }
  }

  os << "\n";

  // Annotation?

  // Print all of the instructions in the basic block
  for (const ir::instruction *instr : bb->get_inst_list())
    print_instruction(instr);
}

void AssemblyWriter::print_instruction(const instruction *instr) {
  // Print out indentation for an instruction.
  os << "  ";

  ir::type *type = instr->get_type();
  if (instr->has_name()) {
    os << instr->get_name();
    os << " = ";
  } else if (!type->is_void_ty()) {
    // Print out the def slot taken.
    int slot_num = slot_tracker.get_local_slot(instr);
    if (slot_num == -1)
      os << "<badref> = ";
    else
      os << "%" << slot_num << " = ";
  }

  // Print out opcode
  os << instr->repr() << " " << type->repr();

  size_t num_ops = instr->get_num_operands();
  if (num_ops > 0)
    os << " ";
  ir::instruction::ops_t ops = instr->ops();
  for (unsigned i = 0; i < num_ops; ++i) {
    if (i)
      os << ", ";
    write_operand(ops[i]);
  }

  os << ";\n";
}

void AssemblyWriter::print_value(const value *v) {
  // Not implemented
}


//-------------------------------
// External interface
//-------------------------------
void module::print(std::ostream &os) {
  SlotTracker slot_tracker(this);
  AssemblyWriter writer(os, slot_tracker);
  writer.print_module(this);
}

void function::print(std::ostream &os) {
  SlotTracker slot_tracker(this);
  AssemblyWriter writer(os, slot_tracker);
  writer.print_function(this);
}

void basic_block::print(std::ostream &os) {
  SlotTracker slot_tracker(this->get_parent());
  AssemblyWriter writer(os, slot_tracker);
  writer.print_basic_block(this);
}

void instruction::print(std::ostream &os) {
  SlotTracker slot_tracker(this->get_parent()->get_parent());
  AssemblyWriter writer(os, slot_tracker);
  writer.print_instruction(this);
}

//-------------------------------
// legacy print interface
//-------------------------------
std::string get_name(ir::value *v, unsigned i) {
  if(v->get_name().empty()){
    std::string name = "%" + std::to_string(i);
    v->set_name(name);
  }
  return v->get_name();
}


void print(module &mod, std::ostream& os) {
  unsigned cnt = 0;
  for(ir::function *fn: mod.get_function_list()){
    os << "def " << fn->get_fn_type()->get_return_ty()->repr() << " " << fn->get_name() << "(" ;
    for(ir::argument* arg: fn->args()) {
      if(arg->get_arg_no() > 0)
        os << ", ";
      os << arg->get_type()->repr() << " " << arg->get_name();
      auto attrs = fn->get_attributes(arg);
      if(attrs.size() > 0)
        os << " ";
      for(ir::attribute attr: attrs)
        os << attr.repr() << " ";
    }
    os << ")" << std::endl;
    os << "{" << std::endl;
    for(ir::basic_block *block: fn->blocks()){
      auto const &predecessors = block->get_predecessors();
      os << block->get_name() << ":";
      if(!predecessors.empty()){
        os << "                 ";
        os << "; preds = ";
        auto const &predecessors = block->get_predecessors();
        for(ir::basic_block *pred: predecessors)
          os << pred->get_name() << (pred!=predecessors.back()?", ":"");
      }
      os << std::endl;
      for(ir::instruction *inst: block->get_inst_list()){
        os << "  ";
        if(!inst->get_type()->is_void_ty()){
          os << get_name(inst, cnt++);
          os << " = ";
        }
        ir::type* type = inst->get_type();
        os << inst->repr() << " " << type->repr();
        ir::instruction::ops_t ops = inst->ops();
        size_t num_ops = inst->get_num_operands();
        if(num_ops > 0)
          os << " ";;
        for(unsigned i = 0; i < num_ops; i++){
          if(auto *x = dynamic_cast<ir::constant*>(ops[i]))
            os << x->repr();
          else
            os << get_name(ops[i], cnt++);
          os << (i < num_ops - 1?", ":"");
        }
        os << ";";
//        os << " (";
//        for(ir::user* usr: inst->get_users())
//          os << get_name(usr, cnt++) << ", " ;
//        os << " )";
        os << std::endl;
      }
    }
    os << "}" << std::endl;
  }
}

void print(function &fn, std::ostream &os) {
  //
}

void print(basic_block &bb, std::ostream &os) {
  auto const &predecessors = bb.get_predecessors();
  os << bb.get_name() << ":";
  if(!predecessors.empty()){
    os << "                 ";
    os << "; preds = ";
    auto const &predecessors = bb.get_predecessors();
    for(ir::basic_block *pred: predecessors)
      os << pred->get_name() << (pred!=predecessors.back()?", ":"");
  }
  os << std::endl;
  for(ir::instruction *inst: bb.get_inst_list()){
    print(*inst, os);
  }
}

void print(instruction &instr, std::ostream &os) {
    instruction *inst = &instr;
    os << "  ";
    if(!inst->get_type()->is_void_ty()){
      os << instr.get_name();
      os << " = ";
    }
    ir::type* type = inst->get_type();
    os << inst->repr() << " " << type->repr();
    ir::instruction::ops_t ops = inst->ops();
    size_t num_ops = inst->get_num_operands();
    if(num_ops > 0)
      os << " ";;
    for(unsigned i = 0; i < num_ops; i++){
      if(auto *x = dynamic_cast<ir::constant*>(ops[i]))
        os << x->repr();
      else
        os << ops[i]->get_name();
      os << (i < num_ops - 1?", ":"");
    }
    os << ";";
//        os << " (";
//        for(ir::user* usr: inst->get_users())
//          os << get_name(usr, cnt++) << ", " ;
//        os << " )";
    os << std::endl;
}


}
}

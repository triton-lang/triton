#ifndef ATIDLAS_MAPPED_OBJECT_H
#define ATIDLAS_MAPPED_OBJECT_H

#include <map>
#include <string>
#include "atidlas/types.h"
#include "atidlas/backend/stream.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

enum leaf_t
{
  LHS_NODE_TYPE,
  PARENT_NODE_TYPE,
  RHS_NODE_TYPE
};

class mapped_object;

typedef std::pair<int_t, leaf_t> mapping_key;
typedef std::map<mapping_key, tools::shared_ptr<mapped_object> > mapping_type;

/** @brief Mapped Object
*
* This object populates the symbolic mapping associated with a symbolic_expression. (root_id, LHS|RHS|PARENT) => mapped_object
* The tree can then be reconstructed in its symbolic form
*/
class mapped_object
{
private:
  virtual void preprocess(std::string &) const;
  virtual void postprocess(std::string &) const;

protected:
  struct MorphBase {
      virtual std::string operator()(std::string const & i) const = 0;
      virtual std::string operator()(std::string const & i, std::string const & j) const = 0;
      virtual ~MorphBase(){}
  };

  static void replace_macro(std::string & str, std::string const &, MorphBase const & morph);
  void register_attribute(std::string & attribute, std::string const & key, std::string const & value);

public:
  struct node_info
  {
    node_info(mapping_type const * _mapping, symbolic_expression const * _symbolic_expression, int_t _root_idx);
    mapping_type const * mapping;
    atidlas::symbolic_expression const * symbolic_expression;
    int_t root_idx;
  };

public:
  mapped_object(std::string const & scalartype, unsigned int id, std::string const & type_key);
  virtual ~mapped_object();

  std::string type_key() const;
  std::string const & name() const;
  std::map<std::string, std::string> const & keywords() const;

  std::string process(std::string const & in) const;
  std::string evaluate(std::map<std::string, std::string> const & accessors) const;
protected:
  std::string name_;
  std::string scalartype_;
  std::string type_key_;
  std::map<std::string, std::string> keywords_;
};

class binary_leaf
{
public:
  binary_leaf(mapped_object::node_info info);

  void process_recursive(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors);
  std::string evaluate_recursive(leaf_t leaf, std::map<std::string, std::string> const & accessors);
protected:
  mapped_object::node_info info_;
};

/** @brief Matrix product
  *
  * Maps prod(matrix_expression, matrix_expression)
  */
class mapped_mproduct : public mapped_object, public binary_leaf
{
public:
  mapped_mproduct(std::string const & scalartype, unsigned int id, node_info info);
};

/** @brief Reduction
*
* Base class for mapping a reduction
*/
class mapped_reduction : public mapped_object, public binary_leaf
{
public:
  mapped_reduction(std::string const & scalartype, unsigned int id, node_info info, std::string const & type_key);

  int_t root_idx() const;
  atidlas::symbolic_expression const & symbolic_expression() const;
  symbolic_expression_node root_node() const;
  bool is_index_reduction() const;
  op_element root_op() const;
};

/** @brief Scalar reduction
*
* Maps a scalar reduction (max, min, argmax, inner_prod, etc..)
*/
class mapped_scalar_reduction : public mapped_reduction
{
public:
  mapped_scalar_reduction(std::string const & scalartype, unsigned int id, node_info info);
};

/** @brief Vector reduction
*
* Maps a row-wise reduction (max, min, argmax, matrix-vector product, etc..)
*/
class mapped_mreduction : public mapped_reduction
{
public:
  mapped_mreduction(std::string const & scalartype, unsigned int id, node_info info);
};

/** @brief Host scalar
 *
 * Maps a host scalar (passed by value)
 */
class mapped_host_scalar : public mapped_object
{
  void preprocess(std::string & str) const;
public:
  mapped_host_scalar(std::string const & scalartype, unsigned int id);
};

/** @brief Tuple
*
* Maps an object passed by pointer
*/
class mapped_tuple : public mapped_object
{
public:
  mapped_tuple(std::string const & scalartype, unsigned int id, size_t size);
private:
  size_t size_;
  std::vector<std::string> names_;
};

/** @brief Handle
*
* Maps an object passed by pointer
*/
class mapped_handle : public mapped_object
{
public:
  mapped_handle(std::string const & scalartype, unsigned int id, std::string const & type_key);
private:
  std::string pointer_;
};

/** @brief Buffered
 *
 * Maps a buffered object (vector, matrix)
 */
class mapped_buffer : public mapped_handle
{
public:
  mapped_buffer(std::string const & scalartype, unsigned int id, std::string const & type_key);
};

class mapped_array : public mapped_buffer
{
private:
  void preprocess(std::string & str) const;
public:
  mapped_array(std::string const & scalartype, unsigned int id, char type);
private:
  std::string ld_;
  std::string start1_;
  std::string start2_;
  std::string stride1_;
  std::string stride2_;
  char type_;
};

class mapped_vdiag : public mapped_object, public binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_vdiag(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_matrix_row : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_matrix_row(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_matrix_column : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_matrix_column(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_repeat : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_repeat(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_matrix_diag : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_matrix_diag(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_outer : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const;
public:
  mapped_outer(std::string const & scalartype, unsigned int id, node_info info);
};

class mapped_cast : public mapped_object
{
  static std::string operator_to_str(operation_node_type type);
public:
  mapped_cast(operation_node_type type, unsigned int id);
};

}
#endif

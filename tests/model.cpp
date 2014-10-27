#include <map>

#include "viennacl/vector.hpp"

#include "atidlas/tools/shared_ptr.hpp"
#include "atidlas/model/import.hpp"

namespace ad = atidlas;

int main()
{
  viennacl::vector<float> x(10000), y(10000), z(10000);
  std::map<std::string, ad::tools::shared_ptr<ad::model> > models = ad::import("geforce_gt_540m.json");
  models["vector-axpy-float32"]->tune(viennacl::scheduler::statement(z, viennacl::op_assign(), x));
  models["vector-axpy-float32"]->execute(viennacl::scheduler::statement(z, viennacl::op_assign(), x));
  return EXIT_SUCCESS;
}

#include <set>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <memory>

#include "rapidjson/document.h"
#include "isaac/backend/parse.h"
#include "isaac/backend/templates/vaxpy.h"
#include "isaac/backend/templates/reduction.h"
#include "isaac/backend/templates/maxpy.h"
#include "isaac/backend/templates/mreduction.h"
#include "isaac/backend/templates/mproduct.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/model/model.h"
#include "isaac/tools/make_vector.hpp"
#include "isaac/tools/timer.hpp"
#include "convert.hpp"


namespace isaac
{

static double time_event(unsigned long sum, driver::Event const & e)
{ return sum + e.elapsed_time();}


std::string model::define_extension(std::string const & extensions, std::string const & ext)
{
  if(extensions.find(ext)!=std::string::npos)
    return std::string("#pragma OPENCL EXTENSION " + ext + " : enable\n");
  return std::string("");
}

void model::fill_program_name(char* program_name, expressions_tuple const & expressions, binding_policy_t binding_policy)
{
  if (expressions.order()==expressions_tuple::INDEPENDENT)
    *program_name++='i';
  else
    *program_name++='s';
  symbolic_binder* binder = NULL;
  if(binding_policy==BIND_TO_HANDLE)
    binder = new bind_to_handle();
  else
    binder = new bind_all_unique();
  for (const auto & elem : expressions.data())
    traverse(*elem, elem->root(), array_expression_representation_functor(*binder, program_name),true);
  *program_name='\0';
  delete binder;
}

driver::Program& model::init(controller<expressions_tuple> const & expressions)
{
  driver::Context const & context = expressions.x().context();
  std::string pname;
  compilation_options_type const & opt = expressions.compilation_options();
  if(opt.program_name.empty())
  {
    char program_name[256];
    fill_program_name(program_name, expressions.x(), BIND_TO_HANDLE);
    pname = std::string(program_name);
  }
  else
    pname = expressions.compilation_options().program_name;

  std::shared_ptr<driver::Program> & program = programs_[context][pname];
  if(!program)
  {
    driver::Device device = queue_.device();
    std::string extensions = device.extensions();
    std::string all_extensions = define_extension(extensions, "cl_khr_fp64");

   std::string srcs;
    for(int i = 0 ; i < templates_.size() ; ++i){
      char buffer[16];
      sprintf(buffer,"%d",i);
      srcs += templates_[i]->generate(buffer, expressions.x(), device);
    }
    srcs += fallback_->generate("fallback", expressions.x(), device);


    program.reset(new driver::Program(context, all_extensions + srcs));
  }
  return *program;
}

model::model(expression_type etype, numeric_type dtype, predictors::random_forest const & predictor, std::vector< tools::shared_ptr<base> > const & templates, driver::CommandQueue & queue) :
  templates_(templates), fallback_(fallbacks[std::make_pair(etype, dtype)]), predictor_(new predictors::random_forest(predictor)), queue_(queue)
{}


model::model(expression_type etype, numeric_type dtype, base const & tp, driver::CommandQueue & queue) : templates_(1,tp.clone()), fallback_(fallbacks[std::make_pair(etype, dtype)]), queue_(queue)
{}

void model::execute(controller<expressions_tuple> const & expr)
{
  driver::Program & program = init(expr);
  std::vector<int_t> x = templates_[0]->input_sizes(expr.x());

  //Specific tuning if requested
  if(expr.dispatcher_options().tune && hardcoded_.find(x)==hardcoded_.end())
  {
    std::vector<double> timings(templates_.size());
    for(int i = 0 ; i < templates_.size() ; ++i)
    {
      std::list<driver::Event> events;
      try{
        char buffer[16];
        sprintf(buffer,"%d",i);
        templates_[i]->enqueue(queue_, program, buffer, *fallback_, control(expr.x(), execution_options_type(0, &events)));
        queue_.synchronize();
        timings[i] = 1e-9*std::accumulate(events.begin(), events.end(), 0, &time_event);
      }catch(...){
        timings[i] = INFINITY;
      }
    }
    //Fill the override
    std::vector<int_t> x = templates_[0]->input_sizes(expr.x());
    hardcoded_[x] = std::distance(timings.begin(),std::min_element(timings.begin(), timings.end()));
  }

  //Prediction
  int label = 0;
  if(expr.dispatcher_options().label>=0)
    label = expr.dispatcher_options().label;
  else  if(hardcoded_.find(x)!=hardcoded_.end())
    label = hardcoded_.at(x);
  else if(predictor_.get())
  {
    std::vector<float> predictions = predictor_->predict(x);
    label = std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
  }

  //Execution
  char buffer[16];
  sprintf(buffer,"%d",label);
  return templates_[label]->enqueue(queue_, program, buffer, *fallback_, expr);
}

model::templates_container const & model::templates() const
{ return templates_; }

///////////////////

namespace detail
{
  static expression_type get_expression_type(std::string const & name)
  {
    if(name=="vaxpy") return VECTOR_AXPY_TYPE;
    if(name=="dot") return REDUCTION_TYPE;
    if(name=="maxpy") return MATRIX_AXPY_TYPE;
    if(name=="gemvN") return ROW_WISE_REDUCTION_TYPE;
    if(name=="gemvT") return COL_WISE_REDUCTION_TYPE;
    if(name=="gemmNN") return MATRIX_PRODUCT_NN_TYPE;
    if(name=="gemmNT") return MATRIX_PRODUCT_NT_TYPE;
    if(name=="gemmTN") return MATRIX_PRODUCT_TN_TYPE;
    if(name=="gemmTT") return MATRIX_PRODUCT_TT_TYPE;
    throw std::invalid_argument("Invalid expression: " + name);
  }

  static numeric_type get_dtype(std::string const & name)
  {
    if(name=="float32") return FLOAT_TYPE;
    if(name=="float64") return DOUBLE_TYPE;
    throw std::invalid_argument("Invalid datatype: " + name);
  }

  static tools::shared_ptr<base> create(std::string const & template_name, std::vector<int> const & a)
  {
    fetching_policy_type fetch[] = {FETCH_FROM_LOCAL, FETCH_FROM_GLOBAL_STRIDED, FETCH_FROM_GLOBAL_CONTIGUOUS};
    if(template_name=="vaxpy")
      return tools::shared_ptr<base>(new vaxpy(a[0], a[1], a[2], fetch[a[3]]));
    else if(template_name=="dot")
      return tools::shared_ptr<base>(new reduction(a[0], a[1], a[2], fetch[a[3]]));
    else if(template_name=="maxpy")
      return tools::shared_ptr<base>(new maxpy(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemvN")!=std::string::npos)
      return tools::shared_ptr<base>(new mreduction_rows(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemvT")!=std::string::npos)
      return tools::shared_ptr<base>(new mreduction_cols(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemmNN")!=std::string::npos)
      return tools::shared_ptr<base>(new mproduct_nn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemmTN")!=std::string::npos)
      return tools::shared_ptr<base>(new mproduct_tn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemmNT")!=std::string::npos)
      return tools::shared_ptr<base>(new mproduct_nt(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemmTT")!=std::string::npos)
      return tools::shared_ptr<base>(new mproduct_tt(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else
      throw std::invalid_argument("Invalid expression: " + template_name);
  }
}

void import(std::string const & fname, driver::CommandQueue & queue, model_map_t& result)
{
  namespace js = rapidjson;
  //Parse the JSON document
  js::Document document;
  std::ifstream t(fname.c_str());
  if(!t) return;
  std::string str;
  t.seekg(0, std::ios::end);
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  document.Parse<0>(str.c_str());
  //Deserialize
  std::vector<std::string> operations = {"vaxpy", "dot", "maxpy", "gemvN", "gemvT", "gemmNN", "gemmTN", "gemmNT", "gemmTT"};
  std::vector<std::string> dtype = {"float32", "float64"};
  for(auto & operation : operations)
  {
    const char * opcstr = operation.c_str();
    if(document.HasMember(opcstr))
    {
      expression_type etype = detail::get_expression_type(operation);
      for(auto & elem : dtype)
      {
        const char * dtcstr = elem.c_str();
        if(document[opcstr].HasMember(dtcstr))
        {
          numeric_type dtype = detail::get_dtype(elem);

          // Get profiles
          std::vector<tools::shared_ptr<base> > templates;
          js::Value const & profiles = document[opcstr][dtcstr]["profiles"];
          for (js::SizeType id = 0 ; id < profiles.Size() ; ++id)
            templates.push_back(detail::create(operation, tools::to_int_array<int>(profiles[id])));

          if(templates.size()>1)
          {
            // Get predictor
            predictors::random_forest predictor(document[opcstr][dtcstr]["predictor"]);
            result[std::make_pair(etype, dtype)] = tools::shared_ptr<model>(new model(etype, dtype, predictor, templates, queue));
          }
          else
            result[std::make_pair(etype, dtype)] = tools::shared_ptr<model>(new model(etype, dtype, *templates[0], queue));
        }
      }
    }
  }
}


std::map<std::pair<expression_type, numeric_type>, tools::shared_ptr<base> > init_fallback()
{
  typedef tools::shared_ptr<base> ptr_t;
  std::map<std::pair<expression_type, numeric_type>, ptr_t > res;
  numeric_type types[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  for(auto DTYPE : types)
  {
    res[std::make_pair(SCALAR_AXPY_TYPE, DTYPE)] = ptr_t(new vaxpy(1,64,128,FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(VECTOR_AXPY_TYPE, DTYPE)] = ptr_t (new vaxpy(1,64,128,FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(REDUCTION_TYPE, DTYPE)] = ptr_t(new reduction(1,64,128,FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(MATRIX_AXPY_TYPE, DTYPE)] = ptr_t(new maxpy(1,8,8,8,8,FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(ROW_WISE_REDUCTION_TYPE, DTYPE)] = ptr_t(new mreduction_rows(1, 8, 8, 4, 16, FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(COL_WISE_REDUCTION_TYPE, DTYPE)] = ptr_t(new mreduction_cols(1, 8, 8, 64, 8, FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(MATRIX_PRODUCT_NN_TYPE, DTYPE)] = ptr_t(new mproduct_nn(1, 8, 8, 8, 1, 4, 1, 4, FETCH_FROM_LOCAL, FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(MATRIX_PRODUCT_TN_TYPE, DTYPE)] = ptr_t(new mproduct_tn(1, 8, 8, 8, 1, 4, 1, 4, FETCH_FROM_LOCAL, FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(MATRIX_PRODUCT_NT_TYPE, DTYPE)] = ptr_t(new mproduct_nt(1, 8, 8, 8, 1, 4, 1, 4, FETCH_FROM_LOCAL, FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(MATRIX_PRODUCT_TT_TYPE, DTYPE)] = ptr_t(new mproduct_tt(1, 8, 8, 8, 1, 4, 1, 4, FETCH_FROM_LOCAL, FETCH_FROM_LOCAL, 8, 8, true));
  }
  return res;
}

//TODO: Clean everything by overloading operator[]
model_map_t init_models(driver::CommandQueue & queue)
{
  model_map_t res;
  numeric_type dtypes[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  expression_type etypes[] = {SCALAR_AXPY_TYPE, VECTOR_AXPY_TYPE, REDUCTION_TYPE, MATRIX_AXPY_TYPE, ROW_WISE_REDUCTION_TYPE, COL_WISE_REDUCTION_TYPE, MATRIX_PRODUCT_NN_TYPE, MATRIX_PRODUCT_NT_TYPE, MATRIX_PRODUCT_TN_TYPE, MATRIX_PRODUCT_TT_TYPE};

  for(numeric_type dtype: dtypes)
    for(expression_type etype: etypes)
      res[std::make_pair(etype, dtype)] = tools::shared_ptr<model>(new model(etype, dtype, *fallbacks[std::make_pair(etype, dtype)], queue));

  if(const char * homepath = std::getenv("HOME"))
    import(std::string(homepath) + "/.isaac/devices/device0.json", queue, res);
  return res;
}

model_map_t& models(driver::CommandQueue & queue)
{
  std::map<driver::CommandQueue, model_map_t>::iterator it = models_.find(queue);
  if(it == models_.end())
    return models_.insert(std::make_pair(queue, init_models(queue))).first->second;
  return it->second;
}

std::map<std::pair<expression_type, numeric_type>, tools::shared_ptr<base> > fallbacks = init_fallback();
std::map<driver::CommandQueue, model_map_t> models_;

}

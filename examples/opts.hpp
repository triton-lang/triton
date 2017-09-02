#ifndef OPTS_HPP
#define OPTS_HPP

#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <map>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace opts{

class InvalidOptions: public std::exception{
public:
  InvalidOptions(std::string const & msg): msg_("Invalid options: " + msg){}
  const char* what() const throw(){ return msg_.c_str();}
private:
  std::string msg_;
};

/**
 * @class OptionBase
 * @brief Base class for command-line options
*/
class OptionBase{
protected:
  template<class ItType>
  std::vector<std::string>::const_iterator get_option(ItType const & begin, ItType const & end){
    auto it = std::find(begin, end, "--" + name_);
    if(it==end && required_)
      throw InvalidOptions("parameter '" + name_ + "' is mandatory");
    if(parent_ && parent_->parent_ && parent_->get_option(begin, it)==it)
      throw InvalidOptions("parameter '" + name_ + "' needs to be nested in group '" + parent_->name_ + "'");
    return it;
  }

public:
  OptionBase(std::string const & name, std::string const & desc, bool required = false, OptionBase* parent = NULL): name_(name), desc_(desc), required_(required), parent_(parent)
  {}

  virtual std::ostream& usage(std::ostream& os, size_t indent) const{
    if(!desc_.empty())
      os << std::string(indent, ' ') << "--" << "\033[1m" << name_ << "\033[0m" << ": " << desc_ << std::endl;
    return os;
  }

  virtual void parse(std::vector<std::string> const & args, std::map<std::string, void*>& values) = 0;

  std::string const & name() const
  { return name_; }

protected:
  const std::string name_;
  const std::string desc_;
  bool required_;
  OptionBase* parent_;
};

/**
 * @class OptionHelp
 * @brief Automatically added --help option
*/
class OptionHelp: public OptionBase{
public:
  OptionHelp() : OptionBase("help", "Display this message", false){}

  void parse(std::vector<std::string> const & args, std::map<std::string, void*>& values){
    if(get_option(args.begin(), args.end()) != args.end())
      values[name_] = (void*)this;
  }
};


/**
 * @class Option
 * @brief Standard, typed option
*/
template<class T>
class Option: public OptionBase{
public:
  typedef std::function<T(std::string const &)> converter_t;
  typedef std::function<void(T const &)> constraint_t;

public:
  Option(std::string const & name, std::string const & desc, T dft, converter_t convert, constraint_t constraint, OptionBase* parent):
    OptionBase(name, desc, false, parent), default_(new T(dft)), convert_(convert), constraint_(constraint){}

  Option(std::string const & name, std::string const & desc, bool required, converter_t convert, constraint_t constraint, OptionBase* parent):
    OptionBase(name, desc, required, parent), convert_(convert), constraint_(constraint){}

  void parse(std::vector<std::string> const & args, std::map<std::string, void*>& values){
    value_ = default_;
    auto it = get_option(args.begin(), args.end());
    if(it!=args.end()){
      auto next = it + 1;
      if(next==args.end() || next->compare(0, 2, "--")==0)
        throw InvalidOptions("parameter " + name_ + " requires an argument");
      else{
        value_.reset(new T(convert_(*next)));
        constraint_(*value_);
      }
    }
    values[name_] = (void*)value_.get();
  }

  std::ostream& usage(std::ostream& os, size_t indent) const{
    OptionBase::usage(os, indent);
    return os;
  }

private:
  std::shared_ptr<T> default_;
  std::shared_ptr<T> value_;
  converter_t convert_;
  constraint_t constraint_;
};


/**
 * @class SwitchOption
 * @brief Boolean option activated with --flag or --no-flag
*/
class SwitchOption: public OptionBase{
public:
  SwitchOption(std::string const & name, std::string const & desc, bool dft, OptionBase* parent):
    OptionBase(name, desc, false, parent), default_(dft)
  {}

  void parse(std::vector<std::string> const & args, std::map<std::string, void*>& values){
    auto it_true = std::find(args.begin(), args.end(), "--" + name_);
    auto it_false = std::find(args.begin(), args.end(), "--no-" + name_);
    value_.reset(new bool(default_));
    if(it_true != args.end()) value_.reset(new bool(true));
    if(it_false != args.end()) value_.reset(new bool(false));
    values[name_] = (void*)value_.get();
  }

private:
  bool default_;
  std::shared_ptr<bool> value_;
};

/* Pre-defined converters */
template<class T>
class MapConverter{
public:
  MapConverter(std::map<std::string, T> const & values): values_(values){}

  inline T operator()(std::string const & str){
    if(values_.find(str) == values_.end())
      throw InvalidOptions("value " + str + " is invalid");
    return values_.at(str);
  }

private:
  std::map<std::string, T> values_;
};

//Read type from stream
template<class T>
class StreamConverter{
public:
  T operator()(std::string const & str){
    T value;
    std::istringstream iss(str);
    iss >> value;
    return value;
  }
};

//Read vector from stream
template<class T>
class StreamConverter<std::vector<T>>{
public:
  std::vector<T> operator()(std::string const & str){
    std::vector<T> result;
    std::istringstream iss(str);
    std::string token;
    while(std::getline(iss, token, ','))
      result.push_back(StreamConverter<T>()(token));
    return result;
  }
};

//Read tuple from stream
template<class... Args>
class StreamConverter<std::tuple<Args...>>{
  template<size_t I, class T, class... U>
  struct TupleReader{
    static std::tuple<T, U...> get(std::istringstream& iss){
      auto x = TupleReader<0,T>::get(iss);
      auto y = TupleReader<I-1, U...>::get(iss);
      return std::tuple_cat(x, y);
    }
  };

  template<class T>
  struct TupleReader<0, T>{
    static std::tuple<T> get(std::istringstream& iss){
      std::string token;
      std::getline(iss, token, ',');
      return std::make_tuple(StreamConverter<T>()(token));
    }
  };

public:
  inline std::tuple<Args...> operator()(std::string const & str){
    std::istringstream iss(str);
    return TupleReader<sizeof...(Args) - 1, Args...>::get(iss);
  }
};

/* Pre-defined constraints */
struct NoOp {
  template<class T>
  void operator()(T const &) {}
};

class SizeConstraint{
public:
  SizeConstraint(size_t size): size_(size){}

  template<class T>
  void operator()(std::vector<T> const & x) const {
    if(x.size()!=size_)
      throw InvalidOptions("parameter must have size " + std::to_string(size_));
  }
private:
  size_t size_;
};

class OneOf{
public:
  OneOf(std::vector<std::string> keys): keys_(keys){}

  void operator()(std::map<std::string, void*> values){
    std::vector<std::string> keys;
    for(auto& x: values)
      keys.push_back(x.first);

    size_t found = 0;
    for(auto& x: keys_)
      if(std::find(keys.begin(), keys.end(), x) != keys.end())
        found++;

    std::string msg;
    for(size_t i = 0; i < keys_.size(); ++i)
      msg += (i>0?", ":"") + keys_[i];

    if(found != 1)
      throw InvalidOptions(std::string(found<1?"At least":"Only") + " one of the following flags must be specified: " + msg);
  }

private:
  std::vector<std::string> keys_;
};

/**
 * @class Options
 * @brief Container for multiple options
 */
class Options: public OptionBase{
public:
  typedef std::function<void(std::map<std::string, void*> const &)> constraint_t;

  std::map<std::string, std::string> set_to_map(std::set<std::string> const & set){
    std::map<std::string, std::string> tmp;
    for(std::string x: set)
      tmp.insert(std::make_pair(x, x));
    return tmp;
  }

public:
  Options(std::string const & name, std::string const & desc, OptionBase* parent): OptionBase(name, desc, false, parent)
  {}

  std::ostream& usage(std::ostream& os, size_t indent) const{
    OptionBase::usage(os, indent);
    for(auto& opt: opts_)
      opt->usage(os, indent + (parent_==NULL)?0:2);
    return os;
  }

  void parse(std::vector<std::string> const & args, std::map<std::string, void*>& values){
    if(parent_==NULL || get_option(args.begin(), args.end()) != args.end()){
      for(auto& opt: opts_)
        opt->parse(args, values_);
      for(auto& constraint: constraints_)
        constraint(values_);
      values[name_] = (void*)&values_;
    }
  }

  void parse(int argc, char* argv[]){
    std::vector<std::string> args(argv, argv + argc);
    parse(args, values_);
  }

  template<class T>
  void add(std::string const & name, std::string const & desc, T dft, typename Option<T>::constraint_t constraint = NoOp())
  { opts_.push_back(std::make_shared<Option<T>>(name, desc, dft, StreamConverter<T>(), constraint, this));}

  template<class T>
  void add(std::string const & name, std::string const & desc, typename Option<T>::constraint_t constraint = NoOp())
  { opts_.push_back(std::make_shared<Option<T>>(name, desc, false, StreamConverter<T>(), constraint, this));}

  void add(std::string const & name, std::string const & desc, std::string dft, std::set<std::string> values)
  { add<std::string>(name, desc, dft, set_to_map(values)); }

  void add(std::string const & name, std::string const & desc, std::set<std::string> values)
  { add<std::string>(name, desc, set_to_map(values)); }

  template<class T>
  void add(std::string const & name, std::string const & desc, std::string dft, std::map<std::string, T> values, typename Option<T>::constraint_t constraint = NoOp())
  { opts_.push_back(std::make_shared<Option<T>>(name, desc, values.at(dft), MapConverter<T>(values), constraint, this)); }

  template<class T>
  void add(std::string const & name, std::string const & desc, std::map<std::string, T> values, typename Option<T>::constraint_t constraint = NoOp())
  { opts_.push_back(std::make_shared<Option<T>>(name, desc, false, MapConverter<T>(values), constraint, this)); }


  void add_switch(std::string const & name, std::string const & desc, bool dft = true)
  { opts_.push_back(std::make_shared<SwitchOption>(name, desc, dft, this)); }

  void add(OptionBase* opt)
  { opts_.push_back(std::shared_ptr<OptionBase>(opt)); }

  Options* add_group(std::string const & name, std::string const & desc){
    opts_.push_back(std::make_shared<Options>(name, desc, this));
    return (Options*)opts_.back().get();
  }

  void add_constraint(constraint_t const & constraint){
    constraints_.push_back(constraint);
  }

  bool has(std::string const & name)
  { return values_.find(name) != values_.end() && values_.at(name)!=NULL; }

  template<class T>
  T get(std::string const & name)
  { return *((T*)values_[name]); }

private:
  std::vector<std::shared_ptr<OptionBase>> opts_;
  std::map<std::string, void*> values_;
  std::vector<constraint_t> constraints_;
};


/* Application */
class Application{
private:
  void show_help() const{
    std::cerr << "Usage: " << name_ << " [OPTS]" << std::endl;
    std::cerr << "Description: " << desc_ << std::endl;
    opts_.usage(std::cerr, 0);
  }

public:
  Application(std::string const & name, std::string const & desc): name_(name), desc_(desc), opts_("root","",NULL)
  { opts_.add(new OptionHelp()); }

  void parse(int argc, char* argv[]){
    try{
      opts_.parse(argc, argv);
    }catch(InvalidOptions const & e){
      std::cerr << e.what() << std::endl;
      show_help();
      exit(EXIT_FAILURE);
    }
    if(opts_.has("help")){
      show_help();
      exit(EXIT_FAILURE);
    }
  }

  Options* options()
  { return &opts_; }

private:
  std::string name_;
  std::string desc_;
  Options opts_;
};

}

#endif

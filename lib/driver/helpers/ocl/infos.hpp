#ifndef ISAAC_DRIVER_HELPERS_OCL_INFOS_HPP_
#define ISAAC_DRIVER_HELPERS_OCL_INFOS_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */



#include "isaac/driver/common.h"
#include <vector>
#include <string>

namespace isaac
{
namespace driver
{
namespace ocl
{

  /** @brief Implementation details for the OpenCL managment layer in ViennaCL */
namespace detail{

/** @brief Helper class for obtaining informations from the OpenCL backend. Deprecated! */
template<typename T>
struct info;

/** \cond */
template<>
struct info<cl_mem>
{
  typedef cl_mem_info type;

  static void get(cl_mem handle, cl_mem_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret)
  {
      dispatch::clGetMemObjectInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_device_id>
{
  typedef cl_device_info type;

  static void get(cl_device_id handle, cl_device_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret)
  {
      dispatch::clGetDeviceInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_kernel>
{
  typedef cl_kernel_info type;

  static void get(cl_kernel handle, cl_kernel_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetKernelInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }

  static void get(cl_kernel handle, cl_device_id dev_id, cl_kernel_work_group_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetKernelWorkGroupInfo(handle, dev_id, param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_context>
{
  typedef cl_context_info type;

  static void get(cl_context handle, cl_context_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetContextInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_program>
{
  typedef cl_program_info type;

  static void get(cl_program handle, cl_program_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetProgramInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }

  static void get(cl_program handle, cl_device_id device, cl_program_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetProgramBuildInfo(handle,device,param_name,param_value_size,param_value,param_value_size_ret);
  }
};


template<>
struct info<cl_event>
{
  typedef cl_profiling_info type;
  static void get(cl_event handle, cl_profiling_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetEventProfilingInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_command_queue>
{
  typedef cl_command_queue_info type;
  static void get(cl_command_queue handle, cl_profiling_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetCommandQueueInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

template<>
struct info<cl_platform_id>
{
  typedef cl_command_queue_info type;
  static void get(cl_platform_id handle, cl_profiling_info param_name,size_t param_value_size,void *param_value,size_t *param_value_size_ret){
      dispatch::clGetPlatformInfo(handle,param_name,param_value_size,param_value,param_value_size_ret);
  }
};

//Info getter
//Some intelligence is needed for some types
template<class RES_T>
struct get_info_impl{

    template<class MEM_T, class INFO_T>
    RES_T operator()(MEM_T const & mem, INFO_T const & info){
        RES_T res;
        detail::info<MEM_T>::get(mem,info,sizeof(RES_T),&res,NULL);
        return res;
    }

    template<class MEM_T, class ARG_MEM_T, class INFO_T>
    RES_T operator()(MEM_T const & mem, ARG_MEM_T const & arg_mem, INFO_T const & info){
        RES_T res;
        detail::info<MEM_T>::get(mem,arg_mem, info,sizeof(RES_T),&res,NULL);
        return res;
    }
};

template<>
struct get_info_impl<std::string>{

    template<class MEM_T, class INFO_T>
    std::string operator()(const MEM_T &mem, const INFO_T &info){
        char buff[1024];
        detail::info<MEM_T>::get(mem,info,1024,buff,NULL);
        return std::string(buff);
    }

    template<class MEM_T, class ARG_MEM_T, class INFO_T>
    std::string operator()(MEM_T const & mem, ARG_MEM_T const & arg_mem, INFO_T const & info){
        char buff[1024];
        detail::info<MEM_T>::get(mem,arg_mem,info,1024,buff,NULL);
        return std::string(buff);
    }
};

template<class T>
struct get_info_impl<std::vector<T> >
{
    template<class MEM_T, class INFO_T>
    std::vector<T> operator()(const MEM_T &mem, const INFO_T &info)
    {
        size_t vec_size;
        detail::info<MEM_T>::get(mem,info,0,NULL,&vec_size);
        std::vector<T> res(vec_size/sizeof(T));
        detail::info<MEM_T>::get(mem,info,vec_size,res.data(),NULL);
        return res;
    }

    template<class MEM_T, class ARG_MEM_T, class INFO_T>
    std::vector<T> operator()(MEM_T const & mem, ARG_MEM_T const & arg_mem, INFO_T const & info)
    {
        size_t vec_size;
        detail::info<MEM_T>::get(mem,arg_mem,info,0,NULL,&vec_size);
        std::vector<T> res(vec_size/sizeof(T));
        detail::info<MEM_T>::get(mem,arg_mem,info,vec_size,res.data(),NULL);
        return res;
    }
};

template<typename T, typename info<T>::type param>
struct return_type;
/** \endcond */

/** \cond */
 #define SET_INFO_RETURN_TYPE(DATA_TYPE,NAME,RETURN_TYPE) template<> struct return_type<DATA_TYPE, NAME> { typedef RETURN_TYPE Result; }

SET_INFO_RETURN_TYPE(cl_command_queue, CL_QUEUE_CONTEXT, cl_context);
SET_INFO_RETURN_TYPE(cl_command_queue, CL_QUEUE_DEVICE, cl_device_id);
SET_INFO_RETURN_TYPE(cl_command_queue, CL_QUEUE_REFERENCE_COUNT, cl_uint);
SET_INFO_RETURN_TYPE(cl_command_queue, CL_QUEUE_PROPERTIES, cl_command_queue_properties);

SET_INFO_RETURN_TYPE(cl_context, CL_CONTEXT_DEVICES, std::vector<cl_device_id>);
SET_INFO_RETURN_TYPE(cl_context, CL_CONTEXT_NUM_DEVICES, cl_uint);
SET_INFO_RETURN_TYPE(cl_context, CL_CONTEXT_REFERENCE_COUNT, cl_uint);
SET_INFO_RETURN_TYPE(cl_context, CL_CONTEXT_PROPERTIES, cl_context_properties);

SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_ADDRESS_BITS, cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_AVAILABLE, cl_bool);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_COMPILER_AVAILABLE, cl_bool);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_WAVEFRONT_WIDTH_AMD, cl_uint);

SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_ENDIAN_LITTLE, cl_bool);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_EXTENSIONS, std::string);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong);
//SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE_SUPPORT, cl_bool);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE2D_MAX_HEIGHT , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE2D_MAX_WIDTH , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE3D_MAX_DEPTH , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE3D_MAX_HEIGHT , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_IMAGE3D_MAX_WIDTH , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_CLOCK_FREQUENCY , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_COMPUTE_UNITS , cl_uint); //The minimum value is 1
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_CONSTANT_ARGS  , cl_uint); //The minimum value is 8
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE   , cl_ulong); //The minimum value is 64 KB
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_MEM_ALLOC_SIZE , cl_ulong); //The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024)
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_PARAMETER_SIZE  , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_READ_IMAGE_ARGS  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_SAMPLERS , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_WORK_GROUP_SIZE , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_WORK_ITEM_SIZES , std::vector<size_t>);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MAX_WRITE_IMAGE_ARGS , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MEM_BASE_ADDR_ALIGN  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_NAME , std::string);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PLATFORM , cl_platform_id);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PROFILE , std::string);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_PROFILING_TIMER_RESOLUTION , size_t);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_QUEUE_PROPERTIES , cl_command_queue_properties);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_SINGLE_FP_CONFIG  , cl_device_fp_config);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_TYPE , cl_device_type);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_VENDOR , std::string);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_VENDOR_ID  , cl_uint);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DEVICE_VERSION  , std::string);
SET_INFO_RETURN_TYPE(cl_device_id,  CL_DRIVER_VERSION  , std::string);

SET_INFO_RETURN_TYPE(cl_event, CL_PROFILING_COMMAND_QUEUED, cl_ulong);
SET_INFO_RETURN_TYPE(cl_event, CL_PROFILING_COMMAND_SUBMIT, cl_ulong);
SET_INFO_RETURN_TYPE(cl_event, CL_PROFILING_COMMAND_START, cl_ulong);
SET_INFO_RETURN_TYPE(cl_event, CL_PROFILING_COMMAND_END, cl_ulong);

SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_FUNCTION_NAME, std::string);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_NUM_ARGS, cl_uint);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_REFERENCE_COUNT, cl_uint);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_CONTEXT, cl_context);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_PROGRAM, cl_program);


SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_WORK_GROUP_SIZE, size_t);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_COMPILE_WORK_GROUP_SIZE, std::vector<size_t>);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_LOCAL_MEM_SIZE, cl_ulong);
SET_INFO_RETURN_TYPE(cl_kernel,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_t);

SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_TYPE, cl_mem_object_type);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_FLAGS, cl_mem_flags);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_SIZE, size_t);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_HOST_PTR, void*);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_MAP_COUNT, cl_uint);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_REFERENCE_COUNT, cl_uint);
SET_INFO_RETURN_TYPE(cl_mem,CL_MEM_CONTEXT, cl_context);

SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_CONTEXT,cl_context);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_DEVICES,std::vector<cl_device_id>);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_NUM_DEVICES,cl_uint);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_SOURCE,std::string);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_BINARY_SIZES,std::vector<size_t>);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_BINARIES,std::vector<unsigned char*>);
//Build
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_BUILD_STATUS, cl_build_status);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_BUILD_OPTIONS, std::string);
SET_INFO_RETURN_TYPE(cl_program,CL_PROGRAM_BUILD_LOG, std::string);

SET_INFO_RETURN_TYPE(cl_platform_id,CL_PLATFORM_PROFILE, std::string);
SET_INFO_RETURN_TYPE(cl_platform_id,CL_PLATFORM_VERSION, std::string);
SET_INFO_RETURN_TYPE(cl_platform_id,CL_PLATFORM_NAME, std::string);
SET_INFO_RETURN_TYPE(cl_platform_id,CL_PLATFORM_VENDOR, std::string);
SET_INFO_RETURN_TYPE(cl_platform_id,CL_PLATFORM_EXTENSIONS, std::string);

#undef SET_INFO_RETURN_TYPE

  /** \endcond */
}

template<cl_device_info param>
typename detail::return_type<cl_device_id, param>::Result info(cl_device_id const & handle){
    typedef typename detail::return_type<cl_device_id, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

template<cl_mem_info param>
typename detail::return_type<cl_mem, param>::Result info(cl_mem const & handle){
    typedef typename detail::return_type<cl_mem, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

//Program

template<cl_program_info param>
typename detail::return_type<cl_program, param>::Result info(cl_program const & handle){
    typedef typename detail::return_type<cl_program, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

template<>
inline typename detail::return_type<cl_program, CL_PROGRAM_BINARIES>::Result info<CL_PROGRAM_BINARIES>(cl_program const & handle)
{
    std::vector<unsigned char *> res;
    std::vector<size_t> sizes = info<CL_PROGRAM_BINARY_SIZES>(handle);
    for(size_t s: sizes)
        res.push_back(new unsigned char[s]);
    dispatch::clGetProgramInfo(handle, CL_PROGRAM_BINARIES, sizeof(unsigned char**), (void*)res.data(), NULL);
    return res;
}

template<cl_program_build_info param>
typename detail::return_type<cl_program, param>::Result info(cl_program const & phandle, cl_device_id const & dhandle){
    typedef typename detail::return_type<cl_program, param>::Result res_t;
    return detail::get_info_impl<res_t>()(phandle,dhandle,param);
}

//Kernel
template<cl_kernel_info param>
typename detail::return_type<cl_kernel, param>::Result info(cl_kernel const & handle){
    typedef typename detail::return_type<cl_kernel, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

template<cl_kernel_work_group_info param>
typename detail::return_type<cl_kernel, param>::Result info(cl_kernel const & khandle, cl_device_id const & dhandle){
    typedef typename detail::return_type<cl_kernel, param>::Result res_t;
    return detail::get_info_impl<res_t>()(khandle,dhandle,param);
}

//Context
template<cl_context_info param>
typename detail::return_type<cl_context, param>::Result info(cl_context const & handle){
    typedef typename detail::return_type<cl_context, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

//Event
template<cl_profiling_info param>
typename detail::return_type<cl_event, param>::Result info(cl_event const & handle){
    typedef typename detail::return_type<cl_event, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

//Command queue
template<cl_command_queue_info param>
typename detail::return_type<cl_command_queue, param>::Result info(cl_command_queue const & handle){
    typedef typename detail::return_type<cl_command_queue, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

//Plaftform
template<cl_platform_info param>
typename detail::return_type<cl_platform_id, param>::Result info(cl_platform_id const & handle){
    typedef typename detail::return_type<cl_platform_id, param>::Result res_t;
    return detail::get_info_impl<res_t>()(handle,param);
}

template<class OCL_TYPE, typename detail::info<OCL_TYPE>::type param>
typename detail::return_type<OCL_TYPE, param>::Result info(OCL_TYPE const & handle){
    return info(handle.get());
}



template<class OCL_TYPE, class OCL_TYPE_ARG, typename detail::info<OCL_TYPE>::type param>
typename detail::return_type<OCL_TYPE, param>::Result info(OCL_TYPE const & handle, OCL_TYPE_ARG const & arg_handle){
    return info(handle.get(), arg_handle.get());
}

}
}
}
#endif // INFOS_HPP

// automatically generated
/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_HSA_OSTREAM_OPS_H_
#define INC_HSA_OSTREAM_OPS_H_

#include "roctracer.h"

#ifdef __cplusplus
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>

namespace roctracer {
namespace hsa_support {
static int HSA_depth_max = 1;
static int HSA_depth_max_cnt = 0;
static std::string HSA_structs_regex = "";
// begin ostream ops for HSA 
// basic ostream ops
namespace detail {
  inline static void print_escaped_string(std::ostream& out, const char *v, size_t len) {
    out << '"'; 
    for (size_t i = 0; i < len && v[i]; ++i) {
      switch (v[i]) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\\b"; break;
      case '\f': out << "\\\f"; break;
      case '\n': out << "\\\n"; break;
      case '\r': out << "\\\r"; break;
      case '\t': out << "\\\t"; break;
      default:
        if (std::isprint((unsigned char)v[i])) std::operator<<(out, v[i]);
        else {
          std::ios_base::fmtflags flags(out.flags());
          out << "\\x" << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)(unsigned char)v[i];
          out.flags(flags);
        }
        break;
      }
    }
    out << '"'; 
  }

  template <typename T>
  inline static std::ostream& operator<<(std::ostream& out, const T& v) {
     using std::operator<<;
     static bool recursion = false;
     if (recursion == false) { recursion = true; out << v; recursion = false; }
     return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const unsigned char &v) {
    out << (unsigned int)v;
    return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const char &v) {
    out << (unsigned char)v;
    return out;
  }

  template <size_t N>
  inline static std::ostream &operator<<(std::ostream &out, const char (&v)[N]) {
    print_escaped_string(out, v, N);
    return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const char *v) {
    print_escaped_string(out, v, strlen(v));
    return out;
  }
// End of basic ostream ops

inline static std::ostream& operator<<(std::ostream& out, const hsa_dim3_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_dim3_t::z").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hsa_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_dim3_t::y").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hsa_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_dim3_t::x").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hsa_support::detail::operator<<(out, v.x);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_agent_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_agent_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_cache_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_cache_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_signal_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_signal_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_signal_group_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_signal_group_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_region_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_region_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_queue_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_queue_t::id").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "id=");
      roctracer::hsa_support::detail::operator<<(out, v.id);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_queue_t::reserved1").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved1=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved1);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_queue_t::size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "size=");
      roctracer::hsa_support::detail::operator<<(out, v.size);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_queue_t::doorbell_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "doorbell_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.doorbell_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_queue_t::features").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "features=");
      roctracer::hsa_support::detail::operator<<(out, v.features);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_queue_t::type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hsa_support::detail::operator<<(out, v.type);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_kernel_dispatch_packet_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_kernel_dispatch_packet_t::completion_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "completion_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.completion_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::reserved2").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::kernel_object").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "kernel_object=");
      roctracer::hsa_support::detail::operator<<(out, v.kernel_object);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::group_segment_size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "group_segment_size=");
      roctracer::hsa_support::detail::operator<<(out, v.group_segment_size);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::private_segment_size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "private_segment_size=");
      roctracer::hsa_support::detail::operator<<(out, v.private_segment_size);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::grid_size_z").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "grid_size_z=");
      roctracer::hsa_support::detail::operator<<(out, v.grid_size_z);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::grid_size_y").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "grid_size_y=");
      roctracer::hsa_support::detail::operator<<(out, v.grid_size_y);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::grid_size_x").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "grid_size_x=");
      roctracer::hsa_support::detail::operator<<(out, v.grid_size_x);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::reserved0").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::workgroup_size_z").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "workgroup_size_z=");
      roctracer::hsa_support::detail::operator<<(out, v.workgroup_size_z);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::workgroup_size_y").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "workgroup_size_y=");
      roctracer::hsa_support::detail::operator<<(out, v.workgroup_size_y);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::workgroup_size_x").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "workgroup_size_x=");
      roctracer::hsa_support::detail::operator<<(out, v.workgroup_size_x);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::setup").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "setup=");
      roctracer::hsa_support::detail::operator<<(out, v.setup);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_kernel_dispatch_packet_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_agent_dispatch_packet_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_agent_dispatch_packet_t::completion_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "completion_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.completion_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_agent_dispatch_packet_t::reserved2").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_agent_dispatch_packet_t::arg").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "arg=");
      roctracer::hsa_support::detail::operator<<(out, v.arg);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_agent_dispatch_packet_t::reserved0").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_agent_dispatch_packet_t::type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hsa_support::detail::operator<<(out, v.type);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_agent_dispatch_packet_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_barrier_and_packet_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_barrier_and_packet_t::completion_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "completion_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.completion_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_and_packet_t::reserved2").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_and_packet_t::dep_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dep_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.dep_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_and_packet_t::reserved1").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved1=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved1);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_and_packet_t::reserved0").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_and_packet_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_barrier_or_packet_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_barrier_or_packet_t::completion_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "completion_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.completion_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_or_packet_t::reserved2").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_or_packet_t::dep_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dep_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.dep_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_or_packet_t::reserved1").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved1=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved1);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_or_packet_t::reserved0").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_barrier_or_packet_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_isa_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_isa_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_wavefront_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_wavefront_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_code_object_reader_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_code_object_reader_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_executable_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_executable_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_loaded_code_object_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_loaded_code_object_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_executable_symbol_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_executable_symbol_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_code_object_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_code_object_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_callback_data_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_callback_data_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_code_symbol_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_code_symbol_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_image_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_format_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_image_format_t::channel_order").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "channel_order=");
      roctracer::hsa_support::detail::operator<<(out, v.channel_order);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_format_t::channel_type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "channel_type=");
      roctracer::hsa_support::detail::operator<<(out, v.channel_type);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_descriptor_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_image_descriptor_t::format").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "format=");
      roctracer::hsa_support::detail::operator<<(out, v.format);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_descriptor_t::array_size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "array_size=");
      roctracer::hsa_support::detail::operator<<(out, v.array_size);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_descriptor_t::depth").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "depth=");
      roctracer::hsa_support::detail::operator<<(out, v.depth);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_descriptor_t::height").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hsa_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_descriptor_t::width").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hsa_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_descriptor_t::geometry").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "geometry=");
      roctracer::hsa_support::detail::operator<<(out, v.geometry);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_data_info_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_image_data_info_t::alignment").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "alignment=");
      roctracer::hsa_support::detail::operator<<(out, v.alignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_data_info_t::size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "size=");
      roctracer::hsa_support::detail::operator<<(out, v.size);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_region_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_image_region_t::range").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "range=");
      roctracer::hsa_support::detail::operator<<(out, v.range);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_image_region_t::offset").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "offset=");
      roctracer::hsa_support::detail::operator<<(out, v.offset);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_sampler_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_sampler_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_sampler_descriptor_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_sampler_descriptor_t::address_mode").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "address_mode=");
      roctracer::hsa_support::detail::operator<<(out, v.address_mode);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_sampler_descriptor_t::filter_mode").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "filter_mode=");
      roctracer::hsa_support::detail::operator<<(out, v.filter_mode);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_sampler_descriptor_t::coordinate_mode").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "coordinate_mode=");
      roctracer::hsa_support::detail::operator<<(out, v.coordinate_mode);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_images_1_00_pfn_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_sampler_destroy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_sampler_destroy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_sampler_destroy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_sampler_create").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_sampler_create=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_sampler_create);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_copy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_copy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_copy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_destroy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_destroy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_destroy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_data_get_info").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_data_get_info=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_data_get_info);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_get_capability").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_get_capability=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_get_capability);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_images_1_pfn_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_data_get_info_with_layout").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_data_get_info_with_layout=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_data_get_info_with_layout);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_get_capability_with_layout").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_get_capability_with_layout=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_get_capability_with_layout);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_sampler_destroy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_sampler_destroy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_sampler_destroy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_sampler_create").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_sampler_create=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_sampler_create);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_copy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_copy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_copy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_destroy").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_destroy=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_destroy);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_data_get_info").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_data_get_info=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_data_get_info);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_get_capability").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hsa_ext_image_get_capability=");
      roctracer::hsa_support::detail::operator<<(out, v.hsa_ext_image_get_capability);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_vendor_packet_header_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_vendor_packet_header_t::reserved").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_vendor_packet_header_t::AmdFormat").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "AmdFormat=");
      roctracer::hsa_support::detail::operator<<(out, v.AmdFormat);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_vendor_packet_header_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_barrier_value_packet_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_barrier_value_packet_t::completion_signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "completion_signal=");
      roctracer::hsa_support::detail::operator<<(out, v.completion_signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::reserved3").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved3=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved3);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::reserved2").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::reserved1").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved1=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved1);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::cond").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cond=");
      roctracer::hsa_support::detail::operator<<(out, v.cond);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::mask").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mask=");
      roctracer::hsa_support::detail::operator<<(out, v.mask);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::value").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "value=");
      roctracer::hsa_support::detail::operator<<(out, v.value);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::signal").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "signal=");
      roctracer::hsa_support::detail::operator<<(out, v.signal);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::reserved0").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hsa_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_barrier_value_packet_t::header").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "header=");
      roctracer::hsa_support::detail::operator<<(out, v.header);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_hdp_flush_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_hdp_flush_t::HDP_REG_FLUSH_CNTL").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "HDP_REG_FLUSH_CNTL=");
      roctracer::hsa_support::detail::operator<<(out, v.HDP_REG_FLUSH_CNTL);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_hdp_flush_t::HDP_MEM_FLUSH_CNTL").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "HDP_MEM_FLUSH_CNTL=");
      roctracer::hsa_support::detail::operator<<(out, v.HDP_MEM_FLUSH_CNTL);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_profiling_dispatch_time_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_profiling_dispatch_time_t::end").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "end=");
      roctracer::hsa_support::detail::operator<<(out, v.end);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_profiling_dispatch_time_t::start").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "start=");
      roctracer::hsa_support::detail::operator<<(out, v.start);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_profiling_async_copy_time_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_profiling_async_copy_time_t::end").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "end=");
      roctracer::hsa_support::detail::operator<<(out, v.end);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_profiling_async_copy_time_t::start").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "start=");
      roctracer::hsa_support::detail::operator<<(out, v.start);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_pool_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_memory_pool_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_pitched_ptr_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_pitched_ptr_t::slice").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "slice=");
      roctracer::hsa_support::detail::operator<<(out, v.slice);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_pitched_ptr_t::pitch").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pitch=");
      roctracer::hsa_support::detail::operator<<(out, v.pitch);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_pool_link_info_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_memory_pool_link_info_t::numa_distance").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numa_distance=");
      roctracer::hsa_support::detail::operator<<(out, v.numa_distance);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_pool_link_info_t::link_type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "link_type=");
      roctracer::hsa_support::detail::operator<<(out, v.link_type);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_pool_link_info_t::max_bandwidth").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "max_bandwidth=");
      roctracer::hsa_support::detail::operator<<(out, v.max_bandwidth);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_pool_link_info_t::min_bandwidth").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "min_bandwidth=");
      roctracer::hsa_support::detail::operator<<(out, v.min_bandwidth);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_pool_link_info_t::max_latency").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "max_latency=");
      roctracer::hsa_support::detail::operator<<(out, v.max_latency);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_pool_link_info_t::min_latency").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "min_latency=");
      roctracer::hsa_support::detail::operator<<(out, v.min_latency);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_image_descriptor_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_image_descriptor_t::data").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "data=");
      roctracer::hsa_support::detail::operator<<(out, v.data);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_image_descriptor_t::deviceID").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "deviceID=");
      roctracer::hsa_support::detail::operator<<(out, v.deviceID);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_image_descriptor_t::version").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "version=");
      roctracer::hsa_support::detail::operator<<(out, v.version);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_pointer_info_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_pointer_info_t::global_flags").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "global_flags=");
      roctracer::hsa_support::detail::operator<<(out, v.global_flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_pointer_info_t::agentOwner").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "agentOwner=");
      roctracer::hsa_support::detail::operator<<(out, v.agentOwner);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_pointer_info_t::sizeInBytes").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sizeInBytes=");
      roctracer::hsa_support::detail::operator<<(out, v.sizeInBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_pointer_info_t::type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hsa_support::detail::operator<<(out, v.type);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_pointer_info_t::size").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "size=");
      roctracer::hsa_support::detail::operator<<(out, v.size);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_ipc_memory_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_ipc_memory_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_gpu_memory_fault_info_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_gpu_memory_fault_info_t::fault_reason_mask").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "fault_reason_mask=");
      roctracer::hsa_support::detail::operator<<(out, v.fault_reason_mask);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_gpu_memory_fault_info_t::virtual_address").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "virtual_address=");
      roctracer::hsa_support::detail::operator<<(out, v.virtual_address);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_gpu_memory_fault_info_t::agent").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "agent=");
      roctracer::hsa_support::detail::operator<<(out, v.agent);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_gpu_hw_exception_info_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_gpu_hw_exception_info_t::reset_cause").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reset_cause=");
      roctracer::hsa_support::detail::operator<<(out, v.reset_cause);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_gpu_hw_exception_info_t::reset_type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reset_type=");
      roctracer::hsa_support::detail::operator<<(out, v.reset_type);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_gpu_hw_exception_info_t::agent").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "agent=");
      roctracer::hsa_support::detail::operator<<(out, v.agent);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_event_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_event_t::event_type").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "event_type=");
      roctracer::hsa_support::detail::operator<<(out, v.event_type);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_svm_attribute_pair_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_svm_attribute_pair_t::value").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "value=");
      roctracer::hsa_support::detail::operator<<(out, v.value);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_svm_attribute_pair_t::attribute").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "attribute=");
      roctracer::hsa_support::detail::operator<<(out, v.attribute);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_vmem_alloc_handle_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_vmem_alloc_handle_t::handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle=");
      roctracer::hsa_support::detail::operator<<(out, v.handle);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_access_desc_t& v)
{
  std::operator<<(out, '{');
  HSA_depth_max_cnt++;
  if (HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max) {
    if (std::string("hsa_amd_memory_access_desc_t::agent_handle").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "agent_handle=");
      roctracer::hsa_support::detail::operator<<(out, v.agent_handle);
      std::operator<<(out, ", ");
    }
    if (std::string("hsa_amd_memory_access_desc_t::permissions").find(HSA_structs_regex) != std::string::npos)   {
      std::operator<<(out, "permissions=");
      roctracer::hsa_support::detail::operator<<(out, v.permissions);
    }
  };
  HSA_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
// end ostream ops for HSA 
};};};

inline static std::ostream& operator<<(std::ostream& out, const hsa_dim3_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_agent_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_cache_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_signal_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_signal_group_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_region_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_queue_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_kernel_dispatch_packet_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_agent_dispatch_packet_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_barrier_and_packet_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_barrier_or_packet_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_isa_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_wavefront_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_code_object_reader_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_executable_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_loaded_code_object_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_executable_symbol_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_code_object_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_callback_data_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_code_symbol_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_format_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_descriptor_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_data_info_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_image_region_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_sampler_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_sampler_descriptor_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_images_1_00_pfn_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_ext_images_1_pfn_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_vendor_packet_header_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_barrier_value_packet_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_hdp_flush_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_profiling_dispatch_time_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_profiling_async_copy_time_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_pool_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_pitched_ptr_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_pool_link_info_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_image_descriptor_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_pointer_info_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_ipc_memory_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_gpu_memory_fault_info_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_gpu_hw_exception_info_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_event_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_svm_attribute_pair_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_vmem_alloc_handle_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hsa_amd_memory_access_desc_t& v)
{
  roctracer::hsa_support::detail::operator<<(out, v);
  return out;
}

#endif //__cplusplus
#endif // INC_HSA_OSTREAM_OPS_H_
 

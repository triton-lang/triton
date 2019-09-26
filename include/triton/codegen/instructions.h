#ifndef _TRITON_CODEGEN_INSTRUCTIONS_H_
#define _TRITON_CODEGEN_INSTRUCTIONS_H_

#include "triton/ir/enums.h"
#include <map>
#include <vector>

namespace triton{
namespace codegen{


enum storage_info_t {
  NONE,
  ANY,
  SHARED,
  DISTRIBUTED,
  REPLICATED
};

typedef std::pair<storage_info_t, std::vector<storage_info_t>> inst_storage_info_t;
static const std::map<ir::value_id_t, inst_storage_info_t> storage_info = {
  // scalars
  { ir::INST_GET_PROGRAM_ID,       {REPLICATED, {}}},
  { ir::INST_GET_NUM_PROGRAMS,     {REPLICATED, {}}},
  // scalar/array
  { ir::INST_PHI,                  {ANY,         {ANY, ANY}}},
  { ir::INST_BINOP,                {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_GETELEMENTPTR,        {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_SELECT,               {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_SQRT,                 {DISTRIBUTED, {DISTRIBUTED}}},
  // cmp
  { ir::INST_ICMP,                 {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_FCMP,                 {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED}}},
  // cast
  { ir::INST_CAST_TRUNC,           {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_ZEXT,            {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_SEXT,            {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_FP_TRUNC,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_FP_EXT,          {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_UI_TO_FP,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_SI_TO_FP,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_FP_TO_UI,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_FP_TO_SI,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_PTR_TO_INT,      {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_INT_TO_PTR,      {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_BIT_CAST,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_CAST_ADDR_SPACE_CAST, {DISTRIBUTED, {DISTRIBUTED}}},
  // io
  { ir::INST_UNMASKED_LOAD,        {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_MASKED_LOAD,          {DISTRIBUTED, {DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_UNMASKED_STORE,       {NONE       , {DISTRIBUTED, DISTRIBUTED}}},
  { ir::INST_MASKED_STORE,         {NONE       , {DISTRIBUTED, DISTRIBUTED, DISTRIBUTED}}},
  // retile
  { ir::INST_RESHAPE,              {DISTRIBUTED, {DISTRIBUTED}}},
  { ir::INST_SPLAT,                {DISTRIBUTED, {REPLICATED}}},
  { ir::INST_BROADCAST,            {DISTRIBUTED, {REPLICATED}}},
  { ir::INST_DOWNCAST,             {DISTRIBUTED, {REPLICATED}}},
  // array arithmetic
  { ir::INST_TRANS,                {SHARED,      {SHARED}}},
  { ir::INST_REDUCE,               {SHARED,      {DISTRIBUTED}}},
  { ir::INST_DOT,                  {DISTRIBUTED, {SHARED, SHARED, DISTRIBUTED}}},
  // terminator
  { ir::INST_RETURN,               {NONE,        {}}},
  { ir::INST_UNCOND_BRANCH,        {NONE,        {}}},
  { ir::INST_COND_BRANCH,          {NONE,        {REPLICATED}}},

  // intrinsics
  { ir::INST_COPY_TO_SHARED,       {SHARED,      {DISTRIBUTED}}},
  { ir::INST_BARRIER,              {NONE,        {}}},
  { ir::INST_MAKE_RANGE_DYN,       {DISTRIBUTED, {}}},
  { ir::INST_MAKE_RANGE_STA,       {DISTRIBUTED, {}}},
  { ir::INST_MAKE_RANGE,           {DISTRIBUTED, {}}}
};

}
}

#endif

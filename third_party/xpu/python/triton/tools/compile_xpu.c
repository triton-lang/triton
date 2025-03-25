/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <xpu/runtime.h>

#define XPUdeviceptr int64_t

// helpers to check for xpu errors
#define XPU_CHECK(ans) {{\
    xpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void xpuAssert(int code, const char *file, int line) {{
  if (code != XPU_SUCCESS) {{
    const char *prefix = "Triton Error [XPU]: ";
    const char *str = xpu_strerror(code);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

static inline uint32_t checksum(const unsigned char *data, size_t length) {{
  uint32_t crc32 = 0;
  for (size_t i = 0; i < length; ++i)
    crc32 += static_cast<uint32_t>(data[i]);
  return crc32;
}}

static inline size_t alignSizeTo4Bytes(size_t size) {{
    return (size + 3) & ~3;
}}

static inline size_t alignSizeTo8Bytes(size_t size) {{
    return (size + 7) & ~7;
}}

static inline int min(int a, int b) {{
  return a < b ? a : b;
}}

// XPU Kernel type
enum kernel_type {{
  KT_CLUSTER = 0,
  KT_SDCDNN = 1,
}};

// Place of XPU kernel binary
enum kernel_place {{
  KP_CPU = 0,
  KP_XPU = 1,
}};

// XPU Kernel
struct xpu_kernel {{
  uint32_t type : 16;
  uint32_t place : 16;
  uint64_t code_addr;
  uint32_t code_byte_size;
  uint32_t code_pc;
  uint32_t param_dword_size;
  uint64_t hash;
  const char *name;
  void *rt_private;
  uint64_t printf_buffer_offset;
}};

static int __xpu_create_func(XPUFunc *pfunc, int type, uint64_t code_addr,
                             uint32_t code_bsz, uint32_t code_pc,
                             uint32_t param_dsz, uint64_t hash,
                             const char *name, bool on_xpu,
                             uint64_t printf_buf_offset) {{
  if (pfunc == NULL)
    return -XPUERR_INVALID_PARAM;

  struct xpu_kernel *kern = new struct xpu_kernel();
  kern->type = type;
  kern->place = (on_xpu) ? KP_XPU : KP_CPU;
  kern->code_addr = code_addr;
  kern->code_byte_size = code_bsz;
  kern->code_pc = code_pc;
  kern->param_dword_size = param_dsz;
  kern->hash = hash;
  kern->name = name;
  kern->printf_buffer_offset = printf_buf_offset;

  *pfunc = kern;

  return 0;
}}

// globals
#define XPUBIN_NAME {kernel_name}_xpubin
XPUFunc {kernel_name}_func = NULL;
unsigned char XPUBIN_NAME[{bin_size}] = {{ {bin_data} }};
void *{kernel_name}_ewt_gptr = NULL;
unsigned char {kernel_name}_ewt_data[] = {{ {ewt_data} }};

static inline void *loadEWTable() {{
  void *gmptr = {kernel_name}_ewt_gptr;
  if (gmptr) {{
    void *data = (void *)&{kernel_name}_ewt_data;
    size_t size = sizeof({kernel_name}_ewt_data);
    XPU_CHECK(xpu_malloc((void **)gmptr, size));
    XPU_CHECK(xpu_memcpy(gmptr, data, size, XPU_HOST_TO_DEVICE));
  }}
  return gmptr;
}}

void unload_{kernel_name}(void) {{
  XPU_CHECK(xpu_free({kernel_name}_ewt_gptr));
  {kernel_name}_ewt_gptr = NULL;
  delete (struct xpu_kernel *){kernel_name}_func;
  {kernel_name}_func = NULL;
}}

void load_{kernel_name}() {{
  void *bin = (void *)&XPUBIN_NAME;
  // Create XPUFunc
  int type = {kernel_type};
  uint64_t code_addr = reinterpret_cast<uint64_t>(bin);
  uint32_t code_byte_size = static_cast<uint32_t>({bin_size});
  uint32_t code_pc = 0;
  uint32_t param_dword_size = 0;
  uint32_t hash = checksum(XPUBIN_NAME, {bin_size});
  bool on_xpu = false;

  XPU_CHECK(__xpu_create_func(&{kernel_name}_func, type, code_addr,
                              code_byte_size, code_pc, param_dword_size, hash,
                              "{kernel_name}", on_xpu, {printf_buf_offset}));
}}

/*
{kernel_docstring}
*/
int {kernel_name}(XPUStream stream, {signature}) {{
  if ({kernel_name}_func == NULL)
      load_{kernel_name}();

  unsigned int gridX = {gridX};
  unsigned int gridY = {gridY};
  unsigned int gridZ = {gridZ};
  if(gridX * gridY * gridZ > 0) {{
    size_t offset = 0;
    {argument_set_code}
    {load_ewtable_code}
    XPU_CHECK(xpu_launch_argument_set(&gridX, sizeof(gridX), offset+0));
    XPU_CHECK(xpu_launch_argument_set(&gridY, sizeof(gridY), offset+4));
    XPU_CHECK(xpu_launch_argument_set(&gridZ, sizeof(gridZ), offset+8));
    XPU_CHECK(xpu_launch_config(min(gridX*gridY*gridZ, {nclusters}), {ncores}));
    XPU_CHECK(xpu_launch_async({kernel_name}_func));
    return 0;
  }}

  return -1;
}}

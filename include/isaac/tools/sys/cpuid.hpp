/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#include <limits.h>
#endif

namespace isaac
{
namespace tools
{
#ifdef _MSC_VER
	inline void cpuid(int code, int *a, int *b, int *c, int *d) {
		int regs[4];
		__cpuid((int *)regs, (int)code);
		*a = regs[0];
		*b = regs[1];
		*c = regs[2];
		*d = regs[3];
	}
#else
	inline void cpuid(int code, int *a, int *b, int *c, int *d) {
		__asm__ __volatile__("cpuid":"=a"(*a),"=b"(*b),
	                         "=c"(*c),"=d"(*d):"a"(code));
	}
#endif
inline std::string cpu_brand(){
  char name[48];
  int* ptr = (int*)name;
  cpuid(0x80000002, ptr, ptr+1, ptr+2, ptr+3);
  cpuid(0x80000003, ptr+4, ptr+5, ptr+6, ptr+7);
  cpuid(0x80000004, ptr+8, ptr+9, ptr+10, ptr+11);
  return std::string(name, name+48);
}



}
}

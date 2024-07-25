#include <cstdio>

void triton_assert(bool cond, char *c) {
  if (!cond)
    fprintf(stderr, "%s\n", c);
}

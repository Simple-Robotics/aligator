#include "aligator/core/mimalloc-resource.hpp"
#include <mimalloc.h>

namespace aligator {

void *mimalloc_resource ::do_allocate(size_t bytes, size_t alignment) {
  return mi_malloc_aligned(bytes, alignment);
}

void mimalloc_resource::do_deallocate(void *ptr, size_t, size_t alignment) {
  mi_free_aligned(ptr, alignment);
}

} // namespace aligator

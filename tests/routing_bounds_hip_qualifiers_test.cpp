// Exercise the HIP-only declaration branch with a host compiler. This catches
// unsupported qualifier spellings before setup reaches the ROCm build.
#define __HIPCC__ 1
#define __host__
#define __device__

#include "../routing/bounds.hpp"

int main() {
  const std::int32_t route_end_x[] = {3};
  const std::int32_t route_end_y[] = {7};
  routing::RoutingQueryBounds bounds;
  bounds.enabled = true;
  bounds.min_x = 0;
  bounds.max_x = 5;
  bounds.min_y = 0;
  bounds.max_y = 10;
  return routing::route_node_admitted(route_end_x, route_end_y, 0, bounds)
             ? 0
             : 1;
}

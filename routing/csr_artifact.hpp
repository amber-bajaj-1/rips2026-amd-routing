#pragma once

#include "../pre-process/import_policy.hpp"
#include "../pre-process/routing_csr_sidecars.hpp"
#include "../sssp/sssp_types.hpp"

#include <filesystem>
#include <optional>

namespace routing {

// Validate the complete host graph payload before it is exposed to routing or
// uploaded by a backend. This remains host-only so artifact tests do not need
// a HIP toolchain.
void validate_csr(const HostCsrF32& graph);

// Load the RIPSCSR1 v4 artifact emitted by interchange_to_csr. Edge weights
// are implicit unit values; the loader materializes them for the SSSP APIs.
HostCsrF32 load_csrbin(
    const std::filesystem::path& path,
    std::optional<interchange::InterchangeArtifactPairId>* artifact_pair_id =
        nullptr,
    interchange::RoutingCsrSidecars* routing_sidecars = nullptr);

}  // namespace routing

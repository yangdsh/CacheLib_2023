#include <stdio.h>

#include <cstring>
#include <string>

#include "cachelib/cachebench/runner/Stressor.h"
#include "rpc.h"

static const std::string kServerHostname = "128.110.219.167";
static const std::string kClientHostname = "128.110.219.156";

static constexpr uint8_t kReqType = 2;
static constexpr uint8_t kPhyPort = 2; // Physical port num of the desired NIC
static constexpr size_t kAppEvLoopMs = 1000;
static constexpr size_t kServerBasePort = 31850;
static constexpr size_t kNumBgThreads = 0;

// Request will ask for certain size response.
struct req_t {
  size_t resp_size;
};

// Per-thread server context
class ServerThreadContext {
 public:
  size_t thread_id_;
  erpc::Rpc<erpc::CTransport>* rpc_ = nullptr; // Store the rpc instance
  std::function<void()>& throttleFn;
  ThroughputStats& stats;

  ~ServerThreadContext() {}
};

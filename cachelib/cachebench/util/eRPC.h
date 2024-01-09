#pragma once

#include <stdio.h>

#include <cstring>
#include <string>

#include "cachelib/allocator/memory/Slab.h"
#include "cachelib/cachebench/runner/Stressor.h"
#include "cachelib/cachebench/util/Request.h"
#include "cachelib/cachebench/workload/GeneratorBase.h"
#include "rpc.h"

namespace facebook {
namespace cachelib {
namespace cachebench {

static const std::string kServerHostname = "128.110.219.167";
static const std::string kClientHostname = "128.110.219.156"; // client-0
// static const std::string kClientHostname = "128.110.219.157"; // client-1

static constexpr uint8_t kReqType = 2;
static constexpr uint8_t kPhyPort = 2; // Physical port num of the desired NIC
static constexpr size_t kAppEvLoopMs = 1000;
static constexpr size_t kServerBasePort = 31850;
static constexpr size_t kNumBgThreads = 0;

// Request metadata.
struct req_meta_t {
  OpType op;
  uint32_t ttl;
  std::optional<uint64_t> reqId;
  size_t key_size;
  size_t value_size;
};

// Request data.
struct req_data_t {
  std::string key;
  std::string value;
};

// Request will ask for certain size response.
struct req_t {
  struct req_meta_t meta;
  struct req_data_t data;
};

struct resp_t {
  OpResultType result;
  size_t data_size;
  // void* data;
  void* data;
};

// Per-thread server context
class ServerThreadContext {
 public:
  size_t thread_id_;
  erpc::Rpc<erpc::CTransport>* rpc_ = nullptr; // Store the rpc instance
  std::function<void()> throttleFn;
  Stressor* stressor;
  ThroughputStats* stats;
  PoolId pid;

  ~ServerThreadContext() {}
};

// Per-thread application context
class ClientThreadContext {
 public:
  size_t thread_id_;
  erpc::Rpc<erpc::CTransport>* rpc_ = nullptr; // Store the rpc instance
  int session_num = 0;                         // Session number

  GeneratorBase* gen;

  size_t num_sm_resps_ = 0; // Number of SM responses
  size_t num_resps_ = 0;    // Number of responses

  erpc::MsgBuffer req_msgbuf;
  erpc::MsgBuffer resp_msgbuf;

  ~ClientThreadContext() {}
};

} // namespace cachebench
} // namespace cachelib
} // namespace facebook

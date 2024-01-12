/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <folly/Random.h>
#include <folly/TokenBucket.h>
#include <folly/system/ThreadName.h>
#include <signal.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_set>

#ifdef BFADM
#include "cachelib/common/BloomFilter.h"
#endif

#include "cachelib/cachebench/cache/Cache.h"
#include "cachelib/cachebench/cache/TimeStampTicker.h"
#include "cachelib/cachebench/runner/Stressor.h"
#include "cachelib/cachebench/util/Config.h"
#include "cachelib/cachebench/util/Exceptions.h"
#include "cachelib/cachebench/util/Parallel.h"
#include "cachelib/cachebench/util/Request.h"
#include "cachelib/cachebench/util/eRPC.h"
#include "cachelib/cachebench/workload/GeneratorBase.h"

namespace facebook {
namespace cachelib {
namespace cachebench {

volatile sig_atomic_t stressor_ctrl_c_pressed = 0;
void stressor_ctrl_c_handler(int) { stressor_ctrl_c_pressed = 1; }

// Implementation of stressor that uses a workload generator to stress an
// instance of the cache.  All item's value in eRPCStressor follows CacheValue
// schema, which contains a few integers for sanity checks use. So it is invalid
// to use item.getMemory and item.getSize APIs.
template <typename Allocator>
class eRPCStressor : public Stressor {
 public:
  using CacheT = Cache<Allocator>;
  using Key = typename CacheT::Key;
  using WriteHandle = typename CacheT::WriteHandle;

  // @param cacheConfig   the config to instantiate the cache instance
  // @param config        stress test config
  // @param generator     workload  generator
  eRPCStressor(CacheConfig cacheConfig, StressorConfig config)
      : config_(std::move(config)),
        throughputStats_(config_.numThreads),
        hardcodedString_(genHardcodedString()),
        endTime_{std::chrono::system_clock::time_point::max()} {
    maxAllocSize = cacheConfig.maxAllocSize;
    //** useEvictionController = cacheConfig.useEvictionControl;
    useNVM = cacheConfig.nvmCacheSizeMB;
    cacheType = cacheConfig.allocator;

    // if either consistency check is enabled or if we want to move
    // items during slab release, we want readers and writers of chained
    // allocs to be synchronized
    typename CacheT::ChainedItemMovingSync movingSync;
    if (config_.usesChainedItems() &&
        (cacheConfig.moveOnSlabRelease || config_.checkConsistency)) {
      lockEnabled_ = true;

      struct CacheStressSyncObj : public CacheT::SyncObj {
        std::unique_lock<folly::SharedMutex> lock;

        CacheStressSyncObj(eRPCStressor& s, std::string itemKey)
            : lock{s.chainedItemAcquireUniqueLock(itemKey)} {}
      };
      movingSync = [this](typename CacheT::Item::Key key) {
        return std::make_unique<CacheStressSyncObj>(*this, key.str());
      };
    }

    if (cacheConfig.useTraceTimeStamp &&
        cacheConfig.tickerSynchingSeconds > 0) {
      // When using trace based replay for generating the workload,
      // TimeStampTicker allows syncing the notion of time between the
      // cache and the workload generator based on timestamps in the trace.
      ticker_ = std::make_shared<TimeStampTicker>(
          config.numThreads, cacheConfig.tickerSynchingSeconds, nullptr);
      cacheConfig.ticker = ticker_;
    }
    cacheConfig.nvmWriteBytesCallback =
        std::bind(&eRPCStressor<Allocator>::getNvmBytesWritten, this);
    try {
      cache_ = std::make_unique<CacheT>(
          cacheConfig, movingSync, cacheConfig.cacheDir, config_.touchValue);
#ifdef BFADM
      bf_ = std::make_unique<BloomFilter>(2, 4, bf_size_);
#endif
    } catch (const std::exception& e) {
      XLOG(INFO) << "Exception while creating cache: " << e.what();
      throw;
    }

    if (config_.opPoolDistribution.size() > cache_->numPools()) {
      throw std::invalid_argument(folly::sformat(
          "more pools specified in the test than in the cache. "
          "test: {}, cache: {}",
          config_.opPoolDistribution.size(), cache_->numPools()));
    }
    if (config_.keyPoolDistribution.size() != cache_->numPools()) {
      throw std::invalid_argument(folly::sformat(
          "different number of pools in the test from in the cache. "
          "test: {}, cache: {}",
          config_.keyPoolDistribution.size(), cache_->numPools()));
    }

    if (config_.opRatePerSec > 0) {
      // opRateBurstSize is default to opRatePerSec if not specified
      rateLimiter_ = std::make_unique<folly::BasicTokenBucket<>>(
          config_.opRatePerSec, config_.opRateBurstSize > 0
                                    ? config_.opRateBurstSize
                                    : config_.opRatePerSec);
    }
  }

  ~eRPCStressor() override { finish(); }

  static void req_handler(erpc::ReqHandle* req_handle, void* _context) {
    auto* c = static_cast<ServerThreadContext*>(_context);
    eRPCStressor* stressor = dynamic_cast<eRPCStressor*>(c->stressor);

    // Check Cache health status.
    if (stressor->cache_->getInconsistencyCount() >=
            stressor->config_.maxInconsistencyCount ||
        stressor->cache_->getInvalidDestructorCount() >=
            stressor->config_.maxInvalidDestructorCount ||
        stressor->cache_->isNvmCacheDisabled() || stressor->shouldTestStop()) {
      std::terminate();
    }

    // Get request buffer.
    const erpc::MsgBuffer* req_msgbuf = req_handle->get_req_msgbuf();

    // Read request metadata from request buffer.
    req_meta_t meta;
    memcpy(&meta, req_msgbuf->buf_, sizeof(req_meta_t));

    // Read the actual request from request buffer.
    std::string key(req_msgbuf->buf_ + sizeof(req_meta_t),
                    req_msgbuf->buf_ + sizeof(req_meta_t) + meta.key_size);
    std::string value(req_msgbuf->buf_ + sizeof(req_meta_t) + meta.key_size,
                      req_msgbuf->buf_ + sizeof(req_meta_t) + meta.key_size +
                          meta.value_size);
    std::unordered_map<std::string, std::string> admFeatureM;

    std::vector<size_t> sizes;
    sizes.push_back(meta.value_size);

    Request request(key, sizes.begin(), sizes.end(), meta.op, meta.ttl,
                    meta.reqId.value_or(0), admFeatureM, value);

    // Process request as per stressByDiscreteDistribution.
    resp_t resp;
    resp.result = OpResultType::kNop;
    resp.reqId = request.requestId;
    resp.data = nullptr;
    resp.data_size = 0;
    stressor->stressByDiscreteDistribution(request, *c, &resp);

    // Use dynamic response based on the size of the data from Cache.
    erpc::MsgBuffer& resp_msgbuf = req_handle->dyn_resp_msgbuf_;
    resp_msgbuf = c->rpc_->alloc_msg_buffer_or_die(
        sizeof(OpResultType) + sizeof(size_t) + resp.data_size);

    // resp.data_size describes the number of bytes in resp as it was given
    // to us by the cache. However, on the client size, we also expect `size`
    // number of bytes and include that in the request as req.sizeBegin. Hence
    // if req.sizeBegin < resp.data_size, then only return req.sizeBegin bytes.
    // resp.data_size is zero if not GET. If GET, limit to what the user
    // expects.
    if (*(request.sizeBegin) < resp.data_size) {
      resp.data_size = *(request.sizeBegin);
    }

    // Write a sequence to buffer.
    memcpy(resp_msgbuf.buf_, &resp.result, sizeof(OpResultType));
    memcpy(resp_msgbuf.buf_ + sizeof(OpResultType), &resp.reqId,
           sizeof(uint64_t));
    memcpy(resp_msgbuf.buf_ + sizeof(OpResultType) + sizeof(uint64_t),
           &resp.data_size, sizeof(size_t));
    memcpy(resp_msgbuf.buf_ + sizeof(OpResultType) + sizeof(uint64_t) +
               sizeof(size_t),
           resp.data, resp.data_size);
    c->rpc_->enqueue_response(req_handle, &resp_msgbuf);
  }

  void server_thread(size_t thread_id) {
    ServerThreadContext c;
    c.thread_id_ = thread_id;
    c.stressor = this;

    // Pid.
    std::mt19937_64 gen(folly::Random::rand64());
    std::discrete_distribution<> opPoolDist(config_.opPoolDistribution.begin(),
                                            config_.opPoolDistribution.end());
    const auto pid = static_cast<PoolId>(opPoolDist(gen));
    c.pid = pid;

    // Throttle function.
    const uint64_t opDelayBatch = config_.opDelayBatch;
    const uint64_t opDelayNs = config_.opDelayNs;
    const std::chrono::nanoseconds opDelay(opDelayNs);
    const bool needDelay = opDelayBatch != 0 && opDelayNs != 0;
    uint64_t opCounter = 0;
    auto throttleFn = [&] {
      if (needDelay && ++opCounter == opDelayBatch) {
        opCounter = 0;
        std::this_thread::sleep_for(opDelay);
      }
      // Limit the rate if specified.
      limitRate();
    };
    c.throttleFn = throttleFn;

    // Throughput stats from object.
    c.stats = &throughputStats_.at(thread_id);

    // Start eRPC server on a unique port.
    size_t port = kServerBasePort + thread_id;
    std::string server_uri = kServerHostname + ":" + std::to_string(port);
    erpc::Nexus nexus(server_uri, 0, kNumBgThreads);
    nexus.register_req_func(kReqType, req_handler);
    erpc::Rpc<erpc::CTransport> rpc(&nexus, static_cast<void*>(&c),
                                    static_cast<uint8_t>(thread_id), nullptr,
                                    kPhyPort);
    c.rpc_ = &rpc;

    printf("Server starting on port %zu...\n", port);
    while (true) {
      rpc.run_event_loop(kAppEvLoopMs);
      if (stressor_ctrl_c_pressed == 1)
        break;
    }
  }

  // Start the stress test by spawning the worker threads and waiting for them
  // to finish the stress operations.
  void start() override {
    {
      std::lock_guard<std::mutex> l(timeMutex_);
      startTime_ = std::chrono::system_clock::now();
    }
    std::cout << folly::sformat("Total {:.2f}M ops to be run",
                                config_.numThreads * config_.numOps / 1e6)
              << std::endl;

    signal(SIGINT, stressor_ctrl_c_handler);
    stressWorker_ = std::thread([this] {
      std::vector<std::thread> workers;

      for (uint64_t i = 0; i < config_.numThreads; ++i) {
        workers.push_back(std::thread(
            [this, throughputStats = &throughputStats_.at(i),
             threadName = folly::sformat("cb_stressor_{}", i), i]() {
              folly::setThreadName(threadName);
              server_thread(i);
            }));
        // erpc::bind_to_core(workers[i], 0, i);
      }
      for (auto& worker : workers) {
        worker.join();
      }
      {
        std::lock_guard<std::mutex> l(timeMutex_);
        endTime_ = std::chrono::system_clock::now();
      }
    });
  }

  double getNvmBytesWritten() {
    double bytesWritten = 0;
    if (cache_) {
      bytesWritten = cache_->getNvmBytesWritten();
      XLOG_EVERY_MS(INFO, 60000) << "NVM bytes written: " << bytesWritten;
    } else {
      XLOG_EVERY_MS(INFO, 60000) << "Error, allocator not set";
    }
    return bytesWritten;
  }

  // Block until all stress workers are finished.
  void finish() override {
    if (stressWorker_.joinable()) {
      stressWorker_.join();
    }
    cache_->clearCache(config_.maxInvalidDestructorCount);
  }

  // abort the stress run by indicating to the workload generator and
  // delegating to the base class abort() to stop the test.
  void abort() override { Stressor::abort(); }

  // obtain stats from the cache instance.
  Stats getCacheStats() const override { return cache_->getStats(); }

  // obtain aggregated throughput stats for the stress run so far.
  ThroughputStats aggregateThroughputStats() const override {
    ThroughputStats res{};
    for (const auto& stats : throughputStats_) {
      res += stats;
    }

    return res;
  }

  void renderWorkloadGeneratorStats(uint64_t elapsedTimeNs,
                                    std::ostream& out) const override {
    // This should be done by the generator client itself.
    // wg_->renderStats(elapsedTimeNs, out);
  }

  void renderWorkloadGeneratorStats(
      uint64_t elapsedTimeNs, folly::UserCounters& counters) const override {
    // This should be done by the generator client itself.
    // wg_->renderStats(elapsedTimeNs, counters);
  }

  uint64_t getTestDurationNs() const override {
    std::lock_guard<std::mutex> l(timeMutex_);
    return std::chrono::nanoseconds{
        std::min(std::chrono::system_clock::now(), endTime_) - startTime_}
        .count();
  }

 private:
  static std::string genHardcodedString() {
    const std::string s = "The quick brown fox jumps over the lazy dog. ";
    std::string val;
    for (int i = 0; i < 4 * 1024 * 1024; i += s.size()) {
      val += s;
    }
    return val;
  }

  folly::SharedMutex& getLock(Key key) {
    auto bucket = MurmurHash2{}(key.data(), key.size()) % locks_.size();
    return locks_[bucket];
  }

  // TODO maintain state on whether key has chained allocs and use it to only
  // lock for keys with chained items.
  auto chainedItemAcquireSharedLock(Key key) {
    using Lock = std::shared_lock<folly::SharedMutex>;
    return lockEnabled_ ? Lock{getLock(key)} : Lock{};
  }
  auto chainedItemAcquireUniqueLock(Key key) {
    using Lock = std::unique_lock<folly::SharedMutex>;
    return lockEnabled_ ? Lock{getLock(key)} : Lock{};
  }

  // populate the input item handle according to the stress setup.
  void populateItem(WriteHandle& handle, const std::string& itemValue = "") {
    if (!config_.populateItem) {
      return;
    }
    XDCHECK(handle);
    XDCHECK_LE(cache_->getSize(handle), 4ULL * 1024 * 1024);
    if (cache_->consistencyCheckEnabled()) {
      cache_->setUint64ToItem(handle, folly::Random::rand64(rng));
    }

    if (!itemValue.empty()) {
      cache_->setStringItem(handle, itemValue);
    } else {
      cache_->setStringItem(handle, hardcodedString_);
    }
  }

  // Runs a number of operations on the cache allocator. The actual
  // operations and key/value used are determined by the workload generator
  // initialized.
  //
  // Throughput and Hit/Miss rates are tracked here as well
  //
  // @param stats       Throughput stats
  void stressByDiscreteDistribution(const Request& req,
                                    ServerThreadContext& c,
                                    resp_t* resp) {
    try {
      auto& stats = *c.stats;
      auto pid = c.pid;
      // at the end of every operation, throttle per the config.
      SCOPE_EXIT { c.throttleFn(); };
        // detect refcount leaks when run in  debug mode.
#ifndef NDEBUG
      auto checkCnt = [](int cnt) {
        if (cnt != 0) {
          throw std::runtime_error(folly::sformat("Refcount leak {}", cnt));
        }
      };
      checkCnt(cache_->getHandleCountForThread());
      SCOPE_EXIT { checkCnt(cache_->getHandleCountForThread()); };
#endif
      ++stats.ops;

      if (*(req.sizeBegin) > 8 && !useEvictionController) {
        *(req.sizeBegin) -= 8;
      }
      if (cacheType == "TinyLFU") {
        *(req.sizeBegin) += 24;
      }
      if (cacheType == "S3FIFO") {
        *(req.sizeBegin) += 8;
      }

      // filter size larger than 4mb
      if (*(req.sizeBegin) >= maxAllocSize) {
        // lastRequestId = req.requestId;
        if (req.requestId) {
          // req might be deleted after calling notifyResult()
          resp->result = OpResultType::kGetMiss;
        }
        return;
      }

      OpType op = req.getOp();
      const std::string* key = &(req.key);
      std::string oneHitKey;
      if (op == OpType::kLoneGet || op == OpType::kLoneSet) {
        oneHitKey = Request::getUniqueKey();
        key = &oneHitKey;
      }
      if (op == OpType::kDel) {
        op = OpType::kGet;
      }

      util::LatencyTracker tracker =
          util::LatencyTracker(cache_->cacheRequestLatency_);
      switch (op) {
      case OpType::kLoneSet:
      case OpType::kSet: {
        if (config_.onlySetIfMiss) {
          auto it = cache_->find(*key);
          if (it != nullptr) {
            return;
          }
        }
        auto lock = chainedItemAcquireUniqueLock(*key);
        resp->result = setKey(pid, stats, key, *(req.sizeBegin), req.ttlSecs,
                              req.nextTime, req.admFeatureMap, req.itemValue);

        break;
      }
      case OpType::kLoneGet:
      case OpType::kGet: {
        ++stats.get;
        auto slock = chainedItemAcquireSharedLock(*key);
        auto xlock = decltype(chainedItemAcquireUniqueLock(*key)){};

        if (ticker_) {
          ticker_->updateTimeStamp(req.timestamp);
        }
        // TODO currently pure lookaside, we should
        // add a distribution over sequences of requests/access patterns
        // e.g. get-no-set and set-no-get
        cache_->recordAccess(*key);
        auto it = cache_->find(*key);
        if (it == nullptr) {
          ++stats.getMiss;
          resp->result = OpResultType::kGetMiss;

          if (config_.enableLookaside) {
            // allocate and insert on miss
            // upgrade access privledges, (lock_upgrade is not
            // appropriate here)
            slock = {};
            xlock = chainedItemAcquireUniqueLock(*key);
            setKey(pid, stats, key, *(req.sizeBegin), req.ttlSecs, req.nextTime,
                   req.admFeatureMap, req.itemValue);
          }
        } //** else if (it->get_is_reinserted()) {
          // from NVM cache
          //** result = OpResultType::kGetMiss;
        //** }
        else {
#ifdef TRUE_TTA
          it->next_timestamp = req.nextTime;
#endif
          resp->result = OpResultType::kGetHit;
          resp->data = const_cast<void*>(it->getMemory());
          resp->data_size = it->getSize();
        }
        break;
      }
      case OpType::kDel: {
        ++stats.del;
        auto lock = chainedItemAcquireUniqueLock(*key);
        auto res = cache_->remove(*key);
        if (res == CacheT::RemoveRes::kNotFoundInRam) {
          ++stats.delNotFound;
        }
        break;
      }
      case OpType::kAddChained: {
        ++stats.get;
        auto lock = chainedItemAcquireUniqueLock(*key);
        auto it = cache_->findToWrite(*key);
        if (!it) {
          ++stats.getMiss;

          ++stats.set;
          it = cache_->allocate(pid, *key, *(req.sizeBegin), req.ttlSecs);
          if (!it) {
            ++stats.setFailure;
            break;
          }
          populateItem(it);
          cache_->insertOrReplace(it);
        }
        XDCHECK(req.sizeBegin + 1 != req.sizeEnd);
        bool chainSuccessful = false;
        for (auto j = req.sizeBegin + 1; j != req.sizeEnd; j++) {
          ++stats.addChained;

          const auto size = *j;
          auto child = cache_->allocateChainedItem(it, size);
          if (!child) {
            ++stats.addChainedFailure;
            continue;
          }
          chainSuccessful = true;
          populateItem(child);
          cache_->addChainedItem(it, std::move(child));
        }
        if (chainSuccessful && cache_->consistencyCheckEnabled()) {
          cache_->trackChainChecksum(it);
        }
        break;
      }
      case OpType::kUpdate: {
        ++stats.get;
        ++stats.update;
        auto lock = chainedItemAcquireUniqueLock(*key);
        if (ticker_) {
          ticker_->updateTimeStamp(req.timestamp);
        }
        auto it = cache_->findToWrite(*key);
        if (it == nullptr) {
          ++stats.getMiss;
          ++stats.updateMiss;
          break;
        }
        cache_->updateItemRecordVersion(it);
        break;
      }
      case OpType::kCouldExist: {
        ++stats.couldExistOp;
        if (!cache_->couldExist(*key)) {
          ++stats.couldExistOpFalse;
        }
        break;
      }
      default:
        throw std::runtime_error(
            folly::sformat("invalid operation generated: {}", (int)op));
        break;
      }
    } catch (const cachebench::EndOfTrace& ex) {
      std::terminate();
    }
  }

  // inserts key into the cache if the admission policy also indicates the
  // key is worthy to be cached.
  //
  // @param pid         pool id to insert the key
  // @param stats       reference to the stats structure.
  // @param key         the key to be inserted
  // @param size        size of the cache value
  // @param ttlSecs     ttl for the value
  // @param featureMap  feature map for admission policy decisions.
  OpResultType setKey(
      PoolId pid,
      ThroughputStats& stats,
      const std::string* key,
      size_t size,
      uint32_t ttlSecs,
      uint32_t nextTime,
      const std::unordered_map<std::string, std::string>& featureMap,
      const std::string& itemValue) {
    // check the admission policy first, and skip the set operation
    // if the policy returns false
    if (config_.admPolicy && !config_.admPolicy->accept(featureMap)) {
      return OpResultType::kSetSkip;
    }
#ifdef BFADM
    if (!bf_->couldExist(0, hasher_(*key)) &&
        !bf_->couldExist(1, hasher_(*key))) {
      bf_->set(bf_insert_id_, hasher_(*key));
      bf_insert_cnt_ += 1;
      if (bf_insert_cnt_ * 4 > bf_size_) {
        std::lock_guard<std::mutex> guard(bf_mutex_);
        if (bf_insert_cnt_ * 4 > bf_size_) {
          bf_->clear(1 - bf_insert_id_);
          bf_insert_id_ = 1 - bf_insert_id_;
          bf_insert_cnt_ = 0;
          std::cout << "reset adm Bloom Filter" << std::endl;
        }
      }
      return OpResultType::kSetSkip;
    }
#endif

    ++stats.set;
    auto it = cache_->allocate(pid, *key, size, ttlSecs);
    if (it == nullptr) {
      ++stats.setFailure;
      return OpResultType::kSetFailure;
    } else {
      populateItem(it, itemValue);
      cache_->insertOrReplace(it);
#ifdef TRUE_TTA
      it->next_timestamp = nextTime;
#endif
      return OpResultType::kSetSuccess;
    }
  }

  void limitRate() {
    if (!rateLimiter_) {
      return;
    }
    rateLimiter_->consumeWithBorrowAndWait(1);
  }

  void checkNvmCacheWarmedUp(uint64_t requestTimestamp) {
    if (hasNvmCacheWarmedUp_) {
      // already notified, nothing to do
      return;
    }
    if (cache_->isNvmCacheDisabled()) {
      return;
    }
    if (cache_->hasNvmCacheWarmedUp()) {
      // wg_->setNvmCacheWarmedUp(requestTimestamp); // this is a do nothing for
      // KVReplayGenerator
      XLOG(INFO) << "NVM cache has been warmed up";
      hasNvmCacheWarmedUp_ = true;
    }
  }

  const StressorConfig config_; // config for the stress run

  std::string cacheType;

  std::vector<ThroughputStats> throughputStats_; // thread local stats

  // locks when using chained item and moving.
  std::array<folly::SharedMutex, 1024> locks_;

  // if locking is enabled.
  std::atomic<bool> lockEnabled_{false};

  std::atomic<uint64_t> total_ops{0};

  Stats prev_stat;

  // memorize rng to improve random performance
  folly::ThreadLocalPRNG rng;

  // string used for generating random payloads
  const std::string hardcodedString_;

  std::unique_ptr<CacheT> cache_;

  bool useEvictionController = 0;
  bool useNVM = 0;
  uint64_t avgSize = 0;
  uint64_t maxAllocSize = 1024 * 1024;

#ifdef BFADM
  std::unique_ptr<BloomFilter> bf_;
  std::hash<std::string> hasher_;
  uint32_t bf_insert_cnt_ = 0;
  uint32_t bf_insert_id_ = 0;
  uint32_t bf_size_ = 10000000;
  std::mutex bf_mutex_;
#endif

  // Ticker that syncs the time according to trace timestamp.
  std::shared_ptr<TimeStampTicker> ticker_;

  // main stressor thread
  std::thread stressWorker_;

  // mutex to protect reading the timestamps.
  mutable std::mutex timeMutex_;

  // start time for the stress test
  std::chrono::time_point<std::chrono::system_clock> startTime_;

  // time when benchmark finished. This is set once the benchmark finishes
  std::chrono::time_point<std::chrono::system_clock> endTime_;

  // Token bucket used to limit the operations per second.
  std::unique_ptr<folly::BasicTokenBucket<>> rateLimiter_;

  // Whether flash cache has been warmed up
  bool hasNvmCacheWarmedUp_{false};
};
} // namespace cachebench
} // namespace cachelib
} // namespace facebook

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

#include <folly/Conv.h>
#include <folly/ProducerConsumerQueue.h>
#include <folly/ThreadLocal.h>
#include <folly/lang/Aligned.h>
#include <folly/logging/xlog.h>
#include <folly/system/ThreadName.h>

#include "cachelib/cachebench/cache/Cache.h"
#include "cachelib/cachebench/util/Exceptions.h"
#include "cachelib/cachebench/util/Parallel.h"
#include "cachelib/cachebench/util/Request.h"
#include "cachelib/cachebench/workload/ReplayGeneratorBase.h"

namespace facebook {
namespace cachelib {
namespace cachebench {

struct ReqWrapper {
  ReqWrapper() = default;

  ReqWrapper(const ReqWrapper& other)
      : key_(other.key_),
        sizes_(other.sizes_),
        req_(key_,
             sizes_.begin(),
             sizes_.end(),
             reinterpret_cast<uint64_t>(this),
             other.req_),
        repeats_(other.repeats_) {
#ifdef TRUE_TTA
          req_.nextTime = other.req_.nextTime;
#endif
        }

  // current outstanding key
  std::string key_;
  std::vector<size_t> sizes_{1};
  // current outstanding req object
  // Use 'this' as the request ID, so that this object can be
  // identified on completion (i.e., notifyResult call)
  Request req_{key_, sizes_.begin(), sizes_.end(), OpType::kGet,
               reinterpret_cast<uint64_t>(this)};

  // number of times to issue the current req object
  // before fetching a new line from the trace
  uint32_t repeats_{0};
};

// KVReplayGenerator generates the cachelib requests based the trace data
// read from the given trace file(s).
// KVReplayGenerator supports amplifying the key population by appending
// suffixes (i.e., stream ID) to each key read from the trace file.
// In order to minimize the contentions for the request submission queues
// which might need to be dispatched by multiple stressor threads,
// the requests are sharded to each stressor by doing hashing over the key.
class KVReplayGenerator : public ReplayGeneratorBase {
 public:
  // Default order is key,op,size,op_count,key_size,ttl
  enum SampleFields : uint8_t {
    KEY = 0,
    OP,
    SIZE,
    OP_COUNT,
    KEY_SIZE,
    TTL,
    OP_TIME,
    CACHE_HIT,
    NEXT,
    END
  };

  const ColumnTable columnTable_ = {
      {SampleFields::OP_TIME, false, {"op_time"}},
      {SampleFields::KEY, true, {"key"}}, /* required */
      {SampleFields::KEY_SIZE, false, {"key_size"}},
      {SampleFields::OP, false, {"op"}}, /* required */
      {SampleFields::OP_COUNT, false, {"op_count"}},
      {SampleFields::SIZE, true, {"size"}}, /* required */
      {SampleFields::CACHE_HIT, false, {"cache_hits"}},
      {SampleFields::NEXT, false, {"next"}},
      {SampleFields::TTL, false, {"ttl"}}};

  explicit KVReplayGenerator(const StressorConfig& config)
      : ReplayGeneratorBase(config), traceStream_(config, 0, columnTable_) {
    for (uint32_t i = 0; i < numShards_; ++i) {
      stressorCtxs_.emplace_back(std::make_unique<StressorCtx>(i));
    }

    genWorker_ = std::thread([this] {
      folly::setThreadName("cb_replay_gen");
      genRequests();
    });

    XLOGF(INFO,
          "Started KVReplayGenerator (amp factor {}, # of stressor threads {})",
          ampFactor_, numShards_);
  }

  virtual ~KVReplayGenerator() {
    XCHECK(shouldShutdown());
    if (genWorker_.joinable()) {
      genWorker_.join();
    }
  }

  // getReq generates the next request from the trace file.
  const Request& getReq(
      uint8_t,
      std::mt19937_64&,
      std::optional<uint64_t> lastRequestId = std::nullopt) override;

  void renderStats(uint64_t, std::ostream& out) const override {
    out << std::endl << "== KVReplayGenerator Stats ==" << std::endl;

    out << folly::sformat("{}: {:.2f} million (parse error: {})",
                          "Total Processed Samples",
                          (double)parseSuccess.load() / 1e6, parseError.load())
        << std::endl;
  }

  void notifyResult(uint64_t requestId, OpResultType result) override;

  void markFinish() override { getStressorCtx().markFinish(); }

  // Parse the request from the trace line and set the ReqWrapper
  bool parseRequest(const std::string& line, std::unique_ptr<ReqWrapper>& req);

  // for unit test
  bool setHeaderRow(const std::string& header) {
    return traceStream_.setHeaderRow(header);
  }

 private:
  // Interval at which the submission queue is polled when it is either
  // full (producer) or empty (consumer).
  // We use polling with the delay since the ProducerConsumerQueue does not
  // support blocking read or writes with a timeout
  static constexpr uint64_t checkIntervalUs_ = 1;
  static constexpr size_t kMaxRequests = 10000;

  using ReqQueue = folly::ProducerConsumerQueue<std::unique_ptr<ReqWrapper>>;

  // StressorCtx keeps track of the state including the submission queues
  // per stressor thread. Since there is only one request generator thread,
  // lock-free ProducerConsumerQueue is used for performance reason.
  // Also, separate queue which is dispatched ahead of any requests in the
  // submission queue is used for keeping track of the requests which need to be
  // resubmitted (i.e., a request having remaining repeat count); there could
  // be more than one requests outstanding for async stressor while only one
  // can be outstanding for sync stressor
  struct StressorCtx {
    explicit StressorCtx(uint32_t id)
        : id_(id), reqQueue_(std::in_place_t{}, kMaxRequests) {}

    bool isFinished() { return finished_.load(std::memory_order_relaxed); }
    void markFinish() { finished_.store(true, std::memory_order_relaxed); }

    uint32_t id_{0};
    std::queue<std::unique_ptr<ReqWrapper>> resubmitQueue_;
    folly::cacheline_aligned<ReqQueue> reqQueue_;
    // Thread that finish its operations mark it here, so we will skip
    // further request on its shard
    std::atomic<bool> finished_{false};
  };

  // Read next trace line from TraceFileStream and fill ReqWrapper
  std::unique_ptr<ReqWrapper> getReqInternal();

  // Used to assign stressorIdx_
  std::atomic<uint32_t> incrementalIdx_{0};

  // A sticky index assigned to each stressor threads that calls into
  // the generator.
  folly::ThreadLocalPtr<uint32_t> stressorIdx_;

  // Vector size is equal to the # of stressor threads;
  // stressorIdx_ is used to index.
  std::vector<std::unique_ptr<StressorCtx>> stressorCtxs_;

  TraceFileStream traceStream_;

  std::thread genWorker_;

  // Used to signal end of file as EndOfTrace exception
  std::atomic<bool> eof{false};

  // Stats
  std::atomic<uint64_t> parseError = 0;
  std::atomic<uint64_t> parseSuccess = 0;

  void genRequests();

  void setEOF() { eof.store(true, std::memory_order_relaxed); }
  bool isEOF() { return eof.load(std::memory_order_relaxed); }

  inline StressorCtx& getStressorCtx(size_t shardId) {
    XCHECK_LT(shardId, numShards_);
    return *stressorCtxs_[shardId];
  }

  inline StressorCtx& getStressorCtx() {
    if (!stressorIdx_.get()) {
      stressorIdx_.reset(new uint32_t(incrementalIdx_++));
    }

    return getStressorCtx(*stressorIdx_);
  }
};

inline bool KVReplayGenerator::parseRequest(const std::string& line,
                                            std::unique_ptr<ReqWrapper>& req) {
  if (!traceStream_.setNextLine(line)) {
    return false;
  }

  auto sizeField = traceStream_.template getField<size_t>(SampleFields::SIZE);
  if (!sizeField.hasValue()) {
    return false;
  }

  // Set key
  req->key_ = traceStream_.template getField<>(SampleFields::KEY).value();

  auto keySizeField =
      traceStream_.template getField<size_t>(SampleFields::KEY_SIZE);
  if (keySizeField.hasValue()) {
    // The key is encoded as <encoded key, key size>.
    // Generate key whose size matches with that of the original one
    size_t keySize = std::max<size_t>(keySizeField.value(), req->key_.size());
    // The key size should not exceed 256
    keySize = std::min<size_t>(keySize, 256);
    req->key_.resize(keySize, '0');
  }

  // Convert timestamp to seconds.
  auto timestampField =
      traceStream_.template getField<uint64_t>(SampleFields::OP_TIME);
  if (timestampField.hasValue()) {
    uint64_t timestampRaw = timestampField.value();
    uint64_t timestampSeconds = timestampRaw / timestampFactor_;
    req->req_.timestamp = timestampSeconds;
  }

  // Set op
  auto op_optional = traceStream_.template getField<>(SampleFields::OP);
  if (op_optional.has_value()) {
    auto op = op_optional.value();
    // TODO implement GET_LEASE and SET_LEASE emulations
    if (!op.compare("GET") || !op.compare("GET_LEASE")) {
      req->req_.setOp(OpType::kGet);
    } else if (!op.compare("SET") || !op.compare("SET_LEASE")) {
      req->req_.setOp(OpType::kSet);
    } else if (!op.compare("DELETE")) {
      req->req_.setOp(OpType::kDel);
    } else {
      return false;
    }
  } else {
    req->req_.setOp(OpType::kGet);
  }

  // Set size
  req->sizes_[0] = sizeField.value();

  // Set op_count
  auto opCountField =
      traceStream_.template getField<uint32_t>(SampleFields::OP_COUNT);
  req->repeats_ = opCountField.value_or(1);
  if (!req->repeats_) {
    return false;
  }
  if (config_.ignoreOpCount) {
    req->repeats_ = 1;
  }

  // Set TTL (optional)
  auto ttlField = traceStream_.template getField<size_t>(SampleFields::TTL);
  req->req_.ttlSecs = ttlField.value_or(0);
  auto nextTimeField = traceStream_.template getField<size_t>(SampleFields::NEXT);
  req->req_.nextTime = nextTimeField.value_or(0);
  if (config_.admissionThreshold != 0 &&
      req->req_.nextTime > parseError + parseSuccess + config_.admissionThreshold)
    return false;
  // XLOG_EVERY_MS(INFO, 1000) << "parse success and error: " << parseSuccess << ' ' << parseError;
  return true;
}

inline std::unique_ptr<ReqWrapper> KVReplayGenerator::getReqInternal() {
  auto reqWrapper = std::make_unique<ReqWrapper>();
  do {
    std::string line;
    traceStream_.getline(line); // can throw

    if (!parseRequest(line, reqWrapper)) {
      parseError++;
      //XLOG_N_PER_MS(ERR, 10, 1000) << folly::sformat(
      //    "Parsing error (total {}): {}", parseError.load(), line);
      reqWrapper->repeats_ = 0;
    } else {
      parseSuccess++;
    }
  } while (reqWrapper->repeats_ == 0);

  return reqWrapper;
}

uint64_t cnt = 0;

inline void KVReplayGenerator::genRequests() {
  while (!shouldShutdown()) {
    std::unique_ptr<ReqWrapper> reqWrapper;
    try {
      reqWrapper = getReqInternal();
    } catch (const EndOfTrace& e) {
      break;
    }

    std::unique_ptr<ReqWrapper> req;
    req.swap(reqWrapper);
    auto shardId = getShard(req->req_.key); //, req->key_.size() + req->sizes_[0]);
    while(1) {
      auto& stressorCtx = getStressorCtx(shardId);
      auto& reqQ = *stressorCtx.reqQueue_;

      if (!reqQ.write(std::move(req)) && !stressorCtx.isFinished() &&
             !shouldShutdown()) {
        // ProducerConsumerQueue does not support blocking, so use sleep
        std::this_thread::sleep_for(
            std::chrono::microseconds{checkIntervalUs_});
        // shardId = (shardId + 1) % config_.numThreads;
      } else break;
    }
  }

  setEOF();
}

thread_local int keySuffixLocal = 100;
thread_local std::unique_ptr<ReqWrapper> curReqWrapper;

const Request& KVReplayGenerator::getReq(uint8_t,
                                         std::mt19937_64&,
                                         std::optional<uint64_t>) {
  std::unique_ptr<ReqWrapper> reqWrapper;
  auto& stressorCtx = getStressorCtx();
  auto& reqQ = *stressorCtx.reqQueue_;
  auto& resubmitQueue = stressorCtx.resubmitQueue_;

  if (ampFactor_ > 1 && keySuffixLocal < ampFactor_) {
    if (!resubmitQueue.empty()) {
      reqWrapper.swap(resubmitQueue.front());
      resubmitQueue.pop();
    } else {
      keySuffixLocal += 1;
      // Use a copy of ReqWrapper except for the last one
      reqWrapper = std::make_unique<ReqWrapper>(*curReqWrapper);

      if (reqWrapper->req_.key.size() > 10) {
        // trunkcate the key
        size_t newSize = std::max<size_t>(reqWrapper->req_.key.size() - 4, 10u);
        reqWrapper->req_.key.resize(newSize, '0');
      }
      reqWrapper->req_.key.append(folly::sformat("{:04d}", keySuffixLocal));
    }
    ReqWrapper* reqPtr = reqWrapper.release();
    return reqPtr->req_;
  }

  while (resubmitQueue.empty() && !reqQ.read(reqWrapper)) {
    if (resubmitQueue.empty() && isEOF()) {
      throw cachelib::cachebench::EndOfTrace("Test stopped or EOF reached");
    }
    // ProducerConsumerQueue does not support blocking, so use sleep
    std::this_thread::sleep_for(std::chrono::microseconds{checkIntervalUs_});
  }

  if (!resubmitQueue.empty() && !reqWrapper) {
    XCHECK(!resubmitQueue.empty());
    reqWrapper.swap(resubmitQueue.front());
    resubmitQueue.pop();
  }
  else if (ampFactor_ > 1) {
    keySuffixLocal = 1;
    curReqWrapper.reset();
    curReqWrapper = std::make_unique<ReqWrapper>(*reqWrapper);
  }
  ReqWrapper* reqPtr = reqWrapper.release();
  return reqPtr->req_;
}

void KVReplayGenerator::notifyResult(uint64_t requestId, OpResultType) {
  // requestId should point to the ReqWrapper object. The ownership is taken
  // here to do the clean-up properly if not resubmitted
  std::unique_ptr<ReqWrapper> reqWrapper(
      reinterpret_cast<ReqWrapper*>(requestId));
  XCHECK_GT(reqWrapper->repeats_, 0u);
  if (--reqWrapper->repeats_ == 0) {
    return;
  }
  // need to insert into the queue again
  getStressorCtx().resubmitQueue_.emplace(std::move(reqWrapper));
}

} // namespace cachebench
} // namespace cachelib
} // namespace facebook

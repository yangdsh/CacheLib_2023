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

#include <folly/MPMCQueue.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <atomic>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include "cachelib/allocator/serialize/gen-cpp2/objects_types.h"
#pragma GCC diagnostic pop

#include <folly/lang/Aligned.h>
#include <folly/synchronization/DistributedMutex.h>

#include "cachelib/allocator/datastruct/AtomicDList.h"
#include "cachelib/allocator/datastruct/AtomicFIFOHashTable.h"
#include "cachelib/allocator/datastruct/DList.h"
#include "cachelib/common/BloomFilter.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

template <typename T, AtomicDListHook<T> T::*HookPtr>
class S3FIFOList {
 public:
  using Mutex = folly::DistributedMutex;
  using LockHolder = std::unique_lock<Mutex>;
  using CompressedPtr = typename T::CompressedPtr;
  using PtrCompressor = typename T::PtrCompressor;
  using ADList = AtomicDList<T, HookPtr>;
  using RefFlags = typename T::Flags;
  using S3FIFOListObject = serialization::S3FIFOListObject;

  S3FIFOList() = default;
  S3FIFOList(const S3FIFOList&) = delete;
  S3FIFOList& operator=(const S3FIFOList&) = delete;

  S3FIFOList(PtrCompressor compressor) noexcept {
    pfifo_ = std::make_unique<ADList>(compressor);
    mfifo_ = std::make_unique<ADList>(compressor);
  }

  // Restore S3FIFOList from saved state.
  //
  // @param object              Save S3FIFOList object
  // @param compressor          PtrCompressor object
  S3FIFOList(const S3FIFOListObject& object, PtrCompressor compressor) {
    pfifo_ = std::make_unique<ADList>(*object.pfifo(), compressor);
    mfifo_ = std::make_unique<ADList>(*object.mfifo(), compressor);
  }

  /**
   * Exports the current state as a thrift object for later restoration.
   */
  S3FIFOListObject saveState() const {
    S3FIFOListObject state;
    *state.pfifo() = pfifo_->saveState();
    *state.mfifo() = mfifo_->saveState();
    return state;
  }

  void replaceNode(T& oldNode, T& newNode) noexcept {
    if (isProbationary(oldNode)) {
      markProbationary(newNode);
      pfifo_->replace(oldNode, newNode);
    } else if (isMain(oldNode)) {
      markMain(newNode);
      mfifo_->replace(oldNode, newNode);
    }
  }

  void removeNode(T& node) noexcept {
    if (isMain(node)) {
      mfifo_->remove(node);
      mfifo_eviction_budget -= 1;
    } else if (isProbationary(node)) {
      pfifo_->remove(node);
    }
    /*if (cid == 4)
      XLOG_EVERY_MS(INFO, 1000) << "<cid=" << cid << "> " << pfifo_->size() << ' ' << mfifo_->size() <<
        " s3 promote/evict  " << s3_reinsert_cnt << ':' << s3_evict_cnt << ' ' << cnt_p << ' ' << cnt_m;
    */
  }

  ADList& getListMain() const noexcept { return *mfifo_; }

  // T* getTail() const noexcept { return pfifo_->getTail(); }

  size_t size() const noexcept { return pfifo_->size() + mfifo_->size(); }

  void setECMode(int mode, int cid_, float pRatio) {
    EC_mode = mode;
    pRatio_ = pRatio;
    cid = cid_;
    candidate_from_pfifo_promote = mode & 1;
    candidate_from_pfifo_evict = mode & 2;
    candidate_from_mfifo_promote = mode & 4;
    candidate_from_mfifo_evict = mode & 8;
  }
  
  void getCandidates(T** nodeList, T** evictList, int& length, int& evictLength) noexcept {
    uint32_t n_promoted = 0, n_evict = 0;
    size_t listSize = pfifo_->size() + mfifo_->size();
    if (listSize == 0) {
      length = 0;
      evictLength = 0;
      return;
    }

    T* curr = nullptr;
    if (!hist_.initialized()) {
      LockHolder l(*mtx_);
      if (!hist_.initialized()) {
        hist_.setFIFOSize(listSize / 2);
        hist_.initHashtable();
      }
    }
    int _mfifo_eviction_budget = mfifo_eviction_budget;
    while (true) {
      if (n_promoted + n_evict == length)
        break;
      bool evict_main = false;
      bool tiny_too_large = pfifo_->size() > (double)(pfifo_->size() + mfifo_->size()) * pRatio_;
      if (!mfifo_populated && !tiny_too_large) {
        mfifo_populated = 1;
        mfifo_eviction_budget = 0;
      }
      if (_mfifo_eviction_budget > 0 && mfifo_populated)
        evict_main = true;

      if (!evict_main) {
        if (pfifo_->size() == 0) {
          _mfifo_eviction_budget = 1;
          continue;
        }
        curr = pfifo_->removeTail();
        curr->set_item_flag(0);
        if (pfifo_->isAccessed(*curr)) {
          mfifo_eviction_budget ++;
          _mfifo_eviction_budget ++;
          pfifo_->unmarkAccessed(*curr);
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          if (candidate_from_pfifo_promote) {
            nodeList[n_promoted++] = curr;
          }
          curr->set_item_flag2(0);
        } else {
          mfifo_eviction_budget ++;
          hist_.insert(hashNode(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          if (candidate_from_pfifo_evict) {
            nodeList[n_promoted++] = curr;
          } else {
            evictList[n_evict++] = curr;
          }
          curr->set_item_flag2(1);
        }
      } else {
        curr = mfifo_->removeTail();
        curr->set_item_flag(1);
        if (curr == nullptr) {
          break;
        }
        if (mfifo_->isAccessed(*curr)) {
          mfifo_->unmarkAccessed(*curr);
          mfifo_->linkAtHead(*curr);
          if (candidate_from_mfifo_promote)
            nodeList[n_promoted++] = curr;
          s3_reinsert_cnt ++;
          curr->set_item_flag2(0);
        } else {
          _mfifo_eviction_budget --;
          mfifo_->linkAtHead(*curr);
          if (candidate_from_mfifo_evict) { // && curr->get_is_reinserted()) {
            nodeList[n_promoted++] = curr;
          } else {
            evictList[n_evict++] = curr;
          }
          s3_evict_cnt ++;
          if (curr->get_is_reinserted())
            cnt_m ++;
          else
            cnt_p ++;
          curr->set_item_flag2(1);
        }
      }
    }
    length = n_promoted;
    evictLength = n_evict;
  }

  T* getEvictionCandidate() {
    size_t listSize = pfifo_->size() + mfifo_->size();
    if (listSize == 0) {
      //XLOG(INFO) << cid << ' ' << pfifo_->size() << ' ' << mfifo_->size();
      return nullptr;
    }

    T* curr = nullptr;
    if (!hist_.initialized()) {
      LockHolder l(*mtx_);
      if (!hist_.initialized()) {
        hist_.setFIFOSize(listSize / 2);
        hist_.initHashtable();
      }
    }

    while (true) {
      bool evict_main = false;
      bool tiny_too_large = pfifo_->size() > (double)(pfifo_->size() + mfifo_->size()) * pRatio_;
      if (!mfifo_populated && !tiny_too_large) {
        mfifo_populated = 1;
        mfifo_eviction_budget = 0;
      }
      if (mfifo_eviction_budget > 0 && mfifo_populated)
        evict_main = true;

      if (tiny_too_large && !evict_main) {
        curr = pfifo_->removeTail();
        if (curr == nullptr) {
          printf("pfifo_->size() = %zu, %zu\n", pfifo_->size(), mfifo_->size());
          mfifo_eviction_budget = 1;
          continue;
        }
        curr->set_item_flag(0);
        if (pfifo_->isAccessed(*curr)) {
          mfifo_eviction_budget ++;
          pfifo_->unmarkAccessed(*curr);
          XDCHECK(isProbationary(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          curr->set_item_flag2(0);
        } else {
          mfifo_eviction_budget ++;
          hist_.insert(hashNode(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          curr->set_item_flag2(1);
          return curr;
        }
      } else {
        curr = mfifo_->removeTail();
        if (curr == nullptr) {
          printf("mfifo_->size() = %zu, %zu\n", pfifo_->size(), mfifo_->size());
          continue;
        }
        curr->set_item_flag(1);
        if (mfifo_->isAccessed(*curr)) {
          mfifo_->unmarkAccessed(*curr);
          mfifo_->linkAtHead(*curr);
          s3_reinsert_cnt ++;
          curr->set_item_flag2(0);
        } else {
          mfifo_->linkAtHead(*curr);
          s3_evict_cnt ++;
          curr->set_item_flag2(1);
          return curr;
        }
      }
    }
  }

  void add(T& node) noexcept {
    if (hist_.initialized() && hist_.contains(hashNode(node))) {
      mfifo_->linkAtHead(node);
      markMain(node);
      unmarkProbationary(node);
      insert_to_mfifo += 1;
      mfifo_eviction_budget ++;
      node.set_is_reinserted(1);
    } else {
      pfifo_->linkAtHead(node);
      markProbationary(node);
      unmarkMain(node);
      insert_to_pfifo += 1;
      node.set_is_reinserted(0);
      /*
      if (cid == 4)
        XLOG_EVERY_MS(INFO, 1000) << "<cid=" << cid << "> insert to p/m: " << insert_to_pfifo << ' ' << insert_to_mfifo;
      */
    }
  }

  void moveToHeadLocked(T& node) noexcept {
    if (isMain(node)) {
      mfifo_->remove(node);
      mfifo_->linkAtHead(node);
    } else if (isProbationary(node)) {
      pfifo_->remove(node);
      pfifo_->linkAtHead(node);
    }
  }

  // Bit MM_BIT_0 is used to record if the item is hot.
  void markProbationary(T& node) noexcept {
    node.template setFlag<RefFlags::kMMFlag0>();
  }

  void unmarkProbationary(T& node) noexcept {
    node.template unSetFlag<RefFlags::kMMFlag0>();
  }

  bool isProbationary(const T& node) const noexcept {
    return node.template isFlagSet<RefFlags::kMMFlag0>();
  }

  // Bit MM_BIT_2 is used to record if the item is cold.
  void markMain(T& node) noexcept {
    node.template setFlag<RefFlags::kMMFlag2>();
  }

  void unmarkMain(T& node) noexcept {
    node.template unSetFlag<RefFlags::kMMFlag2>();
  }

  bool isMain(const T& node) const noexcept {
    return node.template isFlagSet<RefFlags::kMMFlag2>();
  }

 private:
  static uint32_t hashNode(const T& node) noexcept {
    return static_cast<uint32_t>(
        folly::hasher<folly::StringPiece>()(node.getKey()));
  }
  std::unique_ptr<ADList> pfifo_;

  std::unique_ptr<ADList> mfifo_;

  mutable folly::cacheline_aligned<Mutex> mtx_;

  double pRatio_ = 0.05;

  AtomicFIFOHashTable hist_;

  int s3_evict_cnt = 0;
  int s3_reinsert_cnt = 0;
  int cnt_p = 0;
  int cnt_m = 0;
  int cid = -1;
  int insert_to_pfifo = 0;
  int insert_to_mfifo = 0;
  int mfifo_eviction_budget = 0;
  bool mfifo_populated = 0;
  bool candidate_from_pfifo_promote = 0;
  bool candidate_from_pfifo_evict = 0;
  bool candidate_from_mfifo_promote = 0;
  bool candidate_from_mfifo_evict = 0;
  bool EC_mode = 0;
};
} // namespace cachelib
} // namespace facebook

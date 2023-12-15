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
    } else {
      markMain(newNode);
      mfifo_->replace(oldNode, newNode);
    }
  }

  void removeNode(T& node) noexcept {
    if (isProbationary(node)) {
      pfifo_->remove(node);
    } else {
      mfifo_->remove(node);
    }
  }

  ADList& getListMain() const noexcept { return *mfifo_; }

  // T* getTail() const noexcept { return pfifo_->getTail(); }

  size_t size() const noexcept { return pfifo_->size() + mfifo_->size(); }

  void setECMode() {
    ECMode = true;
  }
  
  void getCandidates(T** nodeList, int& length) noexcept {
    uint32_t n_promoted = 0;
    size_t listSize = pfifo_->size() + mfifo_->size();
    if (listSize == 0) {
      length = 0;
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

    while (true) {
      if (n_promoted == length)
        return;
      if (pfifo_->size() > (double)(pfifo_->size() + mfifo_->size()) * pRatio_) {
        // evict from probationary FIFO
        curr = pfifo_->removeTail();
        if (pfifo_->isAccessed(*curr)) {
          pfifo_->unmarkAccessed(*curr);
          XDCHECK(isProbationary(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          //nodeList[n_promoted] = curr;
          //curr->set_item_flag(1);
          //n_promoted ++;
        } else {
          hist_.insert(hashNode(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          nodeList[n_promoted] = curr;
          curr->set_item_flag(0);
          n_promoted ++;
        }
      } else {
        curr = mfifo_->removeTail();
        if (curr == nullptr) {
          break;
        }
        if (mfifo_->isAccessed(*curr)) {
          mfifo_->unmarkAccessed(*curr);
          mfifo_->linkAtHead(*curr);
          nodeList[n_promoted] = curr;
          curr->set_item_flag(1);
          n_promoted ++;
        } else {
          mfifo_->linkAtHead(*curr);
          nodeList[n_promoted] = curr;
          curr->set_item_flag(1);
          n_promoted ++;
        }
      }
    }
    length = n_promoted;
  }

  T* getEvictionCandidate() {
    if (ECMode) {
      if (mfifo_->size() > 0)
        return mfifo_->getTail();
      else
        return pfifo_->getTail();
    }
    size_t listSize = pfifo_->size() + mfifo_->size();
    if (listSize == 0) {
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
      if (pfifo_->size() > (double)(pfifo_->size() + mfifo_->size()) * pRatio_) {
        // evict from probationary FIFO
        curr = pfifo_->removeTail();
        if (curr == nullptr) {
          if (pfifo_->size() != 0) {
            printf("pfifo_->size() = %zu, %zu\n", pfifo_->size(), mfifo_->size());
            pfifo_->resetSize();
          }
          continue;
        }
        if (pfifo_->isAccessed(*curr)) {
          pfifo_->unmarkAccessed(*curr);
          XDCHECK(isProbationary(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
        } else {
          hist_.insert(hashNode(*curr));
          unmarkProbationary(*curr);
          markMain(*curr);
          mfifo_->linkAtHead(*curr);
          return curr;
        }
      } else {
        curr = mfifo_->removeTail();
        if (curr == nullptr) {
          if (mfifo_->size() != 0) {
            printf("mfifo_->size() = %zu, %zu\n", pfifo_->size(), mfifo_->size());
            mfifo_->resetSize();
          }
          continue;
        }
        if (mfifo_->isAccessed(*curr)) {
          mfifo_->unmarkAccessed(*curr);
          mfifo_->linkAtHead(*curr);
        } else {
          mfifo_->linkAtHead(*curr);
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
    } else {
      pfifo_->linkAtHead(node);
      markProbationary(node);
      unmarkMain(node);
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

  constexpr static double pRatio_ = 0.05;

  AtomicFIFOHashTable hist_;

  bool ECMode = false;
};
} // namespace cachelib
} // namespace facebook

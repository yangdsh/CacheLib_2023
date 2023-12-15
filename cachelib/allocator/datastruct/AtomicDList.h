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

#include <folly/logging/xlog.h>

#include <algorithm>
#include <atomic>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include "cachelib/allocator/serialize/gen-cpp2/objects_types.h"
#pragma GCC diagnostic pop

#include <folly/lang/Aligned.h>
#include <folly/synchronization/DistributedMutex.h>

#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

template <typename T>
struct CACHELIB_PACKED_ATTR AtomicDListHook {
  using Time = uint32_t;
  using CompressedPtr = typename T::CompressedPtr;
  using PtrCompressor = typename T::PtrCompressor;

  void setNext(T* const n, const PtrCompressor& compressor) noexcept {
    next_ = compressor.compress(n);
  }

  void setNext(CompressedPtr next) noexcept { next_ = next; }

  void setPrev(T* const p, const PtrCompressor& compressor) noexcept {
    prev_ = compressor.compress(p);
  }

  void setPrev(CompressedPtr prev) noexcept { prev_ = prev; }

  CompressedPtr getNext() const noexcept { return CompressedPtr(next_); }

  T* getNext(const PtrCompressor& compressor) const noexcept {
    return compressor.unCompress(next_);
  }

  CompressedPtr getPrev() const noexcept { return CompressedPtr(prev_); }

  T* getPrev(const PtrCompressor& compressor) const noexcept {
    return compressor.unCompress(prev_);
  }

  // set and get the time when the node was updated in the lru.
  void setUpdateTime(Time time) noexcept { updateTime_ = time; }

  Time getUpdateTime() const noexcept {
    // Suppress TSAN here because we don't care if an item is promoted twice by
    // two get operations running concurrently. It should be very rarely and is
    // just a minor inefficiency if it happens.
    folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
    return updateTime_;
  }

 private:
  CompressedPtr next_{}; // next node in the linked list
  CompressedPtr prev_{}; // previous node in the linked list
  // timestamp when this was last updated to the head of the list
  Time updateTime_{0};
};

// uses a double linked list to implement an LRU. T must be have a public
// member of type Hook and HookPtr must point to that.
template <typename T, AtomicDListHook<T> T::*HookPtr>
class AtomicDList {
 public:
  using Mutex = folly::DistributedMutex;
  using LockHolder = std::unique_lock<Mutex>;
  using CompressedPtr = typename T::CompressedPtr;
  using PtrCompressor = typename T::PtrCompressor;
  using RefFlags = typename T::Flags;
  using AtomicDListObject = serialization::AtomicDListObject;

  AtomicDList() = default;
  AtomicDList(const AtomicDList&) = delete;
  AtomicDList& operator=(const AtomicDList&) = delete;

  explicit AtomicDList(PtrCompressor compressor) noexcept
      : compressor_(std::move(compressor)) {}

  // Restore AtomicDList from saved state.
  //
  // @param object              Save AtomicDList object
  // @param compressor          PtrCompressor object
  AtomicDList(const AtomicDListObject& object, PtrCompressor compressor)
      : compressor_(std::move(compressor)),
        head_(compressor_.unCompress(CompressedPtr{*object.compressedHead()})),
        tail_(compressor_.unCompress(CompressedPtr{*object.compressedTail()})),
        size_(*object.size()) {}

  /**
   * Exports the current state as a thrift object for later restoration.
   */
  AtomicDListObject saveState() const {
    AtomicDListObject state;
    *state.compressedHead() = compressor_.compress(head_).saveState();
    *state.compressedTail() = compressor_.compress(tail_).saveState();
    *state.size() = size_;
    return state;
  }

  T* getNext(const T& node) const noexcept {
    return (node.*HookPtr).getNext(compressor_);
  }

  T* getPrev(const T& node) const noexcept {
    return (node.*HookPtr).getPrev(compressor_);
  }

  void setNext(T& node, T* next) noexcept {
    (node.*HookPtr).setNext(next, compressor_);
  }

  void setNextFrom(T& node, const T& other) noexcept {
    (node.*HookPtr).setNext((other.*HookPtr).getNext());
  }

  void setPrev(T& node, T* prev) noexcept {
    (node.*HookPtr).setPrev(prev, compressor_);
  }

  void setPrevFrom(T& node, const T& other) noexcept {
    (node.*HookPtr).setPrev((other.*HookPtr).getPrev());
  }

  // Links the passed node to the head of the double linked list
  // @param node node to be linked at the head
  void linkAtHead(T& node) noexcept;

  void linkBatchAtHead(T& nodeHead, T& nodeTail, int length) noexcept;

  void linkAtTail(T& node) noexcept;

  // Links the passed node to the head of the double linked list
  // @param node node to be linked at the head
  T* removeTail() noexcept;

  void removeBatchTail(T& nodeHead, T& nodeTail, int length) noexcept;

  // removes the node completely from the linked list and cleans up the node
  // appropriately by setting its next and prev as nullptr.
  void remove(T& node) noexcept;

  // Unlinks the destination node and replaces it with the source node
  //
  // @param oldNode   destination node
  // @param newNode   source node
  void replace(T& oldNode, T& newNode) noexcept;

  T* getHead() const noexcept { return head_; }
  T* getTail() const noexcept { return tail_; }

  size_t size() const noexcept { return size_; }

  void resetSize() noexcept {
    size_ = 0;
    head_ = nullptr;
    tail_ = nullptr;
  }

  void markAccessed(T& node) noexcept {
    node.template setFlag<RefFlags::kMMFlag1>();
  }

  void unmarkAccessed(T& node) noexcept {
    node.template unSetFlag<RefFlags::kMMFlag1>();
  }

  bool isAccessed(const T& node) const noexcept {
    return node.template isFlagSet<RefFlags::kMMFlag1>();
  }

  void sanityCheck(std::string tag); 

 private:
  // unlinks the node from the linked list. Does not correct the next and
  // previous.
  void unlink(const T& node) noexcept;

  const PtrCompressor compressor_{};

  mutable folly::cacheline_aligned<Mutex> mtx_;

  // head of the linked list
  T* head_{nullptr};

  // tail of the linked list
  T* tail_{nullptr};

  // size of the list
  size_t size_{0};
};
} // namespace cachelib
} // namespace facebook

#include "cachelib/allocator/datastruct/AtomicDList-inl.h"

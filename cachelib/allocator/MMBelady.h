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

#include <atomic>
#include <cstring>
#include <map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <folly/Format.h>
#pragma GCC diagnostic pop
#include <folly/container/Array.h>
#include <folly/lang/Aligned.h>
#include <folly/synchronization/DistributedMutex.h>

#include "cachelib/allocator/Cache.h"
#include "cachelib/allocator/CacheStats.h"
#include "cachelib/allocator/Util.h"
#include "cachelib/allocator/datastruct/DList.h"
#include "cachelib/allocator/memory/serialize/gen-cpp2/objects_types.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

// CacheLib's modified LRU policy.
// In classic LRU, the items form a queue according to the last access time.
// Items are inserted to the head of the queue and removed from the tail of the
// queue. Items accessed (used) are moved (promoted) to the head of the queue.
// CacheLib made two variations on top of the classic LRU:
// 1. Insertion point. The items are inserted into a configured insertion point
// instead of always to the head.
// 2. Delayed promotion. Items get promoted at most once in any lru refresh time
// window. lru refresh time and lru refresh ratio controls this internval
// length.
class MMBelady {
 public:
  // unique identifier per MMType
  static const int kId;

  // forward declaration;
  template <typename T>
  using Hook = DListHook<T>;
  using SerializationType = serialization::MMBeladyObject;
  using SerializationConfigType = serialization::MMBeladyConfig;
  using SerializationTypeContainer = serialization::MMBeladyCollection;

  // This is not applicable for MMBelady, just for compile of cache allocator
  enum LruType { NumTypes };

  // Config class for MMBelady
  struct Config {

    Config() = default;
    Config(const Config& rhs) = default;
    Config(Config&& rhs) = default;

    Config& operator=(const Config& rhs) = default;
    Config& operator=(Config&& rhs) = default;

    template <typename... Args>
    void addExtraConfig(Args...) {}
  };

  // The container object which can be used to keep track of objects of type
  // T. T must have a public member of type Hook. This object is wrapper
  // around DList, is thread safe and can be accessed from multiple threads.
  // The current implementation models an LRU using the above DList
  // implementation.
  template <typename T, Hook<T> T::*HookPtr>
  struct Container {
   private:
    using Mutex = folly::DistributedMutex;
    using LockHolder = std::unique_lock<Mutex>;
    using PtrCompressor = typename T::PtrCompressor;
    using Time = typename Hook<T>::Time;
    using CompressedPtr = typename T::CompressedPtr;
    using RefFlags = typename T::Flags;

   public:
    Container() = default;
    Container(Config c, PtrCompressor compressor){

    }

    Container(serialization::MMBeladyObject object, PtrCompressor compressor) {

    }

    Container(const Container&) = delete;
    Container& operator=(const Container&) = delete;

    // context for iterating the MM container. At any given point of time,
    // there can be only one iterator active since we need to lock for
    // iteration. we can support multiple iterators at same time, by using a
    // shared ptr in the context for the lock holder in the future.
    class LockedIterator {
     public:
      // noncopyable but movable.
      LockedIterator(const LockedIterator&) = delete;
      LockedIterator& operator=(const LockedIterator&) = delete;

      LockedIterator(LockedIterator&&) noexcept = default;

      // moves the LockedIterator forward and backward. Calling ++ once the
      // LockedIterator has reached the end is undefined.
      LockedIterator& operator++() {
        iter_ ++;
        candidate_ = iter_->second;
        return *this;
      }
      LockedIterator& operator--() { throw std::logic_error("Not implemented"); }

      T* operator->() noexcept { return get(); }
      T& operator*() noexcept { return *get(); }

      explicit operator bool() const noexcept { return l_.owns_lock(); }

      T* get() noexcept {
        if (candidate_ == nullptr) {
          candidate_ = iter_->second;
        }
        return candidate_; 
      }

      // Invalidates this iterator
      void reset() noexcept {
        // Set index to before first list
        // index_ = kInvalidIndex;
        // Point iterator to first list's rend
        // currIter_ = mlist_.lists_[0]->rend();
      }

      // 1. Invalidate this iterator
      // 2. Unlock
      void destroy() {
        if (l_.owns_lock()) {
          l_.unlock();
        }
      }

      // Reset this iterator to the beginning
      void resetToBegin() noexcept {
        if (!l_.owns_lock()) {
          l_.lock();
        }
      }

     private:
      // private because it's easy to misuse and cause deadlock for MMS3FIFO
      LockedIterator& operator=(LockedIterator&&) noexcept = default;

      // create an iterator with the lock being held.
      LockedIterator(LockHolder l, typename std::multimap<int64_t, T*>::reverse_iterator iter) {
        l_ = std::move(l);
        iter_ = iter;
      }

      T* candidate_ = nullptr;
      
      LockHolder l_;

      typename std::multimap<int64_t, T*>::reverse_iterator iter_;

      // only the container can create iterators
      friend Container<T, HookPtr>;
    };

    bool recordAccess(T& node, AccessMode mode) noexcept {
      return add(node);
    }

    bool add(T& node) noexcept {
      LockHolder l(*lruMutex_);
      uint64_t t = 0;
#ifdef TRUE_TTA
      t = node.next_timestamp;
#endif
      if (node2rank.find(&node) != node2rank.end()) {
        auto rank_iter = node2rank[&node];
        rank2node.erase(rank_iter);
      }
      node2rank[&node] = rank2node.insert({t, &node});
      node.markInMMContainer();
      return true;
    }

    bool remove(T& node) noexcept {
      LockHolder l(*lruMutex_);
      removeLocked(node);
      return true;
    }

    void removeLocked(T& node) {
      uint64_t t = 0;
#ifdef TRUE_TTA
      t = node.next_timestamp;
#endif
      if (node2rank.find(&node) != node2rank.end()) {
        auto rank_iter = node2rank[&node];
        rank2node.erase(rank_iter);
        node2rank.erase(&node);
      }
      node.unmarkInMMContainer();
    }

    void remove(LockedIterator& it) noexcept {
      T& node = it.get();
      removeLocked(node);
    }

    bool replace(T& oldNode, T& newNode) noexcept {
      return true;
    }

    // Obtain an iterator that start from the tail and can be used
    // to search for evictions. This iterator holds a lock to this
    // container and only one such iterator can exist at a time
    LockedIterator getEvictionIterator(bool fromTail = true) noexcept {
      LockHolder l(*lruMutex_);
      XLOG_EVERY_MS(INFO, 1000) << "candidate next_t: " << rank2node.rbegin()->second->next_timestamp
        << ' ' << rank2node.begin()->first << ' ' << rank2node.rbegin()->first
        << ' ' << rank2node.size() << ' ' << rank2node.size();
      return LockedIterator{std::move(l), rank2node.rbegin()};
    }

    // Execute provided function under container lock. Function gets
    // iterator passed as parameter.
    template <typename F>
    void withEvictionIterator(F&& f) {

    }

    // get copy of current config
    Config getConfig() const {
      return lruMutex_->lock_combine([this]() { return config_; });
    }

    // override the existing config with the new one.
    void setConfig(const Config& newConfig) {

    }

    bool isEmpty() const noexcept { return size() == 0; }

    // reconfigure the MMContainer: update refresh time according to current
    // tail age
    void reconfigureLocked(const Time& currTime) {

    }

    // returns the number of elements in the container
    size_t size() const noexcept {
      return lruMutex_->lock_combine([this]() { return rank2node.size(); });
    }

    size_t getListSize(const T& node) noexcept {
      return rank2node.size();
    }

    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) {
      EvictionAgeStat stat{};
      return stat;
    }

    // returns the number of elements in the container
    size_t sizeLocked() const noexcept {
      return rank2node.size();
    }

    void setECMode(int mode, int cid=0, float v=0) {
      return;
    }

    static bool isLRU() {return true;}

    uint8_t getFreq(const T& node) {
      return 0;
    }

    static void markReinserted(T& node) noexcept {return;}

    void moveToHeadLocked(T& node) noexcept {

    }

    size_t counterSize() {return 0;}

    bool moveBatchToHeadLocked(T& nodeHead, T& nodeTail, int length) noexcept {
      return true;
    }

    void getCandidates(T** nodeList, T** evictList, int& length, int& evictLength) noexcept {;}
    

    // Returns the eviction age stats. See CacheStats.h for details
    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) const noexcept;

    // for saving the state of the lru
    //
    // precondition:  serialization must happen without any reader or writer
    // present. Any modification of this object afterwards will result in an
    // invalid, inconsistent state for the serialized data.
    //
    serialization::MMBeladyObject saveState() const noexcept {
      serialization::MMBeladyObject object;
      return object;
    }

    // return the stats for this container.
    MMContainerStat getStats() const noexcept {
      return {0, 0, 0, 0, 0, 0, 0};
    }

    static LruType getLruType(const T& /* node */) noexcept {
      return LruType{};
    }

   private:
    EvictionAgeStat getEvictionAgeStatLocked(
        uint64_t projectedLength) const noexcept;

    static Time getUpdateTime(const T& node) noexcept {
      return (node.*HookPtr).getUpdateTime();
    }

    static void setUpdateTime(T& node, Time time) noexcept {
      (node.*HookPtr).setUpdateTime(time);
    }

    // Bit MM_BIT_0 is used to record if the item is in tail. This
    // is used to implement LRU insertion points
    void markTail(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag0>();
    }

    void unmarkTail(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag0>();
    }

    bool isTail(T& node) const noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag0>();
    }

    // Bit MM_BIT_1 is used to record if the item has been accessed since
    // being written in cache. Unaccessed items are ignored when determining
    // projected update time.
    void markAccessed(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag1>();
    }

    void unmarkAccessed(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag1>();
    }

    bool isAccessed(const T& node) const noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag1>();
    }

    // protects all operations on the lru. We never really just read the state
    // of the LRU. Hence we dont really require a RW mutex at this point of
    // time.
    mutable folly::cacheline_aligned<Mutex> lruMutex_, lruHeadMutex_;

    typename std::multimap<int64_t, T*> rank2node;

    typename std::map<T*, typename std::multimap<int64_t, T*>::iterator> node2rank;

    // insertion point
    T* insertionPoint_{nullptr};

    // size of tail after insertion point
    size_t tailSize_{0};

    // The next time to reconfigure the container.
    std::atomic<Time> nextReconfigureTime_{};

    // How often to promote an item in the eviction queue.
    std::atomic<uint32_t> lruRefreshTime_{};

    // Config for this lru.
    // Write access to the MMLru Config is serialized.
    // Reads may be racy.
    Config config_{};

  };
};
} // namespace cachelib
} // namespace facebook

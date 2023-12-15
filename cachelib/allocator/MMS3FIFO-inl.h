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

namespace facebook {
namespace cachelib {

/* Container Interface Implementation */
template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
MMS3FIFO::Container<T, HookPtr>::Container(serialization::MMS3FIFOObject object,
                                         PtrCompressor compressor)
    : qdlist_(*object.qdlist(), compressor), config_(*object.config()) {
  nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
                             ? std::numeric_limits<Time>::max()
                             : static_cast<Time>(util::getCurrentTimeSec()) +
                                   config_.mmReconfigureIntervalSecs.count();
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
bool MMS3FIFO::Container<T, HookPtr>::recordAccess(T& node,
                                                 AccessMode mode) noexcept {
  if ((mode == AccessMode::kWrite && !config_.updateOnWrite) ||
      (mode == AccessMode::kRead && !config_.updateOnRead)) {
    return false;
  }

  const auto curr = static_cast<Time>(util::getCurrentTimeSec());
  // check if the node is still being memory managed
  if (node.isInMMContainer()) {
    if (!isAccessed(node)) {
      markAccessed(node);
    }
    setUpdateTime(node, curr);

    // auto func = [this, &node, curr]() {
    //   reconfigureLocked(curr);
    //   if (node.isInMMContainer()) {
    //     qdlist_.moveToHead(node);
    //     setUpdateTime(node, curr);
    //   }
    // };

    // lruMutex_->lock_combine(func);
    return true;
  }
  return false;
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat MMS3FIFO::Container<T, HookPtr>::getEvictionAgeStat(
    uint64_t projectedLength) const noexcept {
  return lruMutex_->lock_combine([this, projectedLength]() {
    return getEvictionAgeStatLocked(projectedLength);
  });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat
MMS3FIFO::Container<T, HookPtr>::getEvictionAgeStatLocked(
    uint64_t projectedLength) const noexcept {
  EvictionAgeStat stat{};
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());
  printf("getEvictionAgeStatLocked not implemented yet\n");

  // const T* node = qdlist_.getTail();
  // stat.warmQueueStat.oldestElementAge =
  //     node ? currTime - getUpdateTime(*node) : 0;
  // for (size_t numSeen = 0; numSeen < projectedLength && node != nullptr;
  //      numSeen++, node = qdlist_.getPrev(*node)) {
  // }
  // stat.warmQueueStat.projectedAge = node ? currTime - getUpdateTime(*node)
  //                                        :
  //                                        stat.warmQueueStat.oldestElementAge;
  return stat;
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
void MMS3FIFO::Container<T, HookPtr>::setConfig(const Config& newConfig) {
  lruMutex_->lock_combine([this, newConfig]() {
    config_ = newConfig;
    nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
                               ? std::numeric_limits<Time>::max()
                               : static_cast<Time>(util::getCurrentTimeSec()) +
                                     config_.mmReconfigureIntervalSecs.count();
  });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
typename MMS3FIFO::Config MMS3FIFO::Container<T, HookPtr>::getConfig() const {
  return lruMutex_->lock_combine([this]() { return config_; });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
bool MMS3FIFO::Container<T, HookPtr>::add(T& node) noexcept {
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());

  return lruMutex_->lock_combine([this, &node, currTime]() {
    if (node.isInMMContainer()) {
      return false;
    }

    qdlist_.add(node);
    unmarkAccessed(node);
    node.markInMMContainer();
    setUpdateTime(node, currTime);

    return true;
  });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
typename MMS3FIFO::Container<T, HookPtr>::LockedIterator
MMS3FIFO::Container<T, HookPtr>::getEvictionIterator() noexcept {
  LockHolder l(*lruMutex_);
  return LockedIterator{std::move(l), &qdlist_};
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
void MMS3FIFO::Container<T, HookPtr>::removeLocked(T& node) noexcept {
  qdlist_.removeNode(node);
  node.unmarkInMMContainer();
  return;
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
bool MMS3FIFO::Container<T, HookPtr>::remove(T& node) noexcept {
  return lruMutex_->lock_combine([this, &node]() {
    if (!node.isInMMContainer()) {
      return false;
    }
    removeLocked(node);
    return true;
  });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
void MMS3FIFO::Container<T, HookPtr>::remove(LockedIterator& it) noexcept {
  T& node = *it;
  XDCHECK(node.isInMMContainer());
  removeLocked(node);
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
bool MMS3FIFO::Container<T, HookPtr>::replace(T& oldNode, T& newNode) noexcept {
  return lruMutex_->lock_combine([this, &oldNode, &newNode]() {
    if (!oldNode.isInMMContainer() || newNode.isInMMContainer()) {
      return false;
    }
    const auto updateTime = getUpdateTime(oldNode);

    qdlist_.replaceNode(oldNode, newNode);

    oldNode.unmarkInMMContainer();
    newNode.markInMMContainer();
    setUpdateTime(newNode, updateTime);
    if (isAccessed(oldNode)) {
      markAccessed(newNode);
    } else {
      unmarkAccessed(newNode);
    }
    return true;
  });
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
serialization::MMS3FIFOObject MMS3FIFO::Container<T, HookPtr>::saveState()
    const noexcept {
  serialization::MMS3FIFOConfig configObject;
  *configObject.updateOnWrite() = config_.updateOnWrite;
  *configObject.updateOnRead() = config_.updateOnRead;

  serialization::MMS3FIFOObject object;
  *object.config() = configObject;
  *object.qdlist() = qdlist_.saveState();
  return object;
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
MMContainerStat MMS3FIFO::Container<T, HookPtr>::getStats() const noexcept {
  auto stat = lruMutex_->lock_combine([this]() {
    // we return by array here because DistributedMutex is fastest when the
    // output data fits within 48 bytes.  And the array is exactly 48 bytes, so
    // it can get optimized by the implementation.
    //
    // the rest of the parameters are 0, so we don't need the critical section
    // to return them
    return folly::make_array(qdlist_.size());
  });
  return {stat[0] /* lru size */,
          // stat[1] /* tail time */,
          0, 0 /* refresh time */, 0, 0, 0, 0};
}

template <typename T, MMS3FIFO::Hook<T> T::*HookPtr>
void MMS3FIFO::Container<T, HookPtr>::reconfigureLocked(const Time& currTime) {
  if (currTime < nextReconfigureTime_) {
    return;
  }
  nextReconfigureTime_ = currTime + config_.mmReconfigureIntervalSecs.count();
}
} // namespace cachelib
} // namespace facebook

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


template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::sanityCheck(std::string tag) {
    if (tag != "check")
      return;
    size_t curr_size = 0;
    T* curr = head_;
    while (curr != nullptr) {
      curr_size++;
      curr = getNext(*curr);
    }
    XDCHECK_EQ(curr_size, size_);

    if (curr_size != size_) {
      printf("%s, curr_size: %zu, size: %zu\n", tag.c_str(), curr_size, size_);
      // abort();
    }
    // printf("curr_size: %zu\n", curr_size);
  }


/* Linked list implemenation */
template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::linkAtHead(T& node) noexcept {
  setNext(node, head_);
  setPrev(node, nullptr);
  // fix the prev ptr of head
  if (head_ != nullptr) {
    setPrev(*head_, &node);
  }
  head_ = &node;
  if (tail_ == nullptr) {
    tail_ = &node;
  }
  size_++;
  sanityCheck("linkAtHead");
}

/* Linked list implemenation */
template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::linkBatchAtHead(T& nodeHead, T& nodeTail, int length) noexcept {
  setNext(nodeTail, head_);
  setPrev(nodeHead, nullptr);
  // fix the prev ptr of head
  if (head_ != nullptr) {
    setPrev(*head_, &nodeTail);
  }
  head_ = &nodeHead;
  if (tail_ == nullptr) {
    tail_ = &nodeTail;
  }
  size_+= length;
  sanityCheck("linkAtHead");
}

/* Linked list implemenation */
template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::linkAtTail(T& node) noexcept {
  setNext(node, nullptr);
  setPrev(node, tail_);
  // Fix the next ptr for tail
  if (tail_ != nullptr) {
    setNext(*tail_, &node);
  }
  tail_ = &node;
  if (head_ == nullptr) {
    head_ = &node;
  }
  size_++;
  sanityCheck("linkAtTail");
}

/* note that the next of the tail may not be nullptr  */
template <typename T, AtomicDListHook<T> T::*HookPtr>
T* AtomicDList<T, HookPtr>::removeTail() noexcept {
  T* tail = tail_;
  remove(*tail);
  sanityCheck("removeTail");
  return tail;
}

template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::removeBatchTail(T& nodeHead, T& nodeTail, int length) noexcept {
  auto* const prev = getPrev(nodeHead);
  auto* const next = getNext(nodeTail);
  tail_ = prev;
  setNext(*prev, next);
  size -= length;
}

template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::unlink(const T& node) noexcept {
  XDCHECK_GT(size_, 0u);
  auto* const prev = getPrev(node);
  auto* const next = getNext(node);

  if (&node == head_) {
    head_ = next;
  }
  if (&node == tail_) {
    tail_ = prev;
  }

  // fix the next and prev ptrs of the node before and after us.
  if (prev != nullptr) {
    setNextFrom(*prev, node);
  }
  if (next != nullptr) {
    setPrevFrom(*next, node);
  }
  size_--;
  sanityCheck("unlink");
}

template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::remove(T& node) noexcept {
  unlink(node);
  setNext(node, nullptr);
  setPrev(node, nullptr);
}

template <typename T, AtomicDListHook<T> T::*HookPtr>
void AtomicDList<T, HookPtr>::replace(T& oldNode, T& newNode) noexcept {
  // Update head and tail links if needed
  if (&oldNode == head_) {
    head_ = &newNode;
  }
  if (&oldNode == tail_) {
    tail_ = &newNode;
  }

  // Make the previous and next nodes point to the new node
  auto* const prev = getPrev(oldNode);
  auto* const next = getNext(oldNode);
  if (prev != nullptr) {
    setNext(*prev, &newNode);
  }
  if (next != nullptr) {
    setPrev(*next, &newNode);
  }

  // Make the new node point to the previous and next nodes
  setPrev(newNode, prev);
  setNext(newNode, next);

  // Cleanup the old node
  setPrev(oldNode, nullptr);
  setNext(oldNode, nullptr);
}
} // namespace cachelib
} // namespace facebook

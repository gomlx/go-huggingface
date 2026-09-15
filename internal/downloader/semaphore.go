package downloader

import (
	"container/list"
	"context"
	"sync"
)

// UnlimitedSemaphoreCapacity is the capacity of a semaphore that is not limited,
// making the FIFOSemaphore a no-op.
const UnlimitedSemaphoreCapacity = 0

// FIFOSemaphore limits concurrent acquisitions with strict first-in, first-out (FIFO) ordering.
// It supports dynamic resizing and context cancellation while waiting in the queue.
type FIFOSemaphore struct {
	mu       sync.Mutex
	capacity int
	current  int
	waiters  *list.List // list of *fifoWaiter
}

// Semaphore is an alias to FIFOSemaphore for backward compatibility.
type Semaphore = FIFOSemaphore

type fifoWaiter struct {
	ready chan struct{}
	done  bool
}

// NewFIFOSemaphore returns a FIFOSemaphore that allows at most capacity simultaneous acquisitions.
// If capacity is set to UnlimitedSemaphoreCapacity (or <= 0), there is no limit on acquisitions.
func NewFIFOSemaphore(capacity int) *FIFOSemaphore {
	if capacity <= UnlimitedSemaphoreCapacity {
		capacity = UnlimitedSemaphoreCapacity
	}
	return &FIFOSemaphore{
		capacity: capacity,
		waiters:  list.New(),
	}
}

// NewSemaphore is an alias for NewFIFOSemaphore for backward compatibility.
func NewSemaphore(capacity int) *FIFOSemaphore {
	return NewFIFOSemaphore(capacity)
}

// Acquire reserves a resource slot observing the semaphore's capacity in strict FIFO order.
// If ctx is cancelled before acquisition, it unregisters from the queue and returns ctx.Err().
// It must be matched by exactly one call to Release after the reservation is no longer needed.
func (s *FIFOSemaphore) Acquire(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		return err
	}

	// If capacity allows and nobody is waiting ahead in queue, acquire immediately.
	if (s.capacity == UnlimitedSemaphoreCapacity || s.current < s.capacity) && s.waiters.Len() == 0 {
		s.current++
		s.mu.Unlock()
		return nil
	}

	// Enqueue waiter at the tail to preserve FIFO order.
	w := &fifoWaiter{ready: make(chan struct{})}
	elem := s.waiters.PushBack(w)
	s.mu.Unlock()

	select {
	case <-w.ready:
		// Granted directly by Release() or Resize().
		return nil
	case <-ctx.Done():
		s.mu.Lock()
		if w.done {
			// Raced with Release/Resize that granted us the slot right as ctx was cancelled.
			// Relinquish the acquired slot by passing it on.
			s.mu.Unlock()
			s.Release()
			return ctx.Err()
		}
		s.waiters.Remove(elem)
		s.mu.Unlock()
		return ctx.Err()
	}
}

// Release releases a resource previously allocated with Acquire.
// If other goroutines are waiting in the queue, it passes the slot to the next waiter in FIFO order.
func (s *FIFOSemaphore) Release() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.capacity == UnlimitedSemaphoreCapacity || s.current <= s.capacity {
		for s.waiters.Len() > 0 {
			elem := s.waiters.Front()
			s.waiters.Remove(elem)
			w := elem.Value.(*fifoWaiter)
			if !w.done {
				w.done = true
				close(w.ready)
				// The slot is transferred directly to the waiter without decrementing current.
				return
			}
		}
	}

	s.current--
	if s.current < 0 {
		s.current = 0
	}
}

// Resize dynamically changes the capacity of the semaphore.
// If newCapacity is larger than the previous capacity, waiting Acquire calls are awoken in strict FIFO order.
// If newCapacity is smaller, active acquisitions are not stopped, and new acquisitions will wait until
// active acquisitions drain below the new capacity.
// If newCapacity is set to UnlimitedSemaphoreCapacity (or <= 0) all waiting Acquire are immediately awoken.
func (s *FIFOSemaphore) Resize(newCapacity int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if newCapacity <= UnlimitedSemaphoreCapacity {
		newCapacity = UnlimitedSemaphoreCapacity
	}
	if newCapacity == s.capacity {
		return
	}
	s.capacity = newCapacity

	// Wake up as many waiters as the new capacity permits, in FIFO order.
	for s.waiters.Len() > 0 && (s.capacity == UnlimitedSemaphoreCapacity || s.current < s.capacity) {
		elem := s.waiters.Front()
		s.waiters.Remove(elem)
		w := elem.Value.(*fifoWaiter)
		if !w.done {
			w.done = true
			close(w.ready)
			s.current++
		}
	}
}

// Capacity returns the current capacity limit (UnlimitedSemaphoreCapacity means unlimited).
func (s *FIFOSemaphore) Capacity() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.capacity
}

// Current returns the number of currently active acquisitions.
func (s *FIFOSemaphore) Current() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

// WaitersLen returns the number of goroutines currently waiting in queue.
func (s *FIFOSemaphore) WaitersLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.waiters.Len()
}

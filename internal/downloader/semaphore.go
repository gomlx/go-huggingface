package downloader

import "sync"

// Semaphore that allows dynamic resizing.
//
// It uses a sync.Cond to allow dynamic resizing, but it will be slower than a pure channel implementation
// with a fixed capacity. This cost shouldn't matter for more coarse resource control.
//
// Implementation copied from github.com/gomlx/gomlx/pkg/support/xsync.
type Semaphore struct {
	cond              sync.Cond
	capacity, current int // Tracks capacity and current usage.
}

// NewSemaphore returns a Semaphore that allows at most capacity simultaneous acquisitions.
// If capacity <= 0, there is no limit on acquisitions.
//
// FIFO ordering may be lost during resizes (Semaphore.Resize) to larger capacity, but otherwise it is respected.
func NewSemaphore(capacity int) *Semaphore {
	return &Semaphore{
		cond:     sync.Cond{L: &sync.Mutex{}},
		capacity: capacity,
	}
}

// Acquire resource observing current semaphore capacity.
// It must be matched by exactly one call to Semaphore.Release after the reservation is no longer needed.
func (s *Semaphore) Acquire() {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	for {
		if s.capacity <= 0 || s.current < s.capacity {
			// No limits.
			s.current++
			return
		}
		s.cond.Wait()
	}
}

// Release resource previously allocated with Semaphore.Acquire.
func (s *Semaphore) Release() {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	s.current--
	if s.capacity == 0 || s.current < s.capacity-1 {
		return
	}
	s.cond.Signal()
}

// Resize the number of available resources in the Semaphore.
//
// If the newCapacity is larger than the previous one, this may immediately allow pending Semaphore.Acquire to proceed.
// Notice since all waiting Semaphore.Acquire are awoken (broadcast), the queue order may be lost.
//
// If the newCapacity is smaller than the previous one, it doesn't have any effect on current acquisitions. So if the Semaphore
// is being used to control a worker pool, reducing its size won't stop workers currently executing.
func (s *Semaphore) Resize(newCapacity int) {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	if newCapacity == s.capacity {
		return // No change needed.
	}
	if (newCapacity > 0 && newCapacity < s.capacity) || s.capacity == 0 {
		// Capacity is shrinking, no Semaphore.Acquire will be released.
		s.capacity = newCapacity
		return
	}

	// Wake-up everyone -- to preserve the queue order, we would need to call s.cond.Signal() for the amount of
	// increased capacity, but that would make this call O(capacity), potentially slow for large capacities.
	s.capacity = newCapacity
	s.cond.Broadcast()
}

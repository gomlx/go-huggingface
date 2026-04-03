package pool

import (
	"unsafe"
)

// cacheLineSize is the cache line size (64 bytes on amd64).
const cacheLineSize = 64

// maxProcs is the maximum number of processors we support.
// We use a fixed size array to avoid bounds checks and resizing complexity in the hot path.
// 4096 * 64 bytes = 256KB, which is negligible memory overhead.
const maxProcs = 4096

// Linkable defines the requirements for an item to be managed by the Pool.
// T represents the pointer to the item (e.g., *MyStruct).
type Linkable[T any] interface {
	Next() T
	SetNext(T)
}

// poolHead is the head of the linked list for a P.
// It is padded to cacheLineSize to prevent false sharing.
type poolHead[E any, P interface {
	*E
	Linkable[P]
}] struct {
	head P

	// raceMutex is used only when the race detector is enabled to establish
	// happens-before relationships for the race detector. In non-race builds,
	// it occupies 8 bytes (unused) to maintain consistent struct layout.
	raceMutex

	// Padding to ensure the struct is 64 bytes.
	// We assume 64-bit architecture (pointer size 8 bytes).
	// head (8) + raceMutex (8) = 16.
	// 64 - 16 = 48.
	_ [48]byte
}

// Pool is a lock-free (per-P) pool of objects.
// Objects in the pool never expire.
//
// The objects must implement the Linkable interface (with a pointer receiver.
//
// Example:
//
//	type MyObject struct {
//		ID   int
//		next *MyObject
//	}
//
//	func (o *MyObject) Next() *MyObject        { return o.next }
//	func (o *MyObject) SetNext(next *MyObject) { o.next = next }
//
// Synchronization is done using runtime.procPin/runtime.procUnpin.
//
// It take two opera types:
// E: The struct type (e.g., UserObject)
// P: The pointer type (e.g., *UserObject)
//
// The constraint `interface { *E; Linkable[P] }` enforces that:
// 1. P is exactly a pointer to E (*E)
// 2. P implements Linkable for itself
type Pool[E any, P interface {
	*E
	Linkable[P]
}] struct {
	heads []poolHead[E, P]
	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	// It may not be changed concurrently with calls to Get.
	New func() P
}

// New creates a new Pool.
func New[E any, P interface {
	*E
	Linkable[P]
}](newFunc func() P) *Pool[E, P] {
	// Verify alignment assumption.
	var p poolHead[E, P]
	if unsafe.Sizeof(p) != cacheLineSize {
		// This should only happen on non-64-bit architectures or if pointer size changes.
		// For now we panic to ensure we meet the spec.
		panic("internal/pool: poolHead size is not 64 bytes")
	}

	return &Pool[E, P]{
		heads: make([]poolHead[E, P], maxProcs),
		New:   newFunc,
	}
}

// Get retrieves an object from the pool.
// If the pool is empty, it allocates a new one using New function or zero value.
func (p *Pool[E, P]) Get() P {
	pid := runtime_procPin()
	// Bounds check.
	if pid >= len(p.heads) {
		runtime_procUnpin()
		// If this happens, maxProcs is insufficient.
		panic("internal/pool: GOMAXPROCS exceeds supported limit")
	}

	h := &p.heads[pid]
	h.lock()
	node := h.head
	if node != nil {
		var next P
		next = node.Next()
		h.head = next
		node.SetNext(nil)
	}
	h.unlock()
	runtime_procUnpin()

	if node == nil {
		node = p.New()
	}
	return node
}

// Put returns an object to the pool.
func (p *Pool[E, P]) Put(node P) {
	if node == nil {
		return
	}
	pid := runtime_procPin()
	if pid >= len(p.heads) {
		runtime_procUnpin()
		panic("internal/pool: GOMAXPROCS exceeds supported limit")
	}

	h := &p.heads[pid]
	h.lock()
	node.SetNext(h.head)
	h.head = node
	h.unlock()
	runtime_procUnpin()
}

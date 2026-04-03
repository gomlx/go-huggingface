package pool

import (
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type MyObject struct {
	ID   int
	next *MyObject
}

func (o *MyObject) Next() *MyObject        { return o.next }
func (o *MyObject) SetNext(next *MyObject) { o.next = next }

func TestPool(t *testing.T) {
	// Object to pool.

	var counter int32
	// Create a pool that assigns a unique ID to each new object.
	pool := New(func() *MyObject {
		id := atomic.AddInt32(&counter, 1)
		return &MyObject{ID: int(id)}
	})

	const numGoroutines = 500
	const numIterations = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for range numGoroutines {
		go func() {
			defer wg.Done()
			for range numIterations {
				node := pool.Get()

				// Simulate work
				time.Sleep(time.Duration(rand.Intn(10)) * time.Microsecond)

				pool.Put(node)
			}
		}()
	}

	wg.Wait()

	// After the test, the number of created objects (counter) should not exceed
	// numGoroutines + GOMAXPROCS approximately (since some might be held in cache).
	// Actually, in the worst case, if all goroutines hold an object simultaneously,
	// we need at least numGoroutines objects.
	// Since the pool never expires, the total created should be roughly max concurrency.

	totalCreated := atomic.LoadInt32(&counter)
	t.Logf("Total objects created: %d", totalCreated)

	if totalCreated > numGoroutines+int32(runtime.GOMAXPROCS(0))*2 {
		t.Logf("Warning: High number of objects created: %d (expected around %d)", totalCreated, numGoroutines)
	}
	// We can't strictly assert the upper bound because scheduling is non-deterministic,
	// but it should definitely be less than numGoroutines * numIterations.
	if totalCreated >= int32(numGoroutines*numIterations) {
		t.Errorf("Pool seems ineffective: created %d objects for %d requests", totalCreated, numGoroutines*numIterations)
	}

	// Verify basic properties
	node := pool.Get()
	if node == nil {
		t.Fatal("Get returned nil")
	}
	pool.Put(node)
}

func BenchmarkPool(b *testing.B) {
	pool := New(func() *MyObject { return &MyObject{ID: 7} })
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			n := pool.Get()
			pool.Put(n)
		}
	})
}

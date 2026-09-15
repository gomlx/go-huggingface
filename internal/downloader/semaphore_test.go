package downloader

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFIFOSemaphore_Basic(t *testing.T) {
	sem := NewFIFOSemaphore(2)
	assert.Equal(t, 2, sem.Capacity())
	assert.Equal(t, 0, sem.Current())

	ctx := context.Background()
	require.NoError(t, sem.Acquire(ctx))
	assert.Equal(t, 1, sem.Current())

	require.NoError(t, sem.Acquire(ctx))
	assert.Equal(t, 2, sem.Current())

	sem.Release()
	assert.Equal(t, 1, sem.Current())

	sem.Release()
	assert.Equal(t, 0, sem.Current())
}

func TestFIFOSemaphore_StrictFIFOOrder(t *testing.T) {
	sem := NewFIFOSemaphore(1)

	// Hold the single slot with initial caller
	require.NoError(t, sem.Acquire(context.Background()))

	const numWaiters = 8
	var orderMu sync.Mutex
	var acquisitionOrder []int

	var wg sync.WaitGroup
	wg.Add(numWaiters)

	// Launch waiters in sequence, verifying each is queued in sem.WaitersLen() before launching the next
	for i := 0; i < numWaiters; i++ {
		id := i
		go func() {
			defer wg.Done()
			err := sem.Acquire(context.Background())
			if err != nil {
				return
			}
			orderMu.Lock()
			acquisitionOrder = append(acquisitionOrder, id)
			orderMu.Unlock()

			// Release so next in FIFO queue can proceed
			sem.Release()
		}()

		require.Eventually(t, func() bool {
			return sem.WaitersLen() == id+1
		}, 1*time.Second, 1*time.Millisecond)
	}

	// Release initial slot; the waiters should now proceed one by one in exact FIFO order
	sem.Release()

	wg.Wait()

	require.Len(t, acquisitionOrder, numWaiters)
	expectedOrder := make([]int, numWaiters)
	for i := 0; i < numWaiters; i++ {
		expectedOrder[i] = i
	}
	assert.Equal(t, expectedOrder, acquisitionOrder, "Waiters must be served in strict FIFO order")
}

func TestFIFOSemaphore_ContextCancellation(t *testing.T) {
	sem := NewFIFOSemaphore(1)
	require.NoError(t, sem.Acquire(context.Background()))

	ctxCancel, cancel := context.WithCancel(context.Background())
	var acquiredSecond atomic.Bool

	doneCancel := make(chan error, 1)
	go func() {
		doneCancel <- sem.Acquire(ctxCancel)
	}()

	require.Eventually(t, func() bool {
		return sem.WaitersLen() == 1
	}, 1*time.Second, 1*time.Millisecond)

	// Queue a third waiter that should succeed
	doneThird := make(chan struct{})
	go func() {
		defer close(doneThird)
		err := sem.Acquire(context.Background())
		if err == nil {
			acquiredSecond.Store(true)
			sem.Release()
		}
	}()

	require.Eventually(t, func() bool {
		return sem.WaitersLen() == 2
	}, 1*time.Second, 1*time.Millisecond)

	// Cancel the second waiter
	cancel()
	err := <-doneCancel
	assert.ErrorIs(t, err, context.Canceled)

	// Release the first slot; the third waiter should now acquire it
	sem.Release()

	select {
	case <-doneThird:
		assert.True(t, acquiredSecond.Load(), "Third waiter should acquire after second was cancelled")
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for third waiter to acquire")
	}

	assert.Equal(t, 0, sem.Current(), "No tokens should leak after cancellation")
}

func TestFIFOSemaphore_ResizeIncrease(t *testing.T) {
	sem := NewFIFOSemaphore(1)
	require.NoError(t, sem.Acquire(context.Background()))

	var acquiredOrder []int
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		id := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			require.NoError(t, sem.Acquire(context.Background()))
			mu.Lock()
			acquiredOrder = append(acquiredOrder, id)
			mu.Unlock()
		}()
		require.Eventually(t, func() bool {
			return sem.WaitersLen() == id+1
		}, 1*time.Second, 1*time.Millisecond)
	}

	// Resize capacity from 1 to 3 -> should immediately allow 2 waiters to acquire (ids 0 and 1)
	sem.Resize(3)
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(acquiredOrder) == 2
	}, 1*time.Second, 1*time.Millisecond)

	mu.Lock()
	assert.ElementsMatch(t, []int{0, 1}, acquiredOrder, "Resize should wake the first 2 waiters in queue (0 and 1)")
	mu.Unlock()

	// Release initial holder's slot: token is passed to waiter 2; waiter 3 must still wait
	sem.Release()
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(acquiredOrder) == 3
	}, 1*time.Second, 1*time.Millisecond)

	mu.Lock()
	assert.Equal(t, 2, acquiredOrder[2], "Waiter 2 should acquire next in FIFO order")
	mu.Unlock()

	// Release another slot: token is passed to waiter 3
	sem.Release()
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(acquiredOrder) == 4
	}, 1*time.Second, 1*time.Millisecond)

	mu.Lock()
	assert.Equal(t, 3, acquiredOrder[3], "Waiter 3 should acquire next in FIFO order")
	mu.Unlock()

	wg.Wait()

	// Drain remaining 3 held tokens (total 5 acquired, 2 released so far, 3 remain)
	sem.Release()
	sem.Release()
	sem.Release()
	assert.Equal(t, 0, sem.Current())
}

func TestFIFOSemaphore_ResizeDecrease(t *testing.T) {
	sem := NewFIFOSemaphore(3)
	ctx := context.Background()

	require.NoError(t, sem.Acquire(ctx))
	require.NoError(t, sem.Acquire(ctx))
	require.NoError(t, sem.Acquire(ctx))
	assert.Equal(t, 3, sem.Current())

	// Shrink capacity to 1
	sem.Resize(1)
	assert.Equal(t, 1, sem.Capacity())

	waiterAcquired := make(chan struct{})
	go func() {
		_ = sem.Acquire(ctx)
		close(waiterAcquired)
		sem.Release()
	}()

	require.Eventually(t, func() bool {
		return sem.WaitersLen() == 1
	}, 1*time.Second, 1*time.Millisecond)

	// Release 1 of the original 3: current goes from 3 to 2, waiter should still be blocked (capacity=1)
	sem.Release()
	select {
	case <-waiterAcquired:
		t.Fatal("Waiter should not acquire while current (2) > capacity (1)")
	case <-time.After(20 * time.Millisecond):
	}

	// Release second of original 3: current goes from 2 to 1, waiter should still be blocked
	sem.Release()
	select {
	case <-waiterAcquired:
		t.Fatal("Waiter should not acquire while current (1) == capacity (1)")
	case <-time.After(20 * time.Millisecond):
	}

	// Release third: now current goes below capacity and waiter acquires!
	sem.Release()
	select {
	case <-waiterAcquired:
		// success
	case <-time.After(1 * time.Second):
		t.Fatal("Waiter should have acquired after draining")
	}

	require.Eventually(t, func() bool {
		return sem.Current() == 0
	}, 1*time.Second, 1*time.Millisecond)
}

func TestFIFOSemaphore_Unlimited(t *testing.T) {
	sem := NewFIFOSemaphore(0) // unlimited
	assert.Equal(t, 0, sem.Capacity())

	ctx := context.Background()
	for i := 0; i < 100; i++ {
		require.NoError(t, sem.Acquire(ctx))
	}
	assert.Equal(t, 100, sem.Current())

	for i := 0; i < 100; i++ {
		sem.Release()
	}
	assert.Equal(t, 0, sem.Current())
}

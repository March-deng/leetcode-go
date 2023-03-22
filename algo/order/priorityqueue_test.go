package order

import (
	"sort"
	"testing"
)

func TestPriorityQueue(t *testing.T) {
	src := []int{9, 89, 10, 8, 5, 78, 45, 4, 3}

	p := NewPriorityQueueInt(src)

	sort.Ints(src)

	for i := len(src) - 1; i >= 0; i++ {
		curMax := p.DeleteMax()
		if curMax != src[i] {
			t.Fatal("queue not ordered")
		}
	}
}

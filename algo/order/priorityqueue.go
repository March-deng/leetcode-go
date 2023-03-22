package order

type PriorityQueueInt struct {
	elems []int
	size  int
}

func NewPriorityQueueInt(src []int) *PriorityQueueInt {
	p := &PriorityQueueInt{
		elems: make([]int, 0),
		size:  0,
	}
	for _, i := range src {
		p.Insert(i)
	}
	return p
}

func (p *PriorityQueueInt) IsEmpty() bool {
	return p.size == 0
}

func (p *PriorityQueueInt) Size() int {
	return p.size
}

func (p *PriorityQueueInt) Insert(value int) {
	n := p.size + 1
	p.size = n
	p.elems[n] = value

	p.swim(n)

}

func (p *PriorityQueueInt) DeleteMax() int {
	value := p.elems[1]

	// exchange
	p.elems[1], p.elems[p.size] = p.elems[p.size], p.elems[1]
	p.elems[p.size] = 0
	p.size -= 1
	p.sink(1)
	return value
}

func (p *PriorityQueueInt) swim(i int) {
	for {
		if i <= 1 {
			break
		}
		next := i / 2
		if p.elems[next] < p.elems[i] {
			// exchange and next turn
			p.elems[next], p.elems[i] = p.elems[i], p.elems[next]
			i = next
		} else {
			break
		}
	}
}

func (p *PriorityQueueInt) sink(i int) {
	for {
		next := 2 * i
		if next > p.size {
			break
		}
		if next < p.size && p.elems[next] < p.elems[next+1] {
			next++
		}
		if !(p.elems[i] < p.elems[next]) {
			break
		}
		p.elems[next], p.elems[i] = p.elems[i], p.elems[next]
		i = next
	}
}

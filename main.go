package main

import (
	"fmt"
	"log"
	"time"
)

func main() {

	chs := runPrint(3)

	a := time.After(5 * time.Second)

	select {
	case <-a:

		for _, ch := range chs {
			close(ch)
		}

	}

	time.Sleep(5 * time.Minute)
}

func runPrint(n int) map[int]chan int {
	chs := make(map[int]chan int, n)
	for i := 1; i <= n; i++ {
		chs[i] = make(chan int)
	}

	for i := 1; i <= n; i++ {
		go func(idx int) {
			ch := chs[idx]
			for {
				select {
				case v, ok := <-ch:
					if !ok {
						log.Printf("Num.%d exit \n", idx)
						return
					}
					fmt.Print(v)
					time.Sleep(500 * time.Millisecond)
					nextChIdx := (idx + 1) % n
					if nextChIdx == 0 {
						nextChIdx = n
					}
					nextCh := chs[nextChIdx]
					nextCh <- v + 1
				}
			}
		}(i)
	}
	chs[1] <- 1

	return chs
}

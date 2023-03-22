package order

// 希尔排序
func ShellOrderInt(src []int) {
	// 先算好步长
	steps := make([]int, 0)

	step := 1
	for 3*step < len(src) {
		steps = append(steps, step)
		step = 3*step + 1
	}

	for i := len(steps); i > 0; i-- {
		step = steps[i-1]
		for j := 0; j < len(src)-step; j++ {
			for k := j + step; k >= step; k -= step {
				if src[k] < src[k-step] {
					src[k], src[k-step] = src[k-step], src[k]
				}
			}
		}

	}
}

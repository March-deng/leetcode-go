package order

func QuickOrderInt(src []int) {
	quickOrder(src, 0, len(src)-1)
}

func quickOrder(src []int, low, high int) {
	if low >= high {
		return
	}

	j := partion(src, low, high)
	quickOrder(src, low, j-1)
	quickOrder(src, j+1, high)
}
func partion(src []int, low, high int) int {
	i := low
	j := high + 1
	value := src[low]

	for {
		// progress left i
		for {
			i++
			if src[i] >= value {
				break
			}
			if i == high {
				break
			}
		}

		// progress right j
		for {
			j--
			if src[j-1] <= value {
				break
			}
			if j == low {
				break
			}
		}
		if i >= j {
			break
		}
		// exchange
		src[i], src[j] = src[j], src[i]
	}
	src[low], src[j] = src[j], src[low]
	return j
}

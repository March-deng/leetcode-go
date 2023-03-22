package order

/*归并排序，这是一种递归式的排序算法
 */
func MergeOrderInt(src []int) {
	copied := make([]int, len(src))
	copy(copied, src)

	mergeSort(src, copied, 0, len(src)-1)

}

func mergeSort(src []int, copied []int, low, high int) {
	// 双指针相遇，递归结束
	if low <= high {
		return
	}

	mid := low + (high-low)/2

	mergeSort(src, copied, low, mid)
	mergeSort(src, copied, mid+1, high)

	merge(src, copied, low, mid, high)

}

func merge(src []int, copied []int, low, mid, high int) {
	i := low
	j := mid + 1

	for k := low; k <= high; k++ {
		// 比较当前i， j位置的大小，谁偏小，就把它放到当前k位置上，这部分已经有序了。谁小谁就前进
		// 如果耗掉了当前半边的所有位置，那就把相对的半边全部拷贝到k位置上
		// 其实类似于双指针
		if i > mid {
			src[k] = copied[j]
			j++
		} else if j > high {
			src[k] = copied[i]
			i++
		} else if copied[j] < copied[i] {
			src[k] = copied[j]
			j++
		} else {
			src[k] = copied[i]
			i++
		}
	}

}

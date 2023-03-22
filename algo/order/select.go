package order

// 选择排序
/*伪代码：
从第一个元素开始，找到当前最小元素的索引，然后将其与第一个元素交换位置，此时第一个元素已经有序
*/
func SelectOrderInt(src []int) {
	for i := 0; i < len(src); i++ {
		minIndex := i
		for j := i; j < len(src); j++ {
			if src[j] < src[minIndex] {
				minIndex = j
			}
		}
		// log.Println("current min value:", src[minIndex])
		src[i], src[minIndex] = src[minIndex], src[i]
		// log.Println(src[i])
	}
}

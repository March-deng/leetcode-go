package order

import (
	"log"
	"math"
	"sort"
)

/*
出列排序的限制是，考虑一副扑克牌，每次只能查看最上面的两张牌，交换最上面的两张牌或者将最上面的一张牌放到这副牌的最下面。
*/
func DequeueOrder(src []int) {

}

func findMax(src []int) int {
	if len(src) == 0 {
		return 0
	}
	if len(src) == 1 {
		return src[0]
	}

	// first, we take look at the first two numbers
	for i := 1; i < len(src); i++ {
		if src[0] > src[1] {
			//exchange
			temp := src[0]

			src[0] = src[1]
			src[1] = temp
			// ring
			RingArray(src).MoveLeft()
		} else {
			// ring
			RingArray(src).MoveLeft()
		}
	}

	return src[0]
}

type RingArray []int

func (r RingArray) MoveRight() {
	if len(r) == 0 {
		return
	}
	rightValue := r[len(r)-1]

	// move right
	for i := len(r) - 1; i > 0; i-- {
		r[i] = r[i-1]
	}

	r[0] = rightValue
}

func (r RingArray) MoveLeft() {
	if len(r) == 0 {
		return
	}
	leftValue := r[0]

	for i := 1; i < len(r); i++ {
		r[i-1] = r[i]
	}
	r[len(r)-1] = leftValue

	log.Println(r)
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	length := len(nums1) + len(nums2)
	var (
		leftValue  int
		rightValue int
		count      int
		nums1Index int = -1
		nums2Index int = -1
		start      int
		end        int
		nums1Value int = math.MinInt32
		nums2Value int = math.MinInt32
	)

	if length%2 == 0 {
		start = length / 2
		end = start + 1
	} else {
		start = length/2 + 1
		end = start
	}

	// log.Println(start, end)

	// 双指针遍历
	for {

		if len(nums1) > nums1Index+1 {
			nums1Value = nums1[nums1Index+1]
		}

		if len(nums2) > nums2Index+1 {
			nums2Value = nums2[nums2Index+1]
		}

		if count == end {
			break
		}

		// log.Printf("count: %d, numsIndex: %d, nums2Index: %d", count, nums1Index, nums2Index)

		// 双指针移动

		// 此时左边的值偏小，且未到达终点，左边前进
		if nums1Value < nums2Value && len(nums1) > nums1Index+1 {
			nums1Index++
			count++
			if count == start {
				leftValue = nums1[nums1Index]
			}
			if count == end {
				rightValue = nums1[nums1Index]
			}
			continue
		}

		// 此时右边的值偏小，且未达到终点，右边前进
		if nums1Value >= nums2Value && len(nums2) > nums2Index+1 {
			nums2Index++
			count++
			if count == start {
				leftValue = nums2[nums2Index]
			}
			if count == end {
				rightValue = nums2[nums2Index]
			}
			continue
		}

		// 左边已经全部遍历完，右边遍历
		if len(nums1) == nums1Index+1 {
			nums2Index++
			count++
			if count == start {
				leftValue = nums2[nums2Index]
			}
			if count == end {
				rightValue = nums2[nums2Index]
			}
			continue
		}

		// 右边遍历完，左边遍历
		if len(nums2) == nums2Index+1 {
			nums1Index++
			count++
			if count == start {
				leftValue = nums1[nums1Index]
			}
			if count == end {
				rightValue = nums1[nums1Index]
			}
			continue
		}

	}

	return (float64(leftValue) + float64(rightValue)) / 2
}

func smallestRangeI(nums []int, k int) int {
	if len(nums) < 2 {
		return 0
	}
	sort.Ints(nums)

	min := nums[0]
	max := nums[len(nums)-1]

	originDiff := max - min

	diff := originDiff - 2*k

	if diff > 0 {
		return diff
	} else {
		return 0
	}

}

func findMinArrowShots(points [][]int) int {
	sort.Sort(pointsSorter(points))
	// log.Println(points)
	var count int

	lastRange := make([]int, 0, 2)
	for i := 0; i < len(points); i++ {
		start, end, overlap := getRange(points[i], lastRange)
		if i == 0 {
			lastRange = append(lastRange, 0, 0)
		}
		lastRange[0] = start
		lastRange[1] = end
		if !overlap {
			count++
		}
		// log.Println("last range:", lastRange)
	}
	return count + 1
}

func getRange(cur, pre []int) (int, int, bool) {
	// log.Printf("cur: %v, pre: %v", cur, pre)
	if len(pre) == 0 {
		return cur[0], cur[1], true
	}
	if cur[0] > pre[1] {
		return cur[0], cur[1], false
	}

	start := cur[0]

	var end int

	if pre[1] < cur[1] {
		end = pre[1]
	} else {
		end = cur[1]
	}

	return start, end, true
}

type pointsSorter [][]int

func (p pointsSorter) Len() int {
	return len(p)
}

func (p pointsSorter) Less(i, j int) bool {
	if p[i][0] != p[j][0] {
		return p[i][0] < p[j][0]
	}

	return p[i][1] < p[j][1]
}

func (p pointsSorter) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

package order

import (
	"fmt"
	"log"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// 生成一个长度为1000的int数组
func getRandomIntArray1000(t *testing.T) []int {
	rand.Seed(time.Now().Unix())

	src := make([]int, 1000)

	for i := 0; i < 1000; i++ {
		src[i] = rand.Int() % 10000000
	}
	return src
}

func compareSort(std, sorted []int) bool {

	if len(std) != len(sorted) {
		return false
	}

	for i := 0; i < len(std); i++ {
		if std[i] != sorted[i] {
			log.Printf("index %d not ordered!, expect %d, got: %d", i, std[i], sorted[i])
			return false
		}
	}

	return true
}

func copyInts(src []int) []int {
	output := make([]int, len(src))

	copy(output, src)
	return output
}

func TestSelectOrderInt(t *testing.T) {
	src := getRandomIntArray1000(t)

	copied := copyInts(src)

	SelectOrderInt(copied)

	sort.Ints(src)

	// printInts(src)
	// printInts(copied)

	if !compareSort(src, copied) {
		t.Fatal("select not ordered!")
	}
}

func printInts(src []int) {
	for i := 0; i < len(src); i++ {
		fmt.Printf("%d ", src[i])
	}
}

func TestInsertOrderInt(t *testing.T) {
	src := getRandomIntArray1000(t)

	copied := copyInts(src)

	InsertOrderInt(copied)

	sort.Ints(src)

	// printInts(src)
	// printInts(copied)

	if !compareSort(src, copied) {
		t.Fatal("select not ordered!")
	}
}

func TestShellOrderInt(t *testing.T) {
	src := getRandomIntArray1000(t)

	copied := copyInts(src)

	ShellOrderInt(copied)

	sort.Ints(src)

	// printInts(src)
	printInts(copied)

	if !compareSort(src, copied) {
		t.Fatal("select not ordered!")
	}
}

func TestFindMax(t *testing.T) {
	src := []int{4, 3, 0, 8, 7, 9}
	max := findMax(src)
	log.Println(max)
}

func TestFindMedianSortedArrays(t *testing.T) {
	log.Println(findMedianSortedArrays([]int{1, 3}, []int{2}))
	log.Println(findMedianSortedArrays([]int{1, 2}, []int{3, 4}))

	log.Println(findMedianSortedArrays([]int{1, 2}, nil))

	log.Println(findMedianSortedArrays([]int{3, 4}, []int{1, 2}))

	log.Println(findMedianSortedArrays([]int{1, 3}, []int{2, 4}))
}

func TestSmallestRangeI(t *testing.T) {
	log.Println(smallestRangeI([]int{0, 10}, 2))
	log.Println(smallestRangeI([]int{1, 3, 6}, 3))
	log.Println(smallestRangeI([]int{1, 3, 6}, 2))
}

func TestFindMinArrowShots(t *testing.T) {
	log.Println(findMinArrowShots([][]int{{10, 16}, {2, 8}, {1, 6}, {7, 12}}))
	log.Println(findMinArrowShots([][]int{{1, 2}, {3, 4}, {5, 6}, {7, 8}}))
	log.Println(findMinArrowShots([][]int{{1, 2}, {2, 3}, {3, 4}, {4, 5}}))

	log.Println(findMinArrowShots([][]int{{9, 12}, {1, 10}, {4, 11}, {8, 12}, {3, 9}, {6, 9}, {6, 7}}))
}

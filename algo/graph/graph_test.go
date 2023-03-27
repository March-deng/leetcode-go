package graph

import (
	"log"
	"testing"
)

func TestMinMutation(t *testing.T) {
	log.Println(minMutation("AACCGGTT", "AAACGGTA", []string{"AACCGGTA", "AACCGCTA", "AAACGGTA"}))
}

func TestArrayNesting(t *testing.T) {
	log.Println(arrayNesting([]int{5, 4, 0, 3, 1, 6, 2}))
}

func TestSumDigit(t *testing.T) {
	log.Println(sumDigit(35))
	log.Println(sumDigit(34))
	log.Println(sumDigit(199))
}

func TestMovingCount(t *testing.T) {
	log.Println(movingCount(3, 1, 0))
}

func TestPermutatation(t *testing.T) {
	log.Println(permutation("abc"))
}

func TestCheckValidGrid(t *testing.T) {
	grid := [][]int{
		{0, 11, 16, 5, 20},
		{17, 4, 19, 10, 15},
		{12, 1, 8, 21, 6},
		{3, 18, 23, 14, 9},
		{24, 13, 2, 7, 22},
	}

	log.Println(checkValidGrid(grid))

	log.Println(checkValidGrid([][]int{
		{0, 3, 6},
		{5, 8, 1},
		{2, 7, 4},
	}))
}

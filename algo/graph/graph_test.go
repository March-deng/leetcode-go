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

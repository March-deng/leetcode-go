package dp

import (
	"log"
	"testing"
)

func TestFib(t *testing.T) {
	log.Println(fib(0))
	log.Println(fib(1))
	log.Println(fib(2))
	log.Println(fib(45))
	log.Println(fib(46))
}

func TestUniquePaths(t *testing.T) {
	log.Println(uniquePaths(7, 3))
}

func TestBasicPackage(t *testing.T) {
	log.Println(basicPackage(3, 4, []int{1, 3, 4}, []int{15, 20, 30}))
}

func TestCanPartion(t *testing.T) {
	log.Println(canPartition([]int{1, 5, 11, 5}))
}

func TestWordBreak(t *testing.T) {
	log.Println(wordBreak("leetcode", []string{"leet", "code"}))

}

func TestRobTree(t *testing.T) {
	log.Println(robNode(constructTree()))
}

func constructTree() *TreeNode {
	return &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 2,
			Right: &TreeNode{
				Val: 3,
			},
		},
		Right: &TreeNode{
			Val: 3,
			Right: &TreeNode{
				Val: 1,
			},
		},
	}
}

func TestMinPathSum(t *testing.T) {
	log.Println(minPathSum([][]int{
		{1, 3, 1},
		{1, 5, 1},
		{4, 2, 1},
	}) == 7)
	log.Println(minPathSum([][]int{
		{1, 2, 3},
		{4, 5, 6},
	}) == 12)
}

func TestCountSubStrings(t *testing.T) {
	log.Println(countSubstrings("aaa"))
}

func TestTranslateNum(t *testing.T) {
	log.Println(translateNum(12258))
}

func TestLengthOfLongestSubstring(t *testing.T) {
	log.Println(lengthOfLongestSubstring("abcabcabc"))
}

func TestMaxProfitSellStock(t *testing.T) {
	log.Println(maxProfitSellStock([]int{1, 2}))
}

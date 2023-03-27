package array

import (
	"container/heap"
	"fmt"
	"log"
	"sort"
	"testing"
)

func TestIsPalindrome(t *testing.T) {
	// log.Println(isPalindrome("A man, a plan, a canal: Panama"))
	// log.Println(isPalindrome("race a car"))
	log.Println(isPalindrome("0P"))
	// log.Println(isPalindrome("aba"))

}

func TestFindTheDistanceValue(t *testing.T) {
	// log.Println(findTheDistanceValue([]int{2, 1, 100, 3}, []int{-5, -2, 10, -3, 7}, 6))
	log.Println(findTheDistanceValue([]int{4, 5, 8}, []int{10, 9, 1, 8}, 2))
	log.Println(findTheDistanceValue([]int{2, 1, 100, 3}, []int{-5, -2, 10, 3, -7}, 6))
}

func TestPlusOne(t *testing.T) {
	log.Println(plusOne([]int{1, 2, 3}))

	log.Println(plusOne([]int{9}))
}

func TestMoveZeros(t *testing.T) {
	moveZeroes([]int{0, 1, 0, 3, 12})
	moveZeroes([]int{0, 1})
}

func TestFirstUniqChart(t *testing.T) {
	log.Println(firstUniqChar("aabb"))
}

func TestKWeakestRows(t *testing.T) {
	log.Println(kWeakestRows([][]int{
		{1, 1, 0, 0, 0},
		{1, 1, 1, 1, 0},
		{1, 0, 0, 0, 0},
		{1, 1, 0, 0, 0},
		{1, 1, 1, 1, 1},
	}, 3))
}

func TestPair(t *testing.T) {
	h := hp{}

	h = append(h, pair{pow: 1, idx: 2}, pair{pow: 2, idx: 3}, pair{pow: 5, idx: 4}, pair{pow: 2, idx: 0}, pair{pow: 4, idx: 1})
	heap.Init(&h)

	for i := 0; i < 5; i++ {
		p := heap.Pop(&h).(pair)
		log.Println(p.idx, p.pow)
	}
}

func TestCheckExist(t *testing.T) {
	log.Println(checkIfExist([]int{1, 3, 7, 11}))

	// log.Println(checkNumIfExist([]int{1, 3, 7, 11}, 3, 4, 14))
}

func TestRemoveAnagrams(t *testing.T) {

	// log.Println(removeAnagrams([]string{"abba", "baba", "bbaa", "cd", "ef"}))
	log.Println(removeAnagrams([]string{"z", "z", "z", "gsw", "wsg", "gsw", "krptu"}))
}

func TestFindKthNumber(t *testing.T) {
	log.Println(findKthNumber(3, 3, 5) == 3)
}

func TestCollectFlower(t *testing.T) {

	// log.Println(collect([]int{7, 7, 7, 7, 12, 7, 7}, 10, 2, 3))

	log.Println(minDays([]int{7, 7, 7, 7, 12, 7, 7}, 2, 3))
}
func TestMinAbsoluteSumDiff(t *testing.T) {
	log.Println(minAbsoluteSumDiff([]int{1, 10, 4, 4, 2, 7}, []int{9, 3, 5, 1, 7, 4}))
}

func TestRemoveOuterParentheses(t *testing.T) {
	type tuple struct {
		input  string
		expect string
	}

	tuples := []tuple{
		{
			input:  "(()())(())",
			expect: "()()()",
		},
		{
			input:  "(()())(())(()(()))",
			expect: "()()()()(())",
		},
		{
			input:  "()()",
			expect: "",
		},
	}

	for i, tuple := range tuples {
		if got := removeOuterParentheses(tuple.input); got != tuple.expect {
			t.Logf("tuple %d failed, input: %s, expect: %s, got: %s", i+1, tuple.input, tuple.expect, got)
		}
	}
}

func TestMaxFrequency(t *testing.T) {
	type tuple struct {
		nums   []int
		k      int
		expect int
	}

	tuples := []tuple{
		// {
		// 	nums:   []int{1, 2, 4},
		// 	k:      5,
		// 	expect: 3,
		// },
		// {
		// 	nums:   []int{1, 4, 8, 13},
		// 	k:      5,
		// 	expect: 2,
		// },
		{
			nums:   []int{3, 9, 6},
			k:      2,
			expect: 1,
		},
	}

	for i, tuple := range tuples {
		if got := maxFrequency(tuple.nums, tuple.k); got != tuple.expect {
			t.Logf("tuple %d failed, input nums: %v,input k: %d, expect: %d, got: %d", i+1, tuple.nums, tuple.k, tuple.expect, got)
		}
	}
}

func TestGetLeftAndRight(t *testing.T) {
	// log.Println(waysToSplit([]int{1, 2, 2, 2, 5, 0}))

	log.Println(waysToSplit([]int{0, 0, 0, 0, 0, 0}))
}

func TestValidIPAddress(t *testing.T) {
	log.Println(validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334"))
}

func TestGenerateParenthesis(t *testing.T) {
	result := generateParenthesis(8)

	resultCopy := generateParenthesisCopy(8)

	set := make(map[string]struct{})

	for _, v := range result {
		set[v] = struct{}{}
	}

	for _, v := range resultCopy {
		_, ok := set[v]
		if !ok {
			log.Println("not found: ", v)
		}
	}

}

func TestAddParenthesis(t *testing.T) {
	result := addOneParenthesis("()")

	for _, v := range result {
		log.Println(v)

	}

}

func TestReverseWords(t *testing.T) {
	type tuple struct {
		input  string
		expect string
	}

	tuples := []tuple{
		{
			input:  "Let's take LeetCode contest",
			expect: "s'teL ekat edoCteeL tsetnoc",
		},
		{
			input:  "God Ding",
			expect: "doG gniD",
		},
		{
			input:  "Leetcode",
			expect: "edocteeL",
		},
		{
			input:  "I love u",
			expect: "I evol u",
		},
	}

	for i, tuple := range tuples {
		if got := reverseWords(tuple.input); got != tuple.expect {
			log.Printf("tuple %d failed, input: %s, expect: %s, got: %s", i, tuple.input, tuple.expect, got)
		}
	}
}

func TestMaxValue(t *testing.T) {

	type input struct {
		n      int
		index  int
		maxSum int
	}
	type tuple struct {
		input  input
		expect int
	}

	tuples := []tuple{
		{
			input: input{
				n:      4,
				index:  2,
				maxSum: 6,
			},
			expect: 2,
		},
		{
			input: input{
				n:      6,
				index:  1,
				maxSum: 10,
			},
			expect: 3,
		},
		{
			input: input{
				n:      8,
				index:  7,
				maxSum: 14,
			},
			expect: 4,
		},
		{
			input: input{
				n:      1,
				index:  0,
				maxSum: 24,
			},
			expect: 24,
		},
	}

	for i, tuple := range tuples {
		if got := maxValue(tuple.input.n, tuple.input.index, tuple.input.maxSum); got != tuple.expect {
			log.Printf("tuple %d failed, input: %v, expect: %d, got: %d", i, tuple.input, tuple.expect, got)
		}
	}
}

func TestTwoSumLessThanK(t *testing.T) {
	log.Println(twoSumLessThanK([]int{34, 23, 1, 24, 75, 33, 54, 8}, 60))
}

func TestSearchInts(t *testing.T) {
	log.Println(searchInts([]int{34, 23, 1, 24, 75, 33, 54, 8}, 6))
}

func TestAddBinary(t *testing.T) {
	log.Println(addBinary("110", "1"))
}

func TestFindAndReplacePattern(t *testing.T) {
	log.Println(findAndReplacePattern([]string{"abc", "deq", "mee", "aqq", "dkd", "ccc"}, "abb"))
}

func TestIncreasingTriplet(t *testing.T) {
	// log.Println(increasingTriplet([]int{1, 2, 3, 4, 5, 6}))

	// log.Println(increasingTriplet([]int{5, 4, 3, 2, 1}))
	log.Println(increasingTriplet([]int{2, 1, 5, 4, 0, 6}))
}

func TestDuplicateZeros(t *testing.T) {
	duplicateZeros([]int{1, 0, 2, 3, 0, 4, 5, 0})
	duplicateZeros([]int{0, 0, 0})

	duplicateZeros([]int{8, 4, 5, 0, 0, 0, 0, 7})
}

func TestCountPairs(t *testing.T) {
	log.Println(countPairs([]int{2, 1, 2, 1}, []int{1, 2, 1, 2}))
}

func TestCirqularQueue(t *testing.T) {
	q := Constructor(3)
	m := &q

	log.Println(m.EnQueue(1))
	log.Println(m.EnQueue(2))
	log.Println(m.EnQueue(3))
	log.Println(m.EnQueue(4))

	log.Println(m.Rear())

}

func TestReversePairs(t *testing.T) {
	log.Println(reversePairs([]int{7, 5, 6, 4}))
}

func TestTwoSum11(t *testing.T) {
	log.Println(twoSum([]int{2, 7, 11, 15}, 9))
}

func TestReverseEveryWord(t *testing.T) {
	// log.Println(reverseByWord("the sky is blue"))
	// log.Println(reverseByWord("  hello world  "))
	// log.Println(reverseByWord("a good   example"))
	log.Println(reverseByWord(""))
}

func TestMissingNumber(t *testing.T) {
	log.Println(missingNumber([]int{0, 1, 2}))
	log.Println(missingNumber([]int{0, 1, 3}))
	log.Println(missingNumber([]int{0, 1, 2, 3, 4, 5, 6, 7, 9}))

}

func TestSearchRange(t *testing.T) {
	log.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 8))
}

func TestCuttingRope(t *testing.T) {
	log.Println(cuttingRope(2))
	log.Println(cuttingRope(4))
	log.Println(cuttingRope(10))
}

func TestCountDigitOne(t *testing.T) {
	log.Println(countDigitOne(101))
}

func TestMovesToMakeZigzag(t *testing.T) {

	log.Println(movesToMakeZigzag([]int{1, 2, 3}))
	log.Println(movesToMakeZigzag([]int{9, 6, 1, 6, 2}))

	log.Println(movesToMakeZigzag([]int{2, 7, 10, 9, 8, 9}))
}

func TestMergeSimilarItems(t *testing.T) {
	log.Println(mergeSimilarItems([][]int{{1, 1}, {4, 5}, {3, 8}}, [][]int{{3, 1}, {1, 5}}))
}

func TestGetForlderName(t *testing.T) {
	log.Println(getFolderNames([]string{"pes", "fifa", "gta", "pes(2019)"}))
	log.Println(getFolderNames([]string{"gta", "gta(1)", "gta", "avalon"}))
	log.Println(getFolderNames([]string{"onepiece", "onepiece(1)", "onepiece(2)", "onepiece(3)", "onepiece"}))
	log.Println(getFolderNames([]string{"wano", "wano", "wano", "wano"}))
	log.Println(getFolderNames([]string{"kaido", "kaido(1)", "kaido", "kaido(1)"}))
}
func TestSumNum(t *testing.T) {
	log.Println(sumNums(10))
}

func TestValidateStackSequence(t *testing.T) {
	log.Println(validateStackSequences([]int{1, 2, 3, 4, 5}, []int{4, 5, 3, 2, 1}))
	log.Println(validateStackSequences([]int{1, 2, 3, 4, 5}, []int{5, 4, 3, 2, 1}))
	log.Println(validateStackSequences([]int{1, 2, 3, 4, 5}, []int{4, 3, 5, 1, 2}))
}

func TestFindContinuousSequence(t *testing.T) {
	log.Println(findContinuousSequence(16))
}

func TestFindLongestSubarray(t *testing.T) {
	// log.Println(findLongestSubarray([]string{"A", "1"}))
	log.Println(findLongestSubarray([]string{"A", "1", "B", "C", "D", "2", "3", "4", "E", "5", "F", "G", "6", "7", "H", "I", "J", "K", "L", "M"}))
}

func TestFindDisappearedNumbers(t *testing.T) {
	log.Println(findDisappearedNumbers([]int{4, 3, 2, 7, 8, 2, 3, 1}))
	log.Println(findDisappearedNumbers([]int{1, 1}))
}

func TestConstructArray(t *testing.T) {
	res := constructArr([]int{1, 2, 3, 4, 5})
	log.Println(res)
}

func TestNextPermutation(t *testing.T) {
	nums := []int{3, 2, 1}
	nextPermutation(nums)
	log.Println(nums)
}

func TestSwap(t *testing.T) {
	nums := []int{1, 2, 3, 4}
	swap(nums)
	log.Println(nums)
}

func TestDecodeString(t *testing.T) {
	log.Println(decodeString("3[a]2[bc]") == "aaabcbc")
	log.Println(decodeString("3[a2[c]]") == "accaccacc")
	log.Println(decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef")
	// log.Println(decodeString("100[leetcode]"))
}

func TestArrayInFly(t *testing.T) {
	a := []int{0, 1, 2}

	for i, v := range a {
		a[2] = 10
		if i == 2 {
			fmt.Println(v)
		}
	}
}

func TestSortSubArrayMerge(t *testing.T) {
	var s SubArrayMerge = [][]int{{1, 4}, {2, 3}}

	sort.Sort(s)

	log.Println(s)

	log.Println(overlap(s[0], s[1]))
}

func TestMergeSubArray(t *testing.T) {
	s := [][]int{{1, 4}, {2, 3}}

	res := mergeSubArray(s)

	log.Println(res)
}

func TestSubArraySub(t *testing.T) {
	log.Println(subarraySum([]int{1, 1, 1}, 2))
}

func TestFindUnsortedSubarray(t *testing.T) {
	// fmt.Println(findUnsortedSubarray([]int{2, 6, 4, 8, 10, 9, 15}))
	// log.Println(findUnsortedSubarray([]int{1, 3, 2, 2, 2}))
	log.Println(findUnsortedSubarray([]int{1, 2, 3, 4}))
}

func TestEvenOddBit(t *testing.T) {
	log.Println(evenOddBit(2))
}

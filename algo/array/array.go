package array

import (
	"container/heap"
	"log"
	"math"
	"math/rand"
	"net"
	"sort"
	"strconv"
	"strings"
	"unsafe"
)

// 双指针
func isPalindrome(s string) bool {
	i, j := 0, len(s)-1
	for i < j {
		chari := uint8(s[i])

		if chari < 48 || (chari > 57 && chari < 65) || (chari > 90 && chari < 97) || chari > 122 {
			// 忽略
			i++
			continue
		}

		charj := uint8(s[j])
		if charj < 48 || (charj > 57 && charj < 65) || (charj > 90 && charj < 97) || charj > 122 {
			// 忽略
			j--
			continue
		}

		// 大写转小写
		if chari >= 65 && chari <= 90 {
			chari += 32
		}

		if charj >= 65 && charj <= 90 {
			charj += 32
		}

		if chari != charj {
			return false
		}

		i++
		j--
	}
	return true
}

func findDuplicates(nums []int) []int {
	elements := make(map[int]struct{})
	result := make([]int, 0)

	for _, v := range nums {
		_, ok := elements[v]
		if ok {
			result = append(result, v)
		}
		elements[v] = struct{}{}
	}
	return result
}

func isPerfectSquare(num int) bool {
	if num == 1 {
		return true
	}

	if num == 2 {
		return false
	}

	i, j := 1, num/2+1
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h

		square := h * h
		if square == num {
			return true
		}

		if square < num {
			i = h + 1
		} else {
			j = h
		}
	}

	return false
}

func findTheDistanceValue(arr1 []int, arr2 []int, d int) int {
	// 对两个数组进行排序

	sort.Ints(arr2)

	var count int

	for _, v := range arr1 {
		left, right, leftOk, rightOK := searchInBetween(arr2, v)

		if leftOk && !rightOK {
			if abs(left-v) > d {
				count++
			}
			continue
		}

		if !leftOk && rightOK {
			if abs(right-v) > d {
				count++
			}
			continue
		}

		if abs(right-v) > d && abs(left-v) > d {

			count++
		}
	}
	return count
}

// 第一个返回值小于等于target
// 第二个返回值大于等于target
func searchInBetween(nums []int, target int) (int, int, bool, bool) {
	i, j := 0, len(nums)

	for i < j {

		h := int(uint(i+j) >> 1)

		if nums[h] == target {
			i = h
			break
		}

		if nums[h] > target {
			j = h
		} else {
			i = h + 1
		}
	}

	// 说明target比nums中的任何元素都要大
	if i == len(nums) {
		return nums[i-1], 0, true, false
	}

	// 说明target比nums中的任何元素都要小
	if i == 0 {
		return 0, nums[0], false, true
	}

	return nums[i-1], nums[i], true, true
}

func findTargetSumWays(nums []int, target int) int {
	// 求和
	var sum int
	for _, v := range nums {
		sum += v
	}

	if target > sum {
		return 0
	}

	if (target+sum)%2 != 0 {
		return 0
	}

	partial := (target + sum) / 2
	dpTables := make([]int, partial+1)
	dpTables[0] = 1

	for i := 0; i < len(nums); i++ {
		for j := partial; j >= nums[i]; j-- {
			dpTables[j] += dpTables[j-nums[i]]
		}
	}

	return dpTables[partial]
}

func findMaxForm(strs []string, m int, n int) int {
	if len(strs) == 0 {
		return 0
	}

	dpTables := make([][]int, m+1)

	for i := range dpTables {
		dpTables[i] = make([]int, n+1)
	}

	for _, str := range strs {
		var zeroCount, oneCount int

		for _, char := range str {
			if char == '0' {
				zeroCount++
			} else {
				oneCount++
			}
		}

		for i := m; i >= zeroCount; i-- {
			for j := n; j >= oneCount; j-- {
				dpTables[i][j] = max(dpTables[i][j], dpTables[i-zeroCount][j-oneCount]+1)
			}
		}
	}

	return dpTables[m][n]
}

func myPow(x float64, n int) float64 {
	return math.Pow(x, float64(n))
}

func plusOne(digits []int) []int {
	length := len(digits) - 1
	digits[length]++

	if digits[length] < 10 {
		return digits
	}
	// 模拟进位
	for i := length; i > 0; i-- {
		if digits[i] >= 10 {
			digits[i] = 0
			digits[i-1]++
		}
	}

	if digits[0] == 10 {
		digits = append(digits, 0)
		digits[0] = 1
	}

	return digits
}

func moveZeroes(nums []int) {
	if len(nums) < 2 {
		return
	}
	i, j := 0, 0

	for j < len(nums) {
		// 第一个为0的数
		if nums[i] != 0 {
			i++
			j++
			continue
		}
		// 找到第一个不为0的位置
		if nums[j] == 0 {
			j++
			continue
		}

		nums[i], nums[j] = nums[j], nums[i]
	}

	// log.Println(nums)
}

func firstUniqCharIdx(s string) int {
	set := make([]int, 26)

	for i, char := range s {
		index := int(char - 97)
		if set[index] == 0 {
			set[index] = i + 1
		} else if set[index] > 0 {
			set[index] = -1
		}
	}

	var index int = 1<<32 - 1
	var found bool

	for _, v := range set {
		if v > 0 && v < index {
			found = true
			index = v
		}
	}

	if !found {
		return -1
	}

	return index - 1
}

func canConstruct(ransomNote string, magazine string) bool {
	set := make([]int, 26)

	for _, char := range magazine {
		set[char-97]++
	}

	for _, char := range ransomNote {

		set[char-97]--
		if set[char-97] == -1 {
			return false
		}
	}
	return true
}
func kWeakestRows(mat [][]int, k int) []int {
	h := hp{}
	for i, row := range mat {
		pow := sort.Search(len(row), func(j int) bool { return row[j] == 0 })
		log.Println(pow, i)
		h = append(h, pair{pow, i})
	}
	heap.Init(&h)
	ans := make([]int, k)
	for i := range ans {
		p := heap.Pop(&h).(pair)
		// log.Println(p.idx, p.pow)

		ans[i] = p.idx
	}
	return ans
}

type pair struct{ pow, idx int }
type hp []pair

func (h hp) Len() int { return len(h) }
func (h hp) Less(i, j int) bool {
	a, b := h[i], h[j]
	return a.pow < b.pow || a.pow == b.pow && a.idx < b.idx
}
func (h hp) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v interface{}) { *h = append(*h, v.(pair)) }
func (h *hp) Pop() interface{}   { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }

func checkIfExist(arr []int) bool {
	sort.Ints(arr)

	for i := 0; i < len(arr); i++ {
		if arr[i] < 0 {
			if checkNumIfExist(arr, 0, i, 2*arr[i]) {
				return true
			}
		} else {
			if checkNumIfExist(arr, i+1, len(arr), 2*arr[i]) {
				return true
			}
		}

	}
	return false
}

func checkNumIfExist(nums []int, i, j int, target int) bool {
	if i >= len(nums) {
		return false
	}
	for i < j {
		h := int(uint(i+j) >> 1)

		if nums[h] == target {
			return true
		}

		if nums[h] > target {
			j = h
		} else {
			i = h + 1
		}
	}
	if i >= len(nums) {
		return false
	}

	if nums[i] == target {
		return true
	}

	return false
}

func judgeSquareSum(c int) bool {
	if c <= 0 {
		return false
	}

	i, j := 0, int(math.Sqrt(float64(c)))

	for i <= j {
		sum := i*i + j*j

		if sum == c {
			return true
		}

		if sum < c {
			i++
		} else {
			j--
		}
	}
	return false
}

func removeAnagrams(words []string) []string {
	sort.Sort(anagramSorter(words))

	result := make([]string, 0)

	for i := 0; i < len(words); {
		result = append(result, words[i])
		var j int
		for j = i; j < len(words); j++ {
			if !isAnagram(words[i], words[j]) {
				break
			}
		}

		i = j
	}

	return result
}

type anagramSorter []string

func (a anagramSorter) Len() int {
	return len(a)
}

func (a anagramSorter) Less(i, j int) bool {

	if len(a[i]) == len(a[j]) {
		return a[i] < a[j]
	}

	return len(a[i]) < len(a[j])
}

func (a anagramSorter) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func isAnagram(s, t string) bool {
	if s == t {
		return true
	}
	var c1, c2 [26]int
	for _, ch := range s {
		c1[ch-'a']++
	}
	for _, ch := range t {
		c2[ch-'a']++
	}
	return c1 == c2
}

func maxConsecutive(bottom int, top int, special []int) int {
	sort.Ints(special)

	var (
		count  int
		bot    int = bottom
		result int
	)

	for _, v := range special {
		count = v - bot
		// 更新最低的楼层
		bot = v + 1
		result = max(result, count)
	}

	if top > special[len(special)-1] {
		result = max(result, top-special[len(special)]-1)
	}

	return result
}

func findKthNumber(m int, n int, k int) int {
	left, right := 0, m*n

	for left < right {
		mid := left + (right-left)/2

		count := getCount(m, n, mid)

		if count >= k {
			right = mid
		} else {
			left = mid + 1
		}

	}

	log.Println(left)
	return left
}

func getCount(m, n, x int) int {
	// 整行的部分
	count := (x / n) * n

	for i := x/n + 1; i <= m; i++ {
		count += x / i
	}

	return count

}

func minDays(bloomDay []int, m int, k int) int {
	if len(bloomDay) < m*k {
		return -1
	}

	// 找最大值
	var maxDay int
	for _, v := range bloomDay {
		if v > maxDay {
			maxDay = v
		}
	}

	i, j := 0, maxDay+1

	for i+1 != j {
		h := i + (j-i)/2

		if collect(bloomDay, h, m, k) {
			j = h
		} else {
			i = h
		}
	}
	return j
}

func collect(bloomDay []int, day int, m, k int) bool {
	var (
		row      int
		bouquets int
	)
	for _, v := range bloomDay {
		if v <= day {
			row++
		} else {
			row = 0
		}
		if row == k {
			bouquets++
			row = 0
		}
	}

	return bouquets >= m
}

func minAbsoluteSumDiff(nums1, nums2 []int) int {
	elements := make(sort.IntSlice, len(nums1))
	copy(elements, nums1)

	sort.Sort(elements)

	sum, maxn, n := 0, 0, len(nums1)
	for i, v := range nums2 {
		diff := abs(nums1[i] - v)
		sum += diff
		j := elements.Search(v)
		if j < n {
			maxn = max(maxn, diff-(elements[j]-v))
		}
		if j > 0 {
			maxn = max(maxn, diff-(v-elements[j-1]))
		}
	}
	return (sum - maxn) % (1e9 + 7)
}

func merge(nums1 []int, m int, nums2 []int, n int) {
	var (
		i int = m - 1
		j int = n - 1
		k int = m + n - 1
	)

	for i >= 0 && j >= 0 {
		// 比较当前元素
		if nums1[i] >= nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
		k--
	}

	for j >= 0 {
		nums1[k] = nums2[j]
		k--
		j--
	}
}

// type MinStack struct {
// 	stack []int
// 	min   int
// }

// func Constructor() MinStack {
// 	return MinStack{
// 		stack: make([]int, 0),
// 		min:   math.MinInt64,
// 	}
// }

// func (this *MinStack) Push(val int) {
// 	if len(this.stack) == 0 {
// 		this.stack = append(this.stack, val)
// 		this.min = val
// 		return
// 	}
// 	value := val - this.min
// 	this.stack = append(this.stack, value)

// 	if val < this.min {
// 		this.min = val
// 	}
// }

// // 更新最小值
// func (this *MinStack) Pop() {
// 	if len(this.stack) == 0 {
// 		return
// 	}
// 	val := this.stack[len(this.stack)-1]
// 	this.stack = this.stack[:len(this.stack)-1]

// 	// 不需要更新最小值
// 	if val < 0 {
// 		this.min -= val
// 	}
// }

// func (this *MinStack) Top() int {
// 	if len(this.stack) == 0 {
// 		return 0
// 	}

// 	val := this.stack[len(this.stack)-1]
// 	if len(this.stack) == 1 {
// 		return val
// 	}

// 	if val < 0 {
// 		return this.min
// 	}
// 	return this.min + val
// }

// func (this *MinStack) GetMin() int {
// 	return this.min
// }

func fizzBuzz(n int) (ans []string) {
	buffer := &strings.Builder{}
	for i := 1; i <= n; i++ {

		if i%3 == 0 {
			buffer.WriteString("Fizz")
		}
		if i%5 == 0 {
			buffer.WriteString("Buzz")
		}
		if buffer.Len() != 0 {
			ans = append(ans, buffer.String())
		} else {
			ans = append(ans, strconv.Itoa(i))
		}
		buffer.Reset()
	}
	return
}

func removeOuterParentheses(s string) string {
	builder := &strings.Builder{}

	var level int

	for _, c := range s {
		if c == '(' {
			if level == 0 {
				level++
				continue
			}
			level++
		}

		if c == ')' {
			level--
		}

		if level >= 1 {
			builder.WriteRune(c)
		}
	}

	return builder.String()
}

func maxFrequency(nums []int, k int) int {
	sort.Ints(nums)
	var result int = 1
	for l, r, total := 0, 1, 0; r < len(nums); r++ {
		total += (nums[r] - nums[r-1]) * (r - l)
		for total > k {
			total -= nums[r] - nums[l]
			l++
		}
		result = max(result, r-l+1)
	}
	return result
}

func singleNonDuplicate(nums []int) int {
	i, j := 0, len(nums)-1

	for i < j {
		h := i + (j-i)/2

		// 如果有重复数的话，就移动上下届
		if nums[h] == nums[h^1] {
			i = h + 1
		} else {
			j = h
		}
	}

	return nums[i]
}

func waysToSplit(nums []int) int {
	// 计算前缀和
	for i := 1; i < len(nums); i++ {
		nums[i] += nums[i-1]
	}

	var count int

	for i := 0; i < len(nums)-2; i++ {

		// 获取右划分点的下界
		left := getLeft(nums[i+1:], nums[i])
		left += i + 1

		if left >= len(nums)-1 {
			break
		}

		right := getRight(nums[i+1:], nums[i])
		right += i + 1

		if right < left {
			continue
		}

		// log.Printf("index: %d, left: %d, right: %d", i, left, right)

		count += right - left + 1

	}

	return count % (1e9 + 7)
}

func getLeft(nums []int, target int) int {
	return sort.Search(len(nums), func(i int) bool {
		return nums[i]-target >= target
	})
}

func getRight(nums []int, left int) int {
	index := sort.Search(len(nums), func(i int) bool {
		return nums[len(nums)-1]-nums[i] < nums[i]-left
	})

	if index == len(nums) {
		return index - 2
	}

	return index - 1
}

func validIPAddress(queryIP string) string {

	if strings.Contains(queryIP, ".") {
		if strings.Count(queryIP, ".") != 3 {
			return "Neither"
		}
		ip := net.ParseIP(queryIP)

		if ip.To4() != nil {
			return "IPv4"
		}
		return "Neither"
	}

	if strings.Contains(queryIP, ":") {
		segments := strings.Split(queryIP, ":")
		if len(segments) != 8 {
			return "Neither"
		}

		for _, segment := range segments {
			if len(segment) < 1 || len(segment) > 4 {
				return "Neither"
			}
		}

		ip := net.ParseIP(queryIP)

		if ip != nil {
			return "IPv6"
		}

		return "Neither"
	}

	return "Neither"

}

func generateParenthesis(n int) []string {
	set := make(map[string]struct{})

	prev := make([]string, 1)
	prev[0] = "()"

	for i := 2; i <= n; i++ {
		for _, v := range prev {
			parenthsises := addOneParenthesis(v)

			for _, parentsis := range parenthsises {
				set[parentsis] = struct{}{}
			}
		}

		prev = make([]string, 0, len(set))
		for v := range set {
			prev = append(prev, v)
		}
		set = make(map[string]struct{})
	}

	return prev
}

func addOneParenthesis(v string) []string {
	status := make([]int, len(v))
	result := make([]string, 0)
	builder := &strings.Builder{}
	var (
		level int
	)

	for i, c := range v {
		if level == 0 {
			status[i] = 1
		}
		if c == '(' {
			level++
		}

		if c == ')' {
			level--
		}

		if level == 0 {
			status[i] = -1
		}
	}

	for i := 0; i < len(status); i++ {
		if status[i] == 1 {

			for j := i + 1; j < len(status); j++ {
				if status[j] == -1 {
					builder.WriteString(v[:i])
					builder.WriteRune('(')
					builder.WriteString(v[i : j+1])
					builder.WriteRune(')')

					if j != len(status)-1 {
						builder.WriteString(v[j+1:])
					}
					result = append(result, builder.String())
					builder.Reset()
				}
			}
		}
	}

	result = append(result, "()"+v, v+"()")

	return result

}

func generateParenthesisCopy(n int) []string {
	res := []string{}

	var dfs func(lRemain int, rRemain int, path string)
	dfs = func(lRemain int, rRemain int, path string) {
		if 2*n == len(path) {
			res = append(res, path)
			return
		}
		if lRemain > 0 {
			dfs(lRemain-1, rRemain, path+"(")
		}
		if lRemain < rRemain {
			dfs(lRemain, rRemain-1, path+")")
		}
	}

	dfs(n, n, "")
	return res
}

func hammingDistance(x int, y int) int {
	n := uint(x ^ y)

	var count int

	for i := 0; i < 32; i++ {
		if (n & (1 << i)) > 0 {
			count += 1
		}
	}

	return count
}

// type Solution struct {
// 	original  []int
// 	prefixSum []int
// }

// func Constructor(w []int) Solution {
// 	s := Solution{
// 		original:  w,
// 		prefixSum: make([]int, len(w)),
// 	}

// 	s.prefixSum[0] = w[0]

// 	for i := 1; i < len(w); i++ {
// 		s.prefixSum[i] = s.prefixSum[i-1] + w[i]
// 	}

// 	return s

// }

// func (this *Solution) PickIndex() int {
// 	x := rand.Intn(this.prefixSum[len(this.prefixSum)-1]) + 1

// 	return sort.SearchInts(this.prefixSum, x)
// }

var symbolValues = map[byte]int{'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

func romanToInt(s string) (ans int) {
	n := len(s)
	for i := range s {
		value := symbolValues[s[i]]
		if i < n-1 && value < symbolValues[s[i+1]] {
			ans -= value
		} else {
			ans += value
		}
	}
	return
}

func rotate(matrix [][]int) {
	// 上下翻转
	for i := 0; i < len(matrix)/2; i++ {
		matrix[i], matrix[len(matrix)-i-1] = matrix[len(matrix)-i-1], matrix[i]
	}

	// 对角线翻转
	for i := 0; i < len(matrix); i++ {
		for j := i + 1; j < len(matrix[0]); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

func reverseWords(s string) string {
	raw := []byte(s)
	length := len(raw)

	var start int

	for i := 0; i < length; i++ {
		b := raw[i]

		// 从start到i-1，可以进行翻转
		if b == ' ' {
			// log.Println(start, i)
			sum := start + i - 1
			for j := start; j <= sum/2; j++ {
				raw[j], raw[sum-j] = raw[sum-j], raw[j]
			}
			start = i + 1
			continue
		}

		if i == length-1 {
			// log.Println(start, i)
			sum := start + i
			for j := start; j <= sum/2; j++ {
				raw[j], raw[sum-j] = raw[sum-j], raw[j]
			}
			continue
		}

		if b == ' ' {
			start++
		}

	}

	return Bytes2String(raw)
}

func consecutiveNumbersSum(n int) int {
	var (
		result int
		m      int = n * 2
	)

	for k := 1; k*k < m; k++ {
		if m%k != 0 {
			continue
		}

		if (m/k-(k-1))%2 == 0 {
			result++
		}
	}

	return result
}

func abs(n int) int {
	if n == 0 {
		return 0
	}
	if n < 0 {
		return -n
	}
	return n
}
func findBestValue(arr []int, target int) int {
	sort.Ints(arr)
	n := len(arr)
	prefix := make([]int, n+1)
	for i := 1; i <= n; i++ {
		prefix[i] = prefix[i-1] + arr[i-1]
	}
	l, r, ans := 0, arr[n-1], -1
	for l <= r {
		mid := (l + r) / 2
		index := sort.SearchInts(arr, mid)
		if index < 0 {
			index = -1*index - 1
		}
		cur := prefix[index] + (n-index)*mid
		if cur <= target {
			ans = mid
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	lowerSum := sumByValue(arr, ans)
	upperSum := sumByValue(arr, ans+1)

	if abs(lowerSum-target) > abs(upperSum-target) {
		ans++
	}
	return ans
}

func sumByValue(arr []int, x int) int {
	var sum int
	for _, v := range arr {
		sum += min(v, x)
	}
	return sum
}

type Solution struct {
	radius float64
	x      float64
	y      float64
}

// func Constructor(radius float64, x_center float64, y_center float64) Solution {
// 	return Solution{
// 		radius: radius,
// 		x:      x_center,
// 		y:      y_center,
// 	}
// }

func (this *Solution) RandPoint() []float64 {
	for {
		x := rand.Float64()*2 - 1
		y := rand.Float64()*2 - 1

		if x*x+y*y < 1 {
			return []float64{this.x + x*this.radius, this.y + y*this.radius}
		}
	}
}

func maxValue(n int, index int, maxSum int) int {
	i, j := 0, maxSum+1
	for i+1 != j {
		h := i + (j-i)/2

		sum := getSum(n, index, h)

		// log.Println(h, sum)

		if sum > maxSum {
			j = h
		} else {
			i = h
		}
	}
	return i
}

func getSum(n int, index int, value int) int {

	var (
		leftSum  int
		rightSum int
	)
	leftIndex := index - value

	if leftIndex < 0 {
		// 左边的等差数列和
		leftSum = ((index + 1) * (value - index + value)) / 2
	} else {
		leftSum = (value*value + value) / 2
		leftSum += leftIndex + 1

	}

	// 右边的等差数列和
	rightIndex := index + value
	if rightIndex > n {
		start := rightIndex - n + 1
		rightSum = ((value - start + 1) * (value + start)) / 2
	} else {
		rightSum = (value*value + value) / 2
		rightSum += (n - rightIndex)
	}

	return leftSum + rightSum - value
}

func findPeakGrid(mat [][]int) []int {
	var (
		max int = math.MinInt32
		x   int
		y   int
	)

	for i := range mat {
		for j := range mat[0] {
			if mat[i][j] > max {
				max = mat[i][j]
				x = i
				y = j
			}
		}
	}

	return []int{x, y}
}

func twoSumLessThanK(nums []int, k int) int {
	sort.Ints(nums)

	var result int = -1

	for i := 0; i < len(nums)-1; i++ {
		if nums[i] >= k {
			break
		}

		target := k - nums[i]
		idx := searchInts(nums[i+1:], target)
		if idx == -1 {
			break
		}

		index := idx + i + 1

		result = max(result, nums[i]+nums[index])

	}

	return result
}

// find last one less than target, binary
func searchInts(nums []int, target int) int {
	i, j := 0, len(nums)

	var result int
	for i < j {
		h := i + (j-i)/2

		if nums[h] < target {
			result = h
			i = h + 1
		} else {
			j = h
		}
	}

	if nums[result] >= target {
		return -1
	}

	return result
}

func addBinary(a string, b string) string {
	ans := ""
	carry := 0
	lenA, lenB := len(a), len(b)
	n := max(lenA, lenB)

	for i := 0; i < n; i++ {
		if i < lenA {
			carry += int(a[lenA-i-1] - '0')
		}
		if i < lenB {
			carry += int(b[lenB-i-1] - '0')
		}
		ans = strconv.Itoa(carry%2) + ans
		carry /= 2
	}
	if carry > 0 {
		ans = "1" + ans
	}
	return ans
}

func addStrings(num1 string, num2 string) string {
	ans := ""
	carry := 0
	lenA, lenB := len(num1), len(num2)

	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}

	n := max(lenA, lenB)

	for i := 0; i < n; i++ {
		if i < lenA {
			carry += int(num1[lenA-i-1] - '0')
		}
		if i < lenB {
			carry += int(num2[lenB-i-1] - '0')
		}
		ans = strconv.Itoa(carry%10) + ans
		carry /= 10
	}
	if carry > 0 {
		ans = "1" + ans
	}
	return ans
}

func match(word, pattern string) bool {
	set := map[byte]byte{}
	for i := range word {
		x := word[i]
		y := pattern[i]

		r, ok := set[x]

		if !ok {
			set[x] = y
		} else {
			if r != y {
				return false
			}
		}
	}
	return true
}

func findAndReplacePattern(words []string, pattern string) (ans []string) {
	for _, word := range words {
		if match(word, pattern) && match(pattern, word) {
			ans = append(ans, word)
		}
	}
	return
}

type ArrayReader struct{}

func (a *ArrayReader) get(index int) int {
	return 0
}

func search(reader ArrayReader, target int) int {
	if reader.get(0) == target {
		return 0
	}

	left, right := 0, 1

	for reader.get(right) < target {
		left = right
		right <<= 1
	}

	for left <= right {
		mid := left + (right-left)/2

		num := reader.get(mid)
		if num == target {
			return mid
		}

		if num > target {
			right = mid - 1
		} else {
			left = mid + 1
		}

	}
	return -1
}

func increasingTriplet(nums []int) bool {
	var (
		min       int = math.MaxInt64
		secondMin int = math.MaxInt64
	)

	for i, v := range nums {
		// 有三个数了，检查当前是否满足条件
		if i > 1 {
			if min < secondMin && secondMin < v {
				return true
			}
		}

		if v <= min {
			min = v
			continue
		}

		if v <= secondMin {
			secondMin = v
			continue
		}
	}
	return false
}

func missing(idx int, nums []int) int {
	return nums[idx] - nums[0] - idx
}

func missingElement(nums []int, k int) int {
	var result int = -1
	for i := 1; i < len(nums); i++ {
		if missing(i-1, nums) < k && k <= missing(i, nums) {
			result = nums[i-1] + k - missing(i-1, nums)
			break
		}
	}

	if result == -1 {
		result = nums[len(nums)-1] + k - missing(len(nums)-1, nums)
	}

	return result
}

func duplicateZeros(arr []int) {
	// 先计算出最后一个位置
	var (
		idx   int = -1
		count int
	)

	for count < len(arr) {
		idx++
		if arr[idx] != 0 {
			count++
		} else {
			count += 2
		}
	}

	// log.Println(idx, count)

	start := count - 1

	if start == len(arr) {

		arr[start-1] = arr[idx]

		start -= 2
		idx--
	}

	for start > 0 {
		// log.Println(idx, start)
		if arr[idx] != 0 {
			arr[start] = arr[idx]
			start--
			idx--
		} else {
			arr[start] = 0
			arr[start-1] = 0
			idx--
			start -= 2
		}
	}

	// log.Println(arr)

}

func countPairs(nums1 []int, nums2 []int) int64 {
	for i := range nums1 {
		diff := nums1[i] - nums2[i]

		nums1[i] = diff
	}

	var count int64

	sort.Ints(nums1)

	for i := 0; i < len(nums1)-1; i++ {
		target := -nums1[i]

		left, right := i, len(nums1)

		for left+1 != right {
			mid := left + (right-left)/2

			if nums1[mid] > target {
				right = mid
			} else {
				left = mid
			}
		}

		count += int64(len(nums1) - right)
	}

	return count
}

type MyCircularQueue struct {
	size  int
	queue []int
	// 首元素位置
	front int

	// 尾元素位置
	rear int
}

func Constructor(k int) MyCircularQueue {
	return MyCircularQueue{
		queue: make([]int, k+1),
		size:  k + 1,
	}
}

func (m *MyCircularQueue) EnQueue(value int) bool {
	if m.IsFull() {
		return false
	}

	m.queue[m.rear] = value

	m.rear = (m.rear + 1) % m.size

	return true
}

func (m *MyCircularQueue) DeQueue() bool {
	if m.IsEmpty() {
		return false
	}

	m.front = (m.front + 1) % m.size

	return true
}

func (m *MyCircularQueue) Front() int {
	if m.IsEmpty() {
		return -1
	}
	return m.queue[m.front]
}

func (m *MyCircularQueue) Rear() int {
	if m.IsEmpty() {
		return -1
	}

	log.Println(m.rear)
	return m.queue[(m.rear-1+m.size)%m.size]
}

func (m *MyCircularQueue) IsEmpty() bool {
	return m.front == m.rear
}

func (m *MyCircularQueue) IsFull() bool {
	return m.front == ((m.rear + 1) % m.size)
}

func numberOfPairs(nums []int) []int {
	var (
		pairs int
		items int
	)

	counts := make(map[int]int)

	for _, v := range nums {
		counts[v] += 1
	}

	for _, v := range counts {
		pairs += v / 2
		items += v % 2
	}

	return []int{pairs, items}
}

func reversePairs(nums []int) int {
	tmp := make([]int, len(nums))

	res := mergeSort(nums, tmp, 0, len(nums)-1)
	return res
}

func mergeSort(nums []int, tmp []int, left, right int) int {
	if left >= right {
		return 0
	}

	mid := (left + right) / 2

	res := mergeSort(nums, tmp, left, mid) + mergeSort(nums, tmp, mid+1, right)

	// Merge
	i, j := left, mid+1
	for k := left; k <= right; k++ {
		tmp[k] = nums[k]
	}

	for k := left; k <= right; k++ {
		if j == right+1 || (i <= mid && nums[i] <= nums[j]) {
			nums[k] = tmp[i]
			i++
		} else {
			nums[k] = tmp[j]
			j++
			res += mid - i + 1
		}
	}

	return res
}

// func mergeSort(nums []int, tmp []int, left, right int) int {
// 	if left >= right {
// 		return 0
// 	}

// 	mid := (left + right) / 2

// 	res := mergeSort(nums, tmp, left, mid) + mergeSort(nums, tmp, mid+1, right)

// 	// 合并
// 	i, j := left, mid+1

// 	for k := left; k <= right; k++ {
// 		tmp[k] = nums[k]
// 	}

// 	for k := left; k <= right; k++ {
// 		// log.Printf("left:%d, right: %d, i: %d, j: %d, k: %d, mid : %d", left, right, i, j, k, mid)
// 		// 越界判断
// 		if i == mid+1 {
// 			nums[k] = tmp[j]
// 			j++
// 		} else if j == right+1 {
// 			nums[k] = tmp[i]
// 			i++
// 		} else if tmp[i] <= tmp[j] {
// 			nums[k] = tmp[i]
// 			i++
// 		} else {
// 			nums[k] = tmp[j]
// 			j++
// 			res += mid - i + 1
// 		}

// 	}

// 	return res
// }

func twoSum(nums []int, target int) []int {
	if len(nums) < 2 {
		return nums
	}

	var (
		i = 0
		j = len(nums) - 1
	)

	for i < j {
		if nums[i]+nums[j] == target {
			return []int{nums[i], nums[j]}
		}

		if nums[i]+nums[j] < target {
			i++
		} else {
			j--
		}
	}
	return []int{}
}

func reverseByWord(s string) string {
	res := make([]byte, 0)

	b := []byte(s)
	var (
		i, j = len(b) - 1, len(b) - 1
	)

	for j >= 0 {
		if b[j] == ' ' {
			i--
			j--
			continue
		}

		// 取词
		if i < 0 || b[i] == ' ' {
			res = append(res, b[i+1:j+1]...)
			res = append(res, ' ')
			j = i
		} else {
			i--
			continue
		}

	}

	if len(res) > 0 {
		res = res[:len(res)-1]
	}

	return Bytes2String(res)

}

func Bytes2String(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}

func missingNumber(nums []int) int {
	i := 0
	j := len(nums) - 1

	for i <= j {
		mid := (i + j) / 2

		if nums[mid] == mid {
			i = mid + 1
		} else {
			j = mid - 1
		}
	}

	return i
}

func firstUniqChar(s string) byte {
	idx := firstUniqCharIdx(s)
	if idx == -1 {
		return ' '
	}
	return s[idx]
}

func searchRange(nums []int, target int) int {

	i, j := 0, len(nums)

	for i < j && i < len(nums) {
		h := int(uint(i+j) >> 1)
		if nums[h] < target {
			i = h + 1
		} else {
			j = h
		}
	}

	if i >= len(nums) || nums[i] != target {
		return 0
	}

	var res int

	for ; i < len(nums) && nums[i] == target; i++ {
		res++
	}

	return res
}

func cuttingRope(n int) int {
	products := make([]int, n+1)
	products[1] = 1

	for i := 2; i <= n; i++ {
		for j := 1; j < i; j++ {
			products[i] = maxInThree(products[i], products[i-j]*j, (i-j)*j)
		}
	}

	return products[n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxInThree(a, b, c int) int {
	if a > b {
		return max(a, c)
	}

	if a < b {
		return max(b, c)
	}

	if a > c {
		return max(a, b)
	}

	if a < c {
		return max(b, c)
	}

	if b > c {
		return max(a, b)
	}

	if b < c {
		return max(a, c)
	}

	return a
}

func countDigitOne(n int) int {
	var (
		cur   int = n % 10
		high  int = n / 10
		low   int
		digit int = 1
		res   int
	)

	for cur != 0 || high != 0 {
		// log.Printf("high: %d, cur: %d, low: %d, digit: %d \n", high, cur, low, digit)
		if cur == 0 {
			res += high * digit
		} else if cur == 1 {
			res += high*digit + low + 1
		} else {
			res += (high + 1) * digit
		}

		// 下一轮
		low = cur*digit + low
		cur = high % 10
		digit *= 10
		high = high / 10

	}

	return res

}

func movesToMakeZigzag(nums []int) int {
	var (
		eventCount int
		oddCount   int
		left       int = -1
		right      int = -1
	)
	// 偶数最大
	for i := range nums {
		if i&1 == 0 {
			if i == len(nums)-1 {
				right = -1
			} else {
				right = nums[i+1]
			}

			if left >= nums[i] {
				eventCount += left - nums[i] + 1
				left = nums[i] - 1
			}

			if right >= nums[i] {
				eventCount += right - nums[i] + 1
				right = nums[i] - 1
			}

			left = right
		}
	}

	left = nums[0]
	right = -1

	// 奇数最大
	for i := range nums {
		if i&1 != 0 {
			if i == len(nums)-1 {
				right = -1
			} else {
				right = nums[i+1]
			}

			// log.Println(left, right)

			if left >= nums[i] {
				oddCount += left - nums[i] + 1
				left = nums[i] - 1
			}
			if right >= nums[i] {
				oddCount += right - nums[i] + 1
				right = nums[i] - 1
			}

			left = right
		}
	}

	// log.Println(eventCount, oddCount)

	// 奇数最大

	return min(eventCount, oddCount)
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func mergeSimilarItems(items1 [][]int, items2 [][]int) [][]int {
	values := make(map[int]int)

	for _, item := range items1 {
		v := values[item[0]]

		v += item[1]

		values[item[0]] = v
	}

	for _, item := range items2 {
		v := values[item[0]]

		v += item[1]

		values[item[0]] = v
	}

	result := make([][]int, 0)
	for k, v := range values {
		result = append(result, []int{k, v})
	}

	sort.Sort(itemSorter(result))

	return result
}

type itemSorter [][]int

func (s itemSorter) Len() int {
	return len(s)
}

func (s itemSorter) Less(i, j int) bool {
	return s[i][0] < s[j][0]
}

func (s itemSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func getFolderNames(names []string) []string {
	status := make(map[string]int)

	res := make([]string, 0, len(names))

	for _, name := range names {
		v, ok := status[name]

		if !ok {
			res = append(res, name)
			status[name] = 1
			continue
		}

		for status[name+"("+strconv.Itoa(v)+")"] > 0 {
			v++
		}

		res = append(res, name+"("+strconv.Itoa(v)+")")
		status[name+"("+strconv.Itoa(v)+")"] = 1
		status[name] = v + 1
	}

	return res

}

func sumNums(n int) int {

	s := &numSum{}
	s.sum(n)

	return s.n
}

type numSum struct {
	n int
}

func (n *numSum) sum(num int) int {
	var _ = (num > 1) && (n.sum(num-1) > 0)
	n.n += num

	return n.n
}

func validateStackSequences(pushed []int, popped []int) bool {
	s := &Stack{data: make([]int, 0, len(pushed))}
	var popIndex int

	for _, v := range pushed {
		s.Push(v)
		for popIndex < len(popped) && s.Top() == popped[popIndex] {
			s.Pop()
			popIndex++
		}
	}

	return s.IsEmpty()
}

type Stack struct {
	data []int
}

func (s *Stack) Push(val int) {
	s.data = append(s.data, val)
}

func (s *Stack) Pop() int {
	if s.IsEmpty() {
		return -1 // or any error handling mechanism you like
	}
	n := len(s.data)
	val := s.data[n-1]
	s.data = s.data[:n-1]
	return val
}

func (s *Stack) Top() int {
	if s.IsEmpty() {
		return -1 // or any error handling mechanism you like
	}
	return s.data[len(s.data)-1]
}

func (s *Stack) IsEmpty() bool {
	return len(s.data) == 0
}

func findContinuousSequence(target int) [][]int {
	res := make([][]int, 0)

	arr := make([]int, 0)

	for i := 1; i <= target/2+1; i++ {
		arr = append(arr, i)
	}

	var (
		left  = 0
		right = 1
	)

	for left < right {

		sum := (arr[left] + arr[right]) * (right - left + 1)
		sum /= 2

		// log.Println(arr[left], arr[right], sum)

		if sum == target {
			res = append(res, arr[left:right+1])
			left++
		}

		if sum < target {
			right++
		}

		if sum > target {
			left++
		}

	}

	return res
}

func findLongestSubarray(array []string) []string {

	prefixSum := make([]int, len(array)+1)

	for i, v := range array {

		_, err := strconv.Atoi(v)
		if err != nil {
			prefixSum[i+1] = -1
		} else {
			prefixSum[i+1] = 1
		}
	}

	first := make(map[int]int)
	first[0] = 0

	for i := 1; i < len(prefixSum); i++ {
		prefixSum[i] = prefixSum[i-1] + prefixSum[i]
		_, ok := first[prefixSum[i]]
		if !ok {
			first[prefixSum[i]] = i
		}
	}

	var (
		left      int
		right     int
		maxLength int
	)

	for i := 2; i < len(prefixSum); i++ {
		firstIdx := first[prefixSum[i]]

		length := i - firstIdx + 1
		if length > maxLength {
			left = firstIdx
			right = i
			maxLength = length
		}
	}

	return array[left:right]
}

func findDisappearedNumbers(nums []int) []int {

	for i := 0; i < len(nums); {
		v := nums[i]

		if v != i+1 && nums[v-1] != v {
			tmp := nums[v-1]
			nums[v-1] = v
			nums[i] = tmp
		} else {
			i++
		}
	}

	res := make([]int, 0)
	for i, v := range nums {
		if v != i+1 {
			res = append(res, i+1)
		}
	}

	return res
}

func constructArr(a []int) []int {
	if len(a) == 0 {
		return nil
	}
	res := make([]int, len(a))
	res[0] = 1

	for i := 1; i < len(a); i++ {
		res[i] = res[i-1] * a[i-1]
	}

	p := 1

	for i := len(a) - 2; i >= 0; i-- {
		p *= a[i+1]
		res[i] *= p
	}

	return res
}

func nextPermutation(nums []int) {
	var (
		left  int = len(nums) - 2
		right int = len(nums) - 1
	)

	for left >= 0 && nums[left] >= nums[right] {
		left--
		right--
	}

	if left < 0 {
		swap(nums)
		return
	}

	var swapIdx int

	for i := len(nums) - 1; i > left; {
		if nums[i] <= nums[left] {
			i--
		} else {
			swapIdx = i
			break
		}
	}

	nums[left], nums[swapIdx] = nums[swapIdx], nums[left]
	log.Println(right)

	swap(nums[right:])
}

func swap[k any](nums []k) {
	if len(nums) > 1 {
		length := len(nums)
		for i := 0; i < len(nums)/2; i++ {
			nums[i], nums[length-i-1] = nums[length-i-1], nums[i]
		}
	}
}

func decodeString(s string) string {

	stack := make([]byte, 0, len(s))

	for i := 0; i < len(s); i++ {

		char := s[i]

		if char >= '0' && char <= '9' {
			stack = append(stack, char)
			continue
		}

		if char == '[' {
			stack = append(stack, char)
			continue
		}

		// 出栈
		if char == ']' {
			// log.Println("before stack: ", string(stack))
			var cur []byte

			var charIdx int

			// 将字符入栈, 且截断
			for charIdx = len(stack) - 1; charIdx >= 0; charIdx-- {
				if stack[charIdx] == '[' {
					cur = stack[charIdx+1:]
					stack = stack[:charIdx]
					break
				}
			}

			var factor int

			var factorIdx int

			// 计算因子
			for factorIdx = len(stack) - 1; factorIdx >= 0; factorIdx-- {
				if stack[factorIdx] < '0' || stack[factorIdx] > '9' {
					break
				}
			}

			factor, _ = strconv.Atoi(Bytes2String(stack[factorIdx+1:]))

			if factorIdx != 0 {
				stack = stack[:factorIdx+1]
			}

			// log.Printf("cur: %s, factor: %d", string(cur), factor)
			// log.Println("after stack: ", string(stack))

			res := make([]byte, 0, len(cur)*factor)

			// 展开
			for i := 0; i < factor; i++ {
				res = append(res, cur...)
			}
			stack = append(stack, res...)

			continue
		}

		stack = append(stack, char)

	}
	res := Bytes2String(stack)
	// log.Println(res)
	return res
}

func mergeSubArray(intervals [][]int) [][]int {
	var s SubArrayMerge = intervals

	sort.Sort(s)

	res := make([][]int, 0, len(intervals))

	var interval []int

	for _, v := range intervals {
		if len(interval) == 0 {
			interval = v
		}

		if overlap(interval, v) {
			// log.Println("overlap++", interval, v)
			// merge
			interval = mergeTwo(interval, v)
		} else {
			// log.Println("overlap--", interval, v)
			res = append(res, interval)
			interval = v
		}
	}
	res = append(res, interval)
	return res
}

func overlap(x []int, y []int) bool {
	if x[1] >= y[0] {
		return true
	}
	return false
}

func mergeTwo(x []int, y []int) []int {
	return []int{min(x[0], x[1]), max(x[1], y[1])}
}

type SubArrayMerge [][]int

func (s SubArrayMerge) Len() int {
	return len(s)
}

func (s SubArrayMerge) Less(i, j int) bool {
	x1 := s[i][0]
	y1 := s[i][1]
	x2 := s[j][0]
	y2 := s[j][1]

	if x1 == x2 {
		return y1 < y2
	}
	return x1 < x2
}

func (s SubArrayMerge) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func subarraySum(nums []int, k int) int {
	// 转换成前缀和数组
	for i := 1; i < len(nums); i++ {
		nums[i] += nums[i-1]
	}
	log.Println(nums)
	var (
		count int
	)

	for i := len(nums) - 1; i >= 0; i-- {
		for j := i; j >= 0; j-- {
			var sum int
			if j == 0 {
				sum = nums[i]
			} else {
				sum = nums[i] - nums[j-1]
			}

			if sum == k {
				count++
			}
		}
	}

	return count
}

func findTheWinner(n int, k int) int {
	visited := make([]bool, n+1)
	var count int
	var start int = 1

	for count < n-1 {
		// log.Println("start:", start)
		var step int
		for i := start; i <= n; {
			// log.Println("cur:", i)
			if !visited[i] {

				// log.Println("walk: ", i)
				step++
				if step == k {
					log.Println("evict: ", i)
					visited[i] = true
					// 找到下一个开始的位置
					start = getNextStart(i, visited)
					break
				}
			}
			i++
			if i > n {
				i = 1
			}
		}
		count++
	}

	// 最后一个是false
	for i := 1; i < len(visited); i++ {
		if !visited[i] {
			return i
		}
	}

	return 0
}

func getNextStart(i int, visited []bool) int {
	for j := i + 1; j != i; j++ {
		if j == len(visited) {
			j = 1
		}

		if !visited[j] {
			return j
		}
	}
	return 0
}

func hammingWeight(num uint32) int {
	var count uint32
	for num != 0 {
		count += (num & 1)
		num = num >> 1
	}
	return int(count)
}

type numString []string

func (n numString) Len() int {
	return len(n)
}

func (n numString) Less(i, j int) bool {
	return n[i]+n[j] < n[j]+n[i]
}

func (n numString) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
}

func minNumber(nums []int) string {
	numbers := make(numString, 0, len(nums))

	for _, num := range nums {
		numbers = append(numbers, strconv.Itoa(num))
	}

	sort.Sort(numbers)

	// fmt.Println(numbers)

	var r string

	for _, num := range numbers {
		r += num
	}

	return r

}

func convertTemperature(celsius float64) []float64 {
	return []float64{
		celsius + 273.15,
		celsius*1.80 + 32.00,
	}
}

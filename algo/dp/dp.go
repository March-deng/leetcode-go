package dp

import (
	"log"
	"math"
)

func fib(n int) int {
	if n == 0 {
		return 0
	}

	mod := int(1e9 + 7)

	var a, b int

	a, b = 0, 1
	for i := 2; i < n+1; i++ {
		temp := (a + b) % mod
		a, b = b, temp
	}

	return b
}

func numWays(n int) int {
	if n == 1 {
		return 1
	}

	mod := int(1e9 + 7)

	var a, b int

	a, b = 1, 1
	for i := 2; i < n+1; i++ {
		temp := (a + b) % mod
		a, b = b, temp
	}

	return b

}

func climbStairs(n int) int {
	if n <= 2 {
		return n
	}

	dpTable := make([]int, n+1)
	dpTable[0] = 0
	dpTable[1] = 1
	dpTable[2] = 2

	for i := 3; i <= n; i++ {
		dpTable[i] = dpTable[i-1] + dpTable[i-2]
	}

	return dpTable[n]

}

func minCostClimbingStairs(cost []int) int {
	if len(cost) <= 1 {
		return 0
	}

	dpTable := make([]int, len(cost))
	dpTable[0] = 0
	dpTable[1] = 0

	for i := 2; i < len(cost); i++ {
		dpTable[i] = min(dpTable[i-2]+cost[i-2], dpTable[i-1]+cost[i-1])
	}

	lastIndex := len(cost) - 1

	return min(dpTable[lastIndex]+cost[lastIndex], dpTable[lastIndex-1]+cost[lastIndex-1])
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}

func uniquePaths(m int, n int) int {
	dpTable := make([][]int, n)
	for i := range dpTable {
		dpTable[i] = make([]int, m)
	}

	// 初始化，第一行的所有元素都是1，第一列的所有元素也都是1
	// 初始化第一行
	for i := range dpTable[0] {
		dpTable[0][i] = 1
	}

	// 初始化第一列
	for i := 0; i < n; i++ {
		dpTable[i][0] = 1
	}

	// 遍历求得整个状态表
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if dpTable[i][j] == 0 {
				dpTable[i][j] = dpTable[i][j-1] + dpTable[i-1][j]
			}
		}
	}

	return dpTable[n-1][m-1]
}

func basicPackage(n, volume int, weights, values []int) int {
	return solve(n, 0, volume, weights, values)
}

// i代表第i件物品，j代表此时的剩余重量
func solve(n, i int, j int, weights, values []int) int {
	if n == i {
		return 0
	}
	// 能选择第i件物品，那么选或者不选都试一下，返回最大的
	if weights[i] <= j {
		return max(solve(n, i+1, j, weights, values), solve(n, i+1, j-weights[i], weights, values)+values[i])
	} else { // 不能
		return solve(n, i+1, j, weights, values)
	}
}

func canPartition(nums []int) bool {
	// 去和
	var sum int
	for _, v := range nums {
		sum += v
	}

	if sum%2 != 0 {
		return false
	}

	target := sum / 2
	dpTable := make([]int, 10001)

	for i := 0; i < len(nums); i++ {
		for j := target; j >= nums[i]; j-- {
			dpTable[j] = max(dpTable[j], dpTable[j-nums[i]]+nums[i])
		}
	}

	return dpTable[target] == target

}

func wordBreak(s string, wordDict []string) bool {
	dpTables := make([]bool, len(s)+1)
	dpTables[0] = true

	set := make(map[string]struct{})

	for _, word := range wordDict {
		set[word] = struct{}{}
	}

	for i := 1; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			sub := s[j:i]

			// log.Println(sub)

			if _, ok := set[sub]; ok && dpTables[j] {
				dpTables[i] = true
			}
		}
	}

	return dpTables[len(s)]
}

func robRoom(nums []int) int {
	dpTables := make([]int, len(nums))
	dpTables[0] = nums[0]

	if len(nums) >= 2 {
		dpTables[1] = max(nums[0], nums[1])

		for i := 2; i < len(nums); i++ {
			dpTables[i] = max(dpTables[i-2]+nums[i], dpTables[i-1])
		}
	}

	return dpTables[len(nums)-1]
}

// func rob(nums []int) int {
// 	if len(nums) == 0 {
// 		return 0
// 	}

// 	if len(nums) == 1 {
// 		return nums[0]
// 	}

// 	return max(robRoom(nums[:len(nums)-1]), robRoom(nums[1:]))
// }

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rob(root *TreeNode) int {
	value1, value2 := robNode(root)
	return max(value1, value2)
}

// 第一个返回值是偷，第二个是不偷
func robNode(node *TreeNode) (int, int) {
	if node == nil {
		return 0, 0
	}

	left1, left2 := robNode(node.Left)
	right1, right2 := robNode(node.Right)

	value1, value2 := node.Val+left2+right2, max(left1, left2)+max(right1, right2)
	return value1, value2
}

func max(i, j int) int {
	if i > j {
		return i
	}

	return j
}

func maxProfit(prices []int) int {
	dpTables := make([][]int, len(prices))

	for i := range dpTables {
		dpTables[i] = make([]int, 5)
	}

	dpTables[0][1] = -prices[0]
	dpTables[0][3] = -prices[0]

	for i := 1; i < len(prices); i++ {
		dpTables[i][1] = max(dpTables[i-1][0]-prices[i], dpTables[i-1][1])
		dpTables[i][2] = max(dpTables[i-1][1]+prices[i], dpTables[i-1][2])
		dpTables[i][3] = max(dpTables[i-1][3], dpTables[i-1][2]-prices[i])
		dpTables[i][4] = max(dpTables[i-1][4], dpTables[i-1][3]+prices[i])
	}

	return dpTables[len(prices)-1][4]

}

func findLength(nums1 []int, nums2 []int) int {
	dpTables := make([][]int, len(nums1)+1)

	for i := range dpTables {
		dpTables[i] = make([]int, len(nums2)+1)
	}

	var result int

	// 遍历所有的数
	for i := 0; i < len(nums1); i++ {
		for j := 0; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dpTables[i+1][j+1] = dpTables[i][j] + 1
			}
			if dpTables[i+1][j+1] > result {
				result = dpTables[i+1][j+1]
			}
		}
	}

	return result
}

func minFlipsMonoIncr(s string) int {
	dpTables := make([][]int, len(s))

	for i := range dpTables {
		dpTables[i] = make([]int, 2)
	}

	if s[0] == '0' {
		dpTables[0][0] = 0
		dpTables[0][1] = 1
	} else {
		dpTables[0][0] = 1
		dpTables[0][1] = 0
	}

	for i := 1; i < len(s); i++ {
		switch s[i] {
		case '0':
			dpTables[i][0] = dpTables[i-1][0]
			dpTables[i][1] = min(dpTables[i-1][0], dpTables[i-1][1]) + 1
		case '1':
			dpTables[i][0] = dpTables[i-1][0] + 1
			dpTables[i][1] = min(dpTables[i-1][0], dpTables[i-1][1])
		}
	}

	return min(dpTables[len(s)-1][0], dpTables[len(s)-1][1])

}

func minCost(costs [][]int) int {

	dpTable := costs[0]

	for i := 1; i < len(costs); i++ {
		dpTableNew := make([]int, 3)
		for j := 0; j < len(costs[0]); j++ {
			dpTableNew[j] = min(dpTable[(j+1)%3], dpTable[(j+2)%3]) + costs[i][j]
		}

		dpTable = dpTableNew
	}

	// 最后一行的最小值
	var result int = math.MaxInt32

	for _, v := range dpTable {
		if result > v {
			result = v
		}
	}

	return result
}

func maxValue(grid [][]int) int {

	// 先看第一行和第一列，变为前缀和
	for i := 1; i < len(grid[0]); i++ {
		grid[0][i] = grid[0][i] + grid[0][i-1]
	}

	for i := 1; i < len(grid); i++ {
		grid[i][0] = grid[i][0] + grid[i-1][0]
	}

	// 从第二行第二列开始，状态递推
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			grid[i][j] = max(grid[i-1][j], grid[i][j-1]) + grid[i][j]
		}
	}

	r := len(grid) - 1
	c := len(grid[0]) - 1

	return grid[r][c]
}

func lastRemaining(n int, m int) int {
	var x int
	for i := 2; i <= n; i++ {
		x = (x + m) % i
	}

	return x

}

func minPathSum(grid [][]int) int {
	var (
		i int = len(grid) - 1
		j int = len(grid[0]) - 1
	)

	for i >= 0 && j >= 0 {
		// log.Println("+++++++++++++++++++++++++++")

		// 首先设置自己的值
		if i != len(grid)-1 && j != len(grid[0])-1 {
			grid[i][j] = min(grid[i][j]+grid[i+1][j], grid[i][j]+grid[i][j+1])
		}

		// log.Printf("i: %d, j: %d, value: %d", i, j, grid[i][j])

		if i > 0 {
			// 行，固定列
			for row := i - 1; row >= 0; row-- {
				var (
					downside  int = 1000000
					rightside int = 1000000
				)

				if j != len(grid[0])-1 {
					rightside = grid[row][j+1]
				}

				downside = grid[row+1][j]

				// log.Printf("row: %d, j: %d, value: %d, downside: %d, rightside: %d", row, j, grid[row][j], downside, rightside)

				grid[row][j] = min(grid[row][j]+downside, grid[row][j]+rightside)
			}
		}

		if j > 0 {
			for column := j - 1; column >= 0; column-- {
				var (
					downside  int = 1000000
					rightside int = 1000000
				)

				if i != len(grid)-1 {
					downside = grid[i+1][column]
				}

				rightside = grid[i][column+1]

				// log.Printf("i: %d, column: %d, value: %d, downside: %d, rightside: %d", i, column, grid[i][column], downside, rightside)

				grid[i][column] = min(grid[i][column]+downside, grid[i][column]+rightside)
			}
		}

		i--
		j--

	}

	// log.Println(grid)

	return grid[0][0]

}

func countSubstrings(s string) int {

	var count int

	for i := 0; i < len(s); i++ {
		for j := 0; j <= 1; j++ {
			left := i
			right := i + j

			for left >= 0 && right < len(s) {
				if s[left] == s[right] {
					count++
					left--
					right++
				} else {
					break
				}
			}
		}
	}

	return count
}

func translateNum(num int) int {
	if num < 10 {
		return 1
	}
	digits := getDigits(num)

	dpTable := make([]int, len(digits))
	dpTable[0] = 1

	if digits[0]*10+digits[1] >= 26 {
		dpTable[1] = 1
	} else {
		dpTable[1] = 2
	}

	for i := 2; i < len(digits); i++ {

		if digits[i-1] == 0 || digits[i-1]*10+digits[i] >= 26 {
			dpTable[i] = dpTable[i-1]

		} else {
			dpTable[i] = dpTable[i-2] + dpTable[i-1]
		}

	}

	// log.Println(dpTable)

	return dpTable[len(dpTable)-1]
}

func getDigits(num int) []int {
	res := make([]int, 0)

	for num != 0 {
		res = append(res, num%10)
		num /= 10
	}

	reverse(res)

	return res
}

func reverse(nums []int) {
	if len(nums) > 1 {
		length := len(nums)
		for i := 0; i < len(nums)/2; i++ {
			nums[i], nums[length-i-1] = nums[length-i-1], nums[i]
		}
	}
}

func lengthOfLongestSubstring(s string) int {
	max := 0

	check := make(map[byte]int)

	var lastSub int

	for j := 0; j < len(s); j++ {
		last, ok := check[s[j]]
		if !ok {
			lastSub += 1
		} else {
			if lastSub < j-last {
				lastSub += 1
			} else {
				lastSub = j - last
			}
		}

		check[s[j]] = j

		if lastSub > max {
			max = lastSub
		}

	}

	return max
}

func maxProfitSellStock(prices []int) int {

	if len(prices) <= 1 {
		return 0
	}
	var (
		profit   int
		minPrice int = prices[0]
	)

	if len(prices) <= 1 {
		return 0
	}

	for _, v := range prices[1:] {

		profit = max(profit, v-minPrice)

		log.Println(profit, minPrice)

		minPrice = min(minPrice, v)
	}

	return profit

}

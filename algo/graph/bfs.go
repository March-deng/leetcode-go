package graph

import (
	"reflect"
	"unsafe"
)

func minMutation(start string, end string, bank []string) int {
	/// 将bank转为map
	bankSet := make(map[string]struct{})
	for _, v := range bank {
		bankSet[v] = struct{}{}
	}

	if _, ok := bankSet[end]; !ok {
		return -1
	}

	// 用队列来模拟广度搜索
	queue := make([]string, 0)
	queue = append(queue, start)

	dicBytes := String2Bytes("ACGT")

	// 一次循环就是走了一步
	for step := 0; len(queue) != 0; step++ {
		temp := queue
		queue = make([]string, 0)
		// 让当前队列里所有的元素都向前走一步，如果走一步后能匹配到有效的结果，就说明达到该结果的最小步数就是当前步数，那么该结果就需要被标记为已经被访问到
		for _, cur := range temp {
			// 向前走一步，就是每一位都试着变换一次
			curBytes := []byte(cur)
			for i, curChar := range curBytes {
				// log.Println(curChar)
				for _, char := range dicBytes {
					if curChar != char { // 说明不符合，就能从curChar变为char
						curBytes[i] = char // 替换并检查
						if _, ok := bankSet[Bytes2String(curBytes)]; ok {
							if Bytes2String(curBytes) == end {
								return step + 1
							}

							// 标记，入队
							delete(bankSet, Bytes2String(curBytes))

							queue = append(queue, string(curBytes))
						}
						curBytes[i] = curChar
					}
				}
			}
		}
	}

	// 广度优先搜索

	return -1
}

func String2Bytes(s string) []byte {
	sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
	bh := reflect.SliceHeader{
		Data: sh.Data,
		Len:  sh.Len,
		Cap:  sh.Len,
	}
	return *(*[]byte)(unsafe.Pointer(&bh))
}

func Bytes2String(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}

func setZeroes(matrix [][]int) {

}

func movingCount(m int, n int, k int) int {
	status := make(map[pair]bool)

	checkMovingCount(status, 0, 0, m, n, k)

	return len(status)
}

type pair struct {
	i int
	j int
}

func checkMovingCount(status map[pair]bool, i int, j int, m int, n int, k int) {
	if i < 0 || j < 0 || i >= m || j >= n {
		return
	}
	if sumDigit(i)+sumDigit(j) > k {
		return

	}

	status[pair{i, j}] = true

	if !status[pair{i: i + 1, j: j}] {

		checkMovingCount(status, i+1, j, m, n, k)
	}

	if !status[pair{i: i, j: j + 1}] {

		checkMovingCount(status, i, j+1, m, n, k)
	}

}

func sumDigit(n int) int {
	var sum int

	for n != 0 {
		sum += n % 10
		n = n / 10
	}

	return sum
}

func checkValidGrid(grid [][]int) bool {
	pairs := make([]pair, 8)

	var (
		row int
		col int
	)

	for i := 0; i < len(grid)*len(grid[0])-1; i++ {
		var getExpected bool
		getAllAxis(row, col, pairs)
		expected := grid[row][col] + 1

		for _, p := range pairs {
			if p.i < 0 || p.j < 0 || p.i >= len(grid) || p.j >= len(grid[0]) {
				continue
			}

			if grid[p.i][p.j] == expected {
				getExpected = true
				row = p.i
				col = p.j
				break
			}
		}

		if !getExpected {
			// log.Println(expected)
			return false
		}

	}

	return true
}

// 获取所有坐标
func getAllAxis(row, col int, pairs []pair) {

	pairs[0] = pair{
		i: row - 2,
		j: col - 1,
	}
	pairs[1] = pair{
		i: row - 1,
		j: col - 2,
	}
	pairs[2] = pair{
		i: row + 1,
		j: col - 2,
	}
	pairs[3] = pair{
		i: row + 2,
		j: col - 1,
	}
	pairs[4] = pair{
		i: row + 2,
		j: col + 1,
	}
	pairs[5] = pair{
		i: row + 1,
		j: col + 2,
	}
	pairs[6] = pair{
		i: row - 1,
		j: col + 2,
	}
	pairs[7] = pair{
		i: row - 2,
		j: col + 1,
	}
}

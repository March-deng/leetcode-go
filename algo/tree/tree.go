package tree

import (
	"math"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	q := newQueue()
	q.push(root)

	var count int

	for !q.empty() {
		size := q.size()
		count += size
		for i := 0; i < size; i++ {
			node := q.pop()
			if node.Left != nil && node.Right != nil {
				q.push(node.Left)
				q.push(node.Right)
				continue
			}
			lastLevel := 2 * i

			if node.Left != nil || node.Right != nil {
				lastLevel += 1
			}
			count += lastLevel
			return count
		}
	}

	return count
}

func isBalanced(root *TreeNode) bool {
	height := getHeight(root)

	return !(height == -1)
}

func getHeight(node *TreeNode) int {
	if node == nil {
		return 0
	}
	leftHeight := getHeight(node.Left)
	rightHeight := getHeight(node.Right)

	if leftHeight == -1 || rightHeight == -1 {
		return -1
	}

	if leftHeight-rightHeight > 1 || leftHeight-rightHeight < -1 {
		return -1
	}

	return max(leftHeight, rightHeight) + 1

}

func max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

func binaryTreePaths(root *TreeNode) []string {

	t := &TreePath{
		nodes: make([]*TreeNode, 0),
		paths: make([]string, 0),
		b:     &strings.Builder{},
	}
	t.traversePath(root)
	return t.paths

}

type TreePath struct {
	nodes []*TreeNode
	paths []string
	b     *strings.Builder
}

func (t *TreePath) traversePath(root *TreeNode) {
	if root.Left != nil {
		t.nodes = append(t.nodes, root)
		t.traversePath(root.Left)
		t.nodes = t.nodes[:len(t.nodes)-1]
	}

	if root.Right != nil {
		t.nodes = append(t.nodes, root)
		t.traversePath(root.Right)
		t.nodes = t.nodes[:len(t.nodes)-1]
	}

	if root.Left == nil && root.Right == nil {

		for _, n := range t.nodes {
			t.b.WriteString(strconv.Itoa(n.Val))
			t.b.WriteString("->")
		}
		t.b.WriteString(strconv.Itoa(root.Val))
		t.paths = append(t.paths, t.b.String())
		t.b.Reset()
	}

}

func (t *TreePath) hasPathSum(root *TreeNode, targetSum int) bool {
	if root.Left != nil {
		t.nodes = append(t.nodes, root)
		if t.hasPathSum(root.Left, targetSum) {
			return true
		}
		t.nodes = t.nodes[:len(t.nodes)-1]
	}

	if root.Right != nil {
		t.nodes = append(t.nodes, root)
		if t.hasPathSum(root.Right, targetSum) {
			return true
		}
		t.nodes = t.nodes[:len(t.nodes)-1]
	}

	if root.Left == nil && root.Right == nil {
		var sum int = root.Val
		for _, n := range t.nodes {
			sum += n.Val
		}

		if sum == targetSum {
			return true
		}
	}

	return false
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil && targetSum == 0 {
		return true
	}
	if root == nil {
		return false
	}
	t := &TreePath{
		nodes: make([]*TreeNode, 0),
		paths: nil,
		b:     nil,
	}

	return t.hasPathSum(root, targetSum)
}

func sumOfLeftLeaves(root *TreeNode) int {
	if root == nil {
		return 0
	}
	q := newQueue()
	q.push(root)

	var sum int

	for !q.empty() {
		size := q.size()
		for i := 0; i < size; i++ {
			node := q.pop()

			if node.Right != nil {
				q.push(node.Right)
			}

			// 有没有左节点是叶子结点
			if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
				sum += node.Left.Val
			} else if node.Left != nil {
				q.push(node.Left)
			}
		}
	}

	return sum
}

func findBottomLeftValue(root *TreeNode) int {
	if root == nil {
		return 0
	}
	q := newQueue()
	q.push(root)

	var leftMost = root

	for !q.empty() {
		size := q.size()
		for i := 0; i < size; i++ {
			node := q.pop()
			if i == 0 {
				leftMost = node
			}
			if node.Left != nil {
				q.push(node.Left)
			}
			if node.Right != nil {
				q.push(node.Right)
			}

		}
	}

	return leftMost.Val
}

func constructMaximumBinaryTree(nums []int) *TreeNode {
	return buildMaximumBinaryTree(nums, 0, len(nums))
}

func buildMaximumBinaryTree(nums []int, left, right int) *TreeNode {
	if left >= right {
		return nil
	}

	index := maxIndex(nums, left, right)
	root := &TreeNode{
		Val: nums[index],
	}
	root.Left = buildMaximumBinaryTree(nums, left, index)
	root.Right = buildMaximumBinaryTree(nums, index+1, right)
	return root
}

func maxIndex(nums []int, left, right int) int {
	index := left
	for i := left; i < right; i++ {
		if nums[i] > nums[index] {
			index = i
		}
	}

	return index
}

func buildTree(inorder []int, postorder []int) *TreeNode {
	root := &TreeNode{Val: inorder[0]}

	return root
}

func buildTreeFrom2Array(inorder []int, postorder []int) *TreeNode {
	root := &TreeNode{Val: inorder[0]}
	return root
}

func getAllElements(root1 *TreeNode, root2 *TreeNode) []int {
	a := &AllElementsArray{
		elements: make([]int, 0),
	}

	a.traverseMiddle(root1)
	nums1 := make([]int, 0, len(a.elements))
	copy(nums1[:], a.elements[:])
	a.elements = make([]int, 0)
	a.traverseMiddle(root2)
	nums2 := make([]int, 0, len(a.elements))
	copy(nums2[:], a.elements[:])

	return mergeArray(nums1, nums2)
}

func mergeArray(nums1, nums2 []int) []int {
	result := make([]int, len(nums1)+len(nums2))

	var h, k int

	for i := 0; i < len(result); i++ {
		if h == len(nums1) {
			// log.Println(h, k, i)
			// result[i] = nums2[k]
			// k++
			// continue

			copy(result[h+k:], nums2[k:])
			break
		}
		if k == len(nums2) {
			// result[i] = nums1[h]
			// h++
			// continue
			copy(result[h+k:], nums1[h:])
			break
		}
		// pick one
		if nums1[h] < nums2[k] {
			result[i] = nums1[h]
			h++
		} else {
			result[i] = nums2[k]
			k++
		}
	}

	// 归并

	return result
}

type AllElementsArray struct {
	elements []int
}

func (a *AllElementsArray) traverseMiddle(root *TreeNode) {
	if root == nil {
		return
	}

	a.traverseMiddle(root.Left)
	a.elements = append(a.elements, root.Val)
	a.traverseMiddle(root.Right)
}

type Codec struct {
	builder *strings.Builder
}

func Constructor() Codec {
	return Codec{builder: &strings.Builder{}}
}

// Serializes a tree to a single string.
func (c *Codec) serialize(root *TreeNode) string {
	return c.levelOrder(root)
}

func (c *Codec) levelOrder(root *TreeNode) string {

	c.builder.Reset()

	if root == nil {
		return "[]"
	}
	c.builder.WriteByte('[')

	nodes := make([]*TreeNode, 0)
	nodes = append(nodes, root)

	for len(nodes) != 0 {
		length := len(nodes)

		for i := 0; i < length; i++ {
			node := nodes[i]
			if node == nil {
				c.builder.WriteString("null")
				c.builder.WriteByte(',')
				continue
			}

			c.builder.WriteString(strconv.Itoa(node.Val))
			c.builder.WriteByte(',')

			nodes = append(nodes, node.Left, node.Right)
		}

		nodes = nodes[length:]
	}

	res := c.builder.String()
	res = res[:len(res)-1]

	return res + "]"

}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	if data == "[]" {
		return nil
	}

	data = data[1 : len(data)-1]

	digits := strings.Split(data, ",")

	nodes := make([]*TreeNode, 0, len(data))

	for _, digit := range digits {
		var node *TreeNode
		if digit != "null" {
			val, _ := strconv.Atoi(digit)
			node = &TreeNode{Val: val}
		}

		nodes = append(nodes, node)
	}

	var m int

	for i, node := range nodes {

		if node != nil {
			node.Left = nodes[2*(i-m)+1]
			node.Right = nodes[2*(i-m)+2]
		} else {
			m++
		}
	}

	return nodes[0]

}

func buildBinaryTree(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}

	root := &TreeNode{
		Val: nums[right],
	}

	if left != right {

		var i int
		// 找分割点，也就是数组中第一个大于right的位置
		for i = left; i < right; i++ {
			if nums[i] > nums[right] {
				break
			}
		}

		root.Left = buildBinaryTree(nums, left, i-1)
		root.Right = buildBinaryTree(nums, i, right-1)
	}

	return root

}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
type ValidBst struct {
	currentMin int64
}

func (v *ValidBst) isValidBST(root *TreeNode) bool {
	if root == nil {
		return true
	}

	// 先左子树
	leftValid := v.isValidBST(root.Left)

	if root.Val <= int(v.currentMin) {
		return false
	}

	v.currentMin = int64(root.Val)

	rightValid := v.isValidBST(root.Right)

	return leftValid && rightValid

}

func isValidBST(root *TreeNode) bool {
	v := &ValidBst{
		currentMin: math.MinInt64,
	}

	return v.isValidBST(root)
}

func findFrequentTreeSum(root *TreeNode) []int {
	t := &traversal{
		stats: make(map[int]int),
	}

	t.traverse(root)

	result := make([]int, 0)

	for sum, v := range t.stats {
		if v == t.maxCount {
			result = append(result, sum)
		}
	}

	return result

}

type traversal struct {
	stats    map[int]int
	maxCount int
}

func (t *traversal) traverse(root *TreeNode) int {
	if root == nil {
		return 0
	}

	leftSum := t.traverse(root.Left)
	rightSum := t.traverse(root.Right)

	rootSum := leftSum + rightSum + root.Val

	v, ok := t.stats[rootSum]
	if !ok {
		v = 1
	} else {
		v++
	}

	t.stats[rootSum] = v

	if v > t.maxCount {
		t.maxCount = v
	}

	return rootSum
}

func isSubStructure(a *TreeNode, b *TreeNode) bool {

	if a == nil && b == nil {
		return true
	}

	if a == nil && b != nil {

		return false
	}
	if a != nil && b == nil {
		return false
	}

	if a.Val == b.Val {
		match := checkSubStructure(a, b)
		if match {
			return true
		}
	}

	left := isSubStructure(a.Left, b)
	right := isSubStructure(a.Right, b)

	return left || right
}

func checkSubStructure(a *TreeNode, b *TreeNode) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil && b != nil {

		return false
	}
	if a != nil && b == nil {
		return false
	}

	if a.Val != b.Val {
		return false
	}

	if b.Left == nil {
		return checkSubStructure(a.Right, b.Right)
	}

	if b.Right == nil {
		return checkSubStructure(a.Left, b.Left)
	}

	return checkSubStructure(a.Left, b.Left) && checkSubStructure(a.Right, b.Right)
}

func mirrorTree(root *TreeNode) *TreeNode {
	var n *TreeNode

	if root != nil {
		n = &TreeNode{Val: root.Val}
		n.Left = mirrorTree(root.Right)
		n.Right = mirrorTree(root.Left)
	}

	return n
}

func levelOrder(root *TreeNode) [][]int {
	res := make([][]int, 0)
	if root == nil {
		return res
	}
	var nodes = []*TreeNode{root}

	for len(nodes) > 0 {
		length := len(nodes)

		resl := make([]int, length)

		var cur int

		for i := 0; i < length; i++ {
			n := nodes[i]

			resl[cur] = n.Val
			cur++

			if n.Left != nil {

				nodes = append(nodes, n.Left)
			}
			if n.Right != nil {
				nodes = append(nodes, n.Right)
			}

		}

		if len(nodes) > length {
			nodes = nodes[length:]
		} else {
			nodes = nil
		}

		res = append(res, resl)

	}

	return res
}

func pathSum(root *TreeNode, target int) [][]int {
	p := &PathSumExecutor{result: make([][]int, 0)}
	digits := make([]int, 0)
	p.getPathSum(root, target, 0, digits)
	return p.result
}

type PathSumExecutor struct {
	result [][]int
}

func (p *PathSumExecutor) getPathSum(root *TreeNode, target int, cur int, digits []int) {
	if root == nil {
		return
	}

	if root.Left == nil && root.Right == nil {
		if cur+root.Val == target {
			digits = append(digits, root.Val)
			dst := make([]int, len(digits))
			copy(dst, digits)

			p.result = append(p.result, dst)
		}
		return
	}

	cur += root.Val
	digits = append(digits, root.Val)

	if root.Left != nil {
		p.getPathSum(root.Left, target, cur, digits)
	}

	if root.Right != nil {
		p.getPathSum(root.Right, target, cur, digits)
	}

	return
}

func kthLargest(root *TreeNode, k int) int {
	v, _, _ := getKthLargest(root, k, 0)
	return v
}

func getKthLargest(root *TreeNode, k int, cur int) (int, int, bool) {
	if root == nil {
		return 0, cur, false
	}
	v, rcur, found := getKthLargest(root.Right, k, cur)
	if found {
		return v, rcur, found
	}

	cur = rcur
	cur += 1

	if cur == k {

		return root.Val, cur, true
	}

	return getKthLargest(root.Left, k, cur)

}

func add(a int, b int) int {
	if b == 0 {
		return a
	}

	return add(a^b, int(uint(a&b)<<1))
}

func diameterOfBinaryTree(root *TreeNode) int {
	var diameter int = 1

	depthAndDiameter(root, &diameter)

	return diameter - 1
}

func depthAndDiameter(root *TreeNode, diameter *int) int {
	if root == nil {
		return 0
	}

	leftDepth := depthAndDiameter(root.Left, diameter)
	rightDepth := depthAndDiameter(root.Right, diameter)

	curDepth := max(leftDepth, rightDepth) + 1

	curDiameter := leftDepth + rightDepth + 1

	curMax := max(curDiameter, *diameter)
	*diameter = curMax
	// log.Printf("cur: %d, left: %d, right: %d, depth: %d, diameter: %d", root.Val, leftDepth, rightDepth, curDepth, *diameter)
	return curDepth

}

package main

func main() {

}

func findK(nums []int, k int) int {
	var count int
	for _, v := range nums {
		if k == v {
			count++
		}
	}

	return count
}

type TreeNode struct {
	// 序号
	Serial int

	Nodes []*TreeNode
	// 边的权重
	Weight int
}

func maxDist(root *TreeNode) int {
	var dist int
	var cur int
	traverse(root, &dist, &cur)

	return dist
}

func traverse(root *TreeNode, dist *int, cur *int) {
	if root == nil {
		return
	}
	*cur += root.Weight
	if len(root.Nodes) == 0 {
		if *cur > *dist {
			*dist = *cur
		}

		*cur -= root.Weight

		return
	}

	for _, node := range root.Nodes {
		traverse(node, dist, cur)
	}

	*cur -= root.Weight

}

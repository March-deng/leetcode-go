package tree

import (
	"fmt"
	"log"
	"testing"
)

func constructTree() *TreeNode {
	return &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val: 2,
		},
		Right: &TreeNode{
			Val:   3,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 5},
		},
	}
}

func TestCodecSerialize(t *testing.T) {
	codec := Constructor()
	result := codec.serialize(constructTree())
	log.Print(result)
}

func TestCodecDeserialize(t *testing.T) {
	codec := Constructor()
	root := codec.deserialize("[1,2,3,null,null,4,5,null,null,null,null]")
	log.Println(levelOrder(root))
}

func TestCountNodes(t *testing.T) {
	root := constructTree()

	log.Println(countNodes(root))
}

func TestIsBalanced(t *testing.T) {
	root := constructTree()
	log.Println(isBalanced(root))
}

func TestSumOfLeftLeaves(t *testing.T) {
	root := constructTree()
	log.Println(sumOfLeftLeaves(root))
}

func TestFindBottomLeftValue(t *testing.T) {
	root := constructTree()

	log.Println(findBottomLeftValue(root))
}

func TestMergeArray(t *testing.T) {
	log.Println(mergeArray([]int{1, 2, 3, 4}, []int{5, 6, 7, 8}))

	log.Println(mergeArray([]int{1, 2, 7, 8}, []int{3, 4, 5, 6}))
}

func TestIsPalindrome(t *testing.T) {
	head := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
		}}

	log.Println(isPalindrome(head))
}

func TestIsValidBST(t *testing.T) {
	root := constructTree()

	log.Println(isValidBST(root))
}

func construtList() *ListNode {
	return &ListNode{
		// Next: &ListNode{
		// 	Val: 2,
		// 	Next: &ListNode{
		// 		Val: 3,
		// 		Next: &ListNode{
		// 			Val: 4,
		// 			Next: &ListNode{
		// 				Val: 5,
		// 			},
		// 		},
		// 	},
		// },
		Val: 1,
	}
}

func TestMiddleNode(t *testing.T) {
	log.Println(middleNode(construtList()))
}

func TestOddEvenList(t *testing.T) {
	head := oddEvenList(construtList())

	for head != nil {
		fmt.Println(head.Val)
		head = head.Next
	}
}

func TestDeplicateZeros(t *testing.T) {

}

func TestGetKthFromEnd(t *testing.T) {
	list := &ListNode{Val: 1, Next: &ListNode{Val: 2, Next: &ListNode{Val: 3, Next: &ListNode{Val: 4, Next: &ListNode{Val: 5}}}}}
	head := getKthFromEnd(list, 2)

	for head != nil {
		fmt.Println(head.Val)
		head = head.Next
	}
}

func TestIsSubStructure(t *testing.T) {
	a := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 4,
			Left: &TreeNode{
				Val: 1,
			},
			Right: &TreeNode{
				Val: 2,
			},
		},
		Right: &TreeNode{Val: 5},
	}
	b := &TreeNode{Val: 4, Left: &TreeNode{Val: 1}}

	log.Println(isSubStructure(a, b))
}

func TestCheckSubStructure(t *testing.T) {
	a := &TreeNode{Val: 4, Left: &TreeNode{Val: 1}, Right: &TreeNode{Val: 3}}
	b := &TreeNode{Val: 4, Left: &TreeNode{Val: 1}}

	log.Println(checkSubStructure(a, b))
}

func TestLevelOrder(t *testing.T) {
	tree := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 9,
			Left: &TreeNode{
				Val: 12,
			},
			Right: &TreeNode{
				Val: 8,
				Right: &TreeNode{
					Val: 10,
				},
			},
		},
		Right: &TreeNode{
			Val: 20,
			Left: &TreeNode{
				Val: 15,
				Left: &TreeNode{
					Val: 17,
				},
				Right: &TreeNode{
					Val: 18,
				},
			},
			Right: &TreeNode{
				Val:   7,
				Right: &TreeNode{Val: 40},
			},
		},
	}

	log.Println(levelOrder(tree))
}

func TestPathSum(t *testing.T) {
	tree := &TreeNode{
		Val: -6,
		Right: &TreeNode{
			Val: -3,
			Left: &TreeNode{
				Val:  -6,
				Left: &TreeNode{Val: -6},
				Right: &TreeNode{
					Val:   -5,
					Left:  &TreeNode{Val: 1},
					Right: &TreeNode{Val: 7},
				},
			},
			Right: &TreeNode{
				Val: 0,
				Left: &TreeNode{
					Val: 4,
				},
			},
		},
	}

	resut := pathSum(tree, -21)
	log.Println(resut)
}

func TestKthLargest(t *testing.T) {
	root := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val:   1,
			Right: &TreeNode{Val: 2},
		},
		Right: &TreeNode{Val: 4},
	}
	log.Println(kthLargest(root, 2))
}

func TestAddNum(t *testing.T) {
	log.Println(add(3, 5))
}

func TestDiameterOfBinaryTree(t *testing.T) {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 5},
		},
		Right: &TreeNode{
			Val: 3,
		},
	}
	log.Println(diameterOfBinaryTree(root))
}

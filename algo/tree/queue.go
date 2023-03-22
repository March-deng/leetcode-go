package tree

type queue struct {
	elements []*TreeNode
}

func newQueue() *queue {
	return &queue{
		elements: make([]*TreeNode, 0),
	}
}

// push from back
func (q *queue) push(node *TreeNode) {
	q.elements = append(q.elements, node)
}

// pop from head
func (q *queue) pop() *TreeNode {
	if len(q.elements) == 0 {
		return nil
	}

	top := q.elements[0]
	q.elements = q.elements[1:]
	return top
}

func (q *queue) size() int {
	return len(q.elements)
}

func (q *queue) empty() bool {
	return q.size() == 0
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func isPalindrome(head *ListNode) bool {
	// 找中心节点
	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	if fast != nil {
		slow = slow.Next
	}

	slow = reverse(slow)

	fast = head

	for slow != nil {
		if fast.Val != slow.Val {
			return false
		}

		slow = slow.Next
		fast = fast.Next
	}

	return true
}

// 反转链表
func reverse(head *ListNode) *ListNode {
	var pre *ListNode
	cur := head

	for cur != nil {
		next := cur.Next

		cur.Next = pre
		pre = cur
		cur = next
	}

	return pre
}

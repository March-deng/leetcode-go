package tree

func middleNode(head *ListNode) *ListNode {
	slow := head
	fast := head
	for {
		if fast == nil || fast.Next == nil {
			break
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// 以索引分割奇偶
func oddEvenList(head *ListNode) *ListNode {

	if head == nil {
		return head
	}

	evenHead := &ListNode{}

	evenCur := evenHead

	cur := head
	for cur != nil && cur.Next != nil {

		evenCur.Next = cur.Next
		// 将next删除
		cur.Next = cur.Next.Next

		if cur.Next != nil {
			cur = cur.Next
		}

		evenCur = evenCur.Next
	}

	evenCur.Next = nil

	cur.Next = evenHead.Next

	return head
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	pa, pb := headA, headB
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}
	return pa
}

func deleteNode(head *ListNode, val int) *ListNode {
	cur := head
	headPre := &ListNode{
		Next: head,
	}
	pre := headPre
	for cur != nil {
		// 删除
		if val == cur.Val {
			pre.Next = cur.Next
			break
		}
		pre = cur
		cur = cur.Next
	}

	return headPre.Next
}

func getKthFromEnd(head *ListNode, k int) *ListNode {
	var kthNode = head
	var length = 0

	for kthNode != nil && length < k {
		// fmt.Println(kthNode.Val)
		kthNode = kthNode.Next
		length++
	}

	cur := head

	for kthNode != nil {
		// fmt.Println(kthNode.Val)
		cur = cur.Next
		kthNode = kthNode.Next

	}

	return cur

}

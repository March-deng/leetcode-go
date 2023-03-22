package graph

func arrayNesting(nums []int) int {

	var result int

	set := make(map[int]struct{})

	for i := 0; i < len(nums); i++ {
		cur := nums[i]
		for {
			_, ok := set[cur]
			if ok {
				break
			}

			if cur >= len(nums) {
				break
			}

			set[cur] = struct{}{}
			cur = nums[cur]
		}

		length := len(set)
		if length == len(nums) {
			return len(nums)
		}

		if length > result {
			result = length
		}
		set = make(map[int]struct{})
	}

	return result
}

func permutation(s string) []string {
	p := &Permutator{
		result: make(map[string]struct{}),
		chars:  make([]rune, 0),
		status: make(map[int]bool),
	}

	for i, char := range s {
		p.chars = append(p.chars, char)
		p.status[i] = false
	}
	b := &StringBuilder{chars: make([]rune, 0)}

	for i := range p.chars {

		p.Permutate(b, i)
	}
	result := make([]string, 0, len(p.result))
	for k := range p.result {
		result = append(result, k)
	}
	return result
}

type Permutator struct {
	result map[string]struct{}
	chars  []rune
	status map[int]bool
}

func (p *Permutator) Permutate(builder *StringBuilder, cur int) {
	p.status[cur] = true
	builder.Add(p.chars[cur])
	if builder.Len() == len(p.chars) {
		p.result[builder.String()] = struct{}{}
	} else {
		for i, exist := range p.status {
			if !exist {
				p.Permutate(builder, i)
			}
		}
	}

	builder.Remove()

	p.status[cur] = false
}

type StringBuilder struct {
	chars []rune
}

func (s *StringBuilder) Add(r rune) {
	s.chars = append(s.chars, r)
}

func (s *StringBuilder) Len() int {
	return len(s.chars)
}

func (s *StringBuilder) Remove() {
	if len(s.chars) != 0 {
		s.chars = s.chars[:len(s.chars)-1]
	}
}

func (s *StringBuilder) String() string {
	return string(s.chars)
}

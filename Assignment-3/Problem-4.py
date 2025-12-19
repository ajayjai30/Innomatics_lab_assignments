class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        count = {}
        good_pairs = 0
        for num in nums:
            if num in count:
                good_pairs += count[num]
                count[num] += 1
            else:
                count[num] = 1            
        return good_pairs

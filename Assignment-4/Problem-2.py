class Solution:
    def numTeams(self, rating: List[int]) -> int:
        n = len(rating)
        total_teams = 0
        for j in range(n):
            left_smaller = left_greater = 0
            right_smaller = right_greater = 0
            for i in range(j):
                if rating[i] < rating[j]:
                    left_smaller += 1
                else:
                    left_greater += 1
            for k in range(j + 1, n):
                if rating[k] < rating[j]:
                    right_smaller += 1
                else:
                    right_greater += 1
            total_teams += left_smaller * right_greater
            total_teams += left_greater * right_smaller
            
        return total_teams
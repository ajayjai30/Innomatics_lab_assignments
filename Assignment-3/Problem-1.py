class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        array=[start+2*i for i in range(n)]
        result=0
        for i in array:
            result^=i
        return result
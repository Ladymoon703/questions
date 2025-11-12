from solution import antisymmetric_matvec  # 待会贴给你
a = [1,2,3,4,5,4,3,2,1]   # 2n-1=9 => n=5
v = [1,1,1,1,1]
print(antisymmetric_matvec(a, v))
# 期望输出 [15, 20, 23, 20, 15]
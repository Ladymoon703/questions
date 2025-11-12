import numpy as np

def antisymmetric_matvec(a, v):
    """
    反向对称矩阵-向量乘积（压缩存储版）
    a: 长度 2n-1 的对角线元素 [a_{n-1}, ..., a_0, ..., a_{-(n-1)}]
    v: 长度 n 的向量
    返回: 长度 n 的结果向量
    """
    n = len(v)
    res = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # 第 i 行：A[i, j] = a[n-1 + i - j]
        for j in range(n):
            idx = (n - 1) + i - j
            res[i] += a[idx] * v[j]
    return res.tolist()
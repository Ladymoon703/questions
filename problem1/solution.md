# 题目1: 反向对称矩阵-向量乘积

## 问题描述

设 A 为一 n × n 矩阵。若 A 的每一条从左上到右下的对角线上的元素都相等，则称 A 为**反向对称矩阵**。

对于反向对称矩阵 A，每条对角线上的元素都相等，即 $A_{i,j} = a_{n-1+i-j}$。

## 1. 算法设计

### 伪代码

```
Algorithm: AntiSymmetricMatrixVectorProduct
Input: 压缩表示的反向对称矩阵 (存储为向量 a[0..2n-2])，向量 v[0..n-1]
Output: 矩阵-向量乘积 result = A · v

1. Initialize result[0..n-1] = 0
2. For k = 0 to 2n-2:
     sum = 0
     # 确定对角线上元素的位置
     For i = 0 to n-1:
         j = i - (k - (n-1))
         If 0 <= j < n:
             sum += v[j]
     result 中对应位置 += a[k] * sum
3. Return result
```

### 优化算法

由于反向对称矩阵的特殊结构，我们可以更高效地计算：

```python
def antisymmetric_matvec(a, v):
    """
    a: 长度为 2n-1 的数组，存储对角线元素
    v: 长度为 n 的向量
    返回: A·v
    """
    n = len(v)
    result = [0] * n
    
    # 对于每条对角线 k (k从0到2n-2)
    for k in range(2 * n - 1):
        diag_idx = k - (n - 1)  # 对角线索引
        
        # 计算该对角线对结果的贡献
        for i in range(n):
            j = i - diag_idx
            if 0 <= j < n:
                result[i] += a[k] * v[j]
    
    return result
```

### 算法说明

1. **压缩存储**: 反向对称矩阵可以用一个长度为 2n-1 的向量存储，因为只需要存储每条对角线的唯一值

2. **计算过程**: 
   - 遍历每条对角线
   - 对每条对角线，找到该对角线上的所有元素位置
   - 累加该对角线对结果向量的贡献

3. **示例** (n=5):
   ```
   矩阵结构:
   a₄  a₃  a₂  a₁  a₀
   a₅  a₄  a₃  a₂  a₁
   a₆  a₅  a₄  a₃  a₂
   a₇  a₆  a₅  a₄  a₃
   a₈  a₇  a₆  a₅  a₄
   ```

## 2. 复杂度分析

### 时间复杂度

**上界证明**: O(n²)

- 外层循环: 遍历 2n-1 条对角线，O(n)
- 内层循环: 对每条对角线，最多遍历 n 个元素，O(n)
- 总时间: O(n) × O(n) = O(n²)

### 空间复杂度

- **存储矩阵**: O(n) - 只需存储 2n-1 个对角线值
- **辅助空间**: O(n) - 结果向量
- **总空间**: O(n)

相比普通矩阵-向量乘法：
- 时间复杂度相同: O(n²)
- 空间复杂度优化: O(n) vs O(n²)

### 优化空间

可以进一步优化到 O(n log n)，通过以下方法：
- 使用卷积的性质
- 利用FFT加速卷积运算
- 将问题转化为多项式乘法

但在本题的基本要求下，O(n²)的算法已经是高效且直观的解决方案。

## 3. 正确性证明

**定理**: 算法正确计算 A·v。

**证明**:
1. 矩阵-向量乘积的定义: $(A \cdot v)_i = \sum_{j=0}^{n-1} A_{i,j} \cdot v_j$

2. 对于反向对称矩阵: $A_{i,j} = a_{n-1+i-j}$

3. 因此: $(A \cdot v)_i = \sum_{j=0}^{n-1} a_{n-1+i-j} \cdot v_j$

4. 令 $k = n-1+i-j$，则对每个 k，我们累加所有满足条件的 $a_k \cdot v_j$

5. 算法正是按这个逻辑实现，遍历每个对角线 k，累加其对结果的贡献

因此算法正确。□

## 参考文献

- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.
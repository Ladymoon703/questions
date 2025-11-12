# 题目2: FFT字符串匹配

## 问题描述

给定主串 S（长度为 n）和模式串 P（长度为 m），P 中可能包含通配符 `?`（可以匹配任意单个字符）。使用快速傅里叶变换（FFT）算法找出 P 在 S 中的所有匹配位置。

要求时间复杂度: O(n log n)

## 算法原理

### 1. 核心思想

将字符串匹配问题转化为**多项式乘法**问题，利用FFT加速卷积运算。

### 2. 匹配条件

对于位置 i，如果 P 在 S[i..i+m-1] 处匹配，则需要满足：
- 对所有 j ∈ [0, m-1]：P[j] = '?' 或 P[j] = S[i+j]

### 3. 数学建模

将字符映射为数值：
- 字母 'a' → 1, 'b' → 2, ..., 'z' → 26
- 通配符 '?' → 0

定义匹配度函数：
$$
M(i) = \sum_{j=0}^{m-1} (S[i+j] - P[j])^2 \cdot mask[j]
$$

其中 mask[j] = 1 当 P[j] ≠ '?'，否则为 0。

**关键性质**: M(i) = 0 当且仅当位置 i 完全匹配

### 4. 展开公式

$$
M(i) = \sum_{j=0}^{m-1} mask[j] \cdot S[i+j]^2 - 2 \sum_{j=0}^{m-1} mask[j] \cdot P[j] \cdot S[i+j] + \sum_{j=0}^{m-1} mask[j] \cdot P[j]^2
$$

这可以分解为三个部分：
1. **项1**: S² 与 mask 的卷积
2. **项2**: S 与 (P·mask) 的卷积
3. **项3**: P²·mask 的和（常数）

### 5. FFT加速

每个卷积都可以用FFT在 O(n log n) 时间内计算：

```
卷积定理: conv(A, B) = IFFT(FFT(A) · FFT(B))
```

## 算法步骤

```
Algorithm: FFT_Pattern_Matching
Input: 文本 S[0..n-1], 模式 P[0..m-1]
Output: 所有匹配位置的列表

1. 将字符串转换为数值数组
   S_nums ← char_to_num(S)
   P_nums ← char_to_num(P)

2. 创建掩码数组
   mask[j] ← 1 if P[j] ≠ '?' else 0

3. 反转模式串（用于卷积）
   P_rev ← reverse(P_nums)
   mask_rev ← reverse(mask)

4. 使用FFT计算三个卷积
   conv1 ← FFT_Conv(S_nums², mask_rev)
   conv2 ← FFT_Conv(S_nums, (P_nums · mask)_rev)
   const ← sum(P_nums² · mask)

5. 计算匹配度
   For each position i:
       score[i] ← conv1[i] - 2·conv2[i] + const

6. 找出匹配位置
   matches ← {i | score[i] ≈ 0}

7. Return matches
```

## 复杂度分析

### 时间复杂度

**总体: O(n log n)**

详细分解：
- 字符串转数值: O(n + m) = O(n)
- 三次FFT卷积: 每次 O(n log n)
  - FFT: O(n log n)
  - 逐点乘法: O(n)
  - IFFT: O(n log n)
- 查找匹配位置: O(n)

**总时间**: 3 × O(n log n) + O(n) = **O(n log n)**

### 空间复杂度

- 原始数组: O(n + m)
- FFT中间结果: O(n)（可复用）
- 卷积结果: O(n)

**总空间**: **O(n)**

### 与朴素算法对比

| 算法 | 时间复杂度 | 空间复杂度 | 支持通配符 |
|------|-----------|-----------|----------|
| 朴素匹配 | O(nm) | O(1) | ✓ |
| KMP | O(n+m) | O(m) | ✗ |
| **FFT匹配** | **O(n log n)** | **O(n)** | **✓** |

## 实现细节

### 1. 处理浮点精度

由于FFT使用浮点运算，需要容差判断：

```python
epsilon = 1e-6
if abs(score[i]) < epsilon:
    # 匹配成功
```

### 2. 优化FFT性能

将数组填充到2的幂次：

```python
size = 2 ** ceil(log2(n + m - 1))
```

### 3. 边界情况

- m > n: 直接返回空列表
- P 全是通配符: 所有位置都匹配
- P 无通配符: 标准字符串匹配

## 示例

### 示例1: 无通配符

```
文本: "abracadabra"
模式: "abra"

转换:
S = [1,2,18,1,3,1,4,1,2,18,1]
P = [1,2,18,1]

匹配位置: [0, 7]
验证: "abra", "abra" ✓
```

### 示例2: 带通配符

```
文本: "abcdefgh"
模式: "d?f"

转换:
S = [1,2,3,4,5,6,7,8]
P = [4,0,6]
mask = [1,0,1]

匹配位置: [3]
验证: "def" (中间e可以是任意字符) ✓
```

## 扩展应用

这个算法可以扩展到：
1. **多模式匹配**: 同时搜索多个模式
2. **近似匹配**: 允许k个字符不同
3. **二维模式匹配**: 在图像中查找模式
4. **DNA序列匹配**: 生物信息学应用

## 参考文献

1. Fischer, M. J., & Paterson, M. S. (1974). String-Matching and Other Products. *SIAM-AMS Proceedings*, 7.
2. Clifford, P., & Clifford, R. (2007). Simple deterministic wildcard matching. *Information Processing Letters*, 101(2), 53-54.

## 运行示例

```bash
python fft_pattern_matching.py
```

输出：
```
=== FFT字符串匹配演示 ===

文本: abracadabra
模式: abra
匹配位置: [0, 7]
验证: ['abra', 'abra']

文本: abcdefghijklmn
模式: d?f
匹配位置: [3]
验证: ['def']
```
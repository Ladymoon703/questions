"""
题目2: 使用FFT进行字符串匹配
使用快速傅里叶变换实现模式匹配，支持通配符
"""

import numpy as np


def fft_pattern_matching(text, pattern):
    """
    使用FFT在文本中查找所有模式匹配位置
    
    参数:
        text: 主串S (长度为n)
        pattern: 模式串P (长度为m)，可包含通配符'?'
    
    返回:
        matches: 所有匹配位置的列表
    
    时间复杂度: O(n log n)
    """
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    # 将字符串转换为数值表示
    # 字母映射到1-26，通配符映射到0
    def char_to_num(c):
        if c == '?':
            return 0
        return ord(c.lower()) - ord('a') + 1
    
    text_nums = np.array([char_to_num(c) for c in text])
    pattern_nums = np.array([char_to_num(c) for c in pattern])
    
    # 反转模式串（用于卷积）
    pattern_reversed = pattern_nums[::-1]
    
    # 创建掩码（标记哪些位置不是通配符）
    pattern_mask = (pattern_nums != 0).astype(int)
    pattern_mask_reversed = pattern_mask[::-1]
    
    # 使用三个卷积来计算匹配度
    # 匹配公式: sum((T[i] - P[j])^2 * mask[j]) = 0 表示完全匹配
    # 展开: sum(T[i]^2 * mask[j]) - 2*sum(T[i]*P[j]*mask[j]) + sum(P[j]^2*mask[j])
    
    # 项1: T^2 与 mask 的卷积
    conv1 = np.convolve(text_nums ** 2, pattern_mask_reversed, mode='valid')
    
    # 项2: T 与 P*mask 的卷积
    pattern_weighted = pattern_nums * pattern_mask
    conv2 = np.convolve(text_nums, pattern_weighted[::-1], mode='valid')
    
    # 项3: P^2*mask 的和（常数）
    const = np.sum((pattern_nums ** 2) * pattern_mask)
    
    # 计算匹配度
    matching_scores = conv1 - 2 * conv2 + const
    
    # 找出匹配位置（误差接近0的位置）
    matches = []
    epsilon = 1e-6  # 浮点数比较容差
    
    for i in range(len(matching_scores)):
        if abs(matching_scores[i]) < epsilon:
            matches.append(i)
    
    return matches


def fft_pattern_matching_optimized(text, pattern):
    """
    优化版本：使用FFT加速卷积运算
    
    时间复杂度: O(n log n)
    """
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    # 字符转数值
    def char_to_num(c):
        if c == '?':
            return 0
        return ord(c.lower()) - ord('a') + 1
    
    text_nums = np.array([char_to_num(c) for c in text])
    pattern_nums = np.array([char_to_num(c) for c in pattern])
    
    # 反转模式串
    pattern_reversed = pattern_nums[::-1]
    pattern_mask = (pattern_nums != 0).astype(int)
    pattern_mask_reversed = pattern_mask[::-1]
    
    # 填充到2的幂次，以优化FFT性能
    size = 2 ** int(np.ceil(np.log2(n + m - 1)))
    
    # 使用FFT进行快速卷积
    def fast_convolve(a, b):
        a_padded = np.pad(a, (0, size - len(a)))
        b_padded = np.pad(b, (0, size - len(b)))
        
        fft_a = np.fft.fft(a_padded)
        fft_b = np.fft.fft(b_padded)
        
        result = np.fft.ifft(fft_a * fft_b).real
        return result[:n - m + 1]
    
    # 三个卷积项
    conv1 = fast_convolve(text_nums ** 2, pattern_mask_reversed)
    conv2 = fast_convolve(text_nums, (pattern_nums * pattern_mask)[::-1])
    const = np.sum((pattern_nums ** 2) * pattern_mask)
    
    matching_scores = conv1 - 2 * conv2 + const
    
    # 找出匹配位置
    matches = []
    epsilon = 1e-6
    
    for i in range(len(matching_scores)):
        if abs(matching_scores[i]) < epsilon:
            matches.append(i)
    
    return matches


def demo():
    """演示函数"""
    print("=== FFT字符串匹配演示 ===\n")
    
    # 测试1: 无通配符
    text1 = "abracadabra"
    pattern1 = "abra"
    matches1 = fft_pattern_matching(text1, pattern1)
    print(f"文本: {text1}")
    print(f"模式: {pattern1}")
    print(f"匹配位置: {matches1}")
    print(f"验证: {[text1[i:i+len(pattern1)] for i in matches1]}\n")
    
    # 测试2: 带通配符
    text2 = "abcdefghijklmn"
    pattern2 = "d?f"
    matches2 = fft_pattern_matching(text2, pattern2)
    print(f"文本: {text2}")
    print(f"模式: {pattern2}")
    print(f"匹配位置: {matches2}")
    print(f"验证: {[text2[i:i+len(pattern2)] for i in matches2]}\n")
    
    # 测试3: 多个通配符
    text3 = "aaabbbcccaaabbb"
    pattern3 = "a?b"
    matches3 = fft_pattern_matching(text3, pattern3)
    print(f"文本: {text3}")
    print(f"模式: {pattern3}")
    print(f"匹配位置: {matches3}")
    print(f"验证: {[text3[i:i+len(pattern3)] for i in matches3]}\n")
    
    # 性能比较
    print("=== 性能测试 ===")
    import time
    
    large_text = "a" * 10000 + "pattern" * 100 + "b" * 10000
    large_pattern = "p?t?e?n"
    
    start = time.time()
    matches_opt = fft_pattern_matching_optimized(large_text, large_pattern)
    time_opt = time.time() - start
    
    print(f"文本长度: {len(large_text)}")
    print(f"模式长度: {len(large_pattern)}")
    print(f"找到 {len(matches_opt)} 个匹配")
    print(f"FFT算法耗时: {time_opt:.4f}秒")
    print(f"理论复杂度: O(n log n) = O({len(large_text)} × {np.log2(len(large_text)):.2f})")


if __name__ == "__main__":
    demo()
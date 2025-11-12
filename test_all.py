"""
测试脚本 - 检查所有题目是否正确
"""

def test_problem1():
    """测试题目1：反向对称矩阵"""
    print("\n" + "="*50)
    print("测试题目1：反向对称矩阵")
    print("="*50)
    
    try:
        import numpy as np
        
        # 测试用例：5x5矩阵
        n = 5
        # 对角线元素 a0 到 a8
        diag_elements = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # 构建完整矩阵（用于验证）
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k = (n - 1) + i - j
                A[i, j] = diag_elements[k]
        
        print(f"反向对称矩阵 A (n={n}):")
        print(A)
        
        # 测试向量
        v = np.array([1, 2, 3, 4, 5])
        print(f"\n向量 v: {v}")
        
        # 标准矩阵乘法（用于验证）
        expected = A @ v
        print(f"\n期望结果（标准矩阵乘法）: {expected}")
        
        # 你的算法实现
        def antisymmetric_matvec(a, v):
            n = len(v)
            result = np.zeros(n)
            
            for k in range(2 * n - 1):
                diag_idx = k - (n - 1)
                for i in range(n):
                    j = i - diag_idx
                    if 0 <= j < n:
                        result[i] += a[k] * v[j]
            
            return result
        
        result = antisymmetric_matvec(diag_elements, v)
        print(f"你的算法结果: {result}")
        
        # 验证
        if np.allclose(result, expected):
            print("\n✅ 题目1测试通过！")
            return True
        else:
            print("\n❌ 题目1测试失败！")
            print(f"差异: {result - expected}")
            return False
            
    except Exception as e:
        print(f"\n❌ 题目1运行错误: {e}")
        return False


def test_problem2():
    """测试题目2：FFT字符串匹配"""
    print("\n" + "="*50)
    print("测试题目2：FFT字符串匹配")
    print("="*50)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'problem2'))
        
        from fft_pattern_matching import fft_pattern_matching
        
        # 测试用例1：无通配符
        text1 = "abracadabra"
        pattern1 = "abra"
        matches1 = fft_pattern_matching(text1, pattern1)
        expected1 = [0, 7]
        
        print(f"测试1 - 文本: {text1}")
        print(f"       模式: {pattern1}")
        print(f"       结果: {matches1}")
        print(f"       期望: {expected1}")
        
        if matches1 == expected1:
            print("       ✅ 通过")
            test1_pass = True
        else:
            print("       ❌ 失败")
            test1_pass = False
        
        # 测试用例2：带通配符
        text2 = "abcdefgh"
        pattern2 = "d?f"
        matches2 = fft_pattern_matching(text2, pattern2)
        expected2 = [3]
        
        print(f"\n测试2 - 文本: {text2}")
        print(f"       模式: {pattern2}")
        print(f"       结果: {matches2}")
        print(f"       期望: {expected2}")
        
        if matches2 == expected2:
            print("       ✅ 通过")
            test2_pass = True
        else:
            print("       ❌ 失败")
            test2_pass = False
        
        # 测试用例3：多个通配符
        text3 = "aaabbbccc"
        pattern3 = "a?b"
        matches3 = fft_pattern_matching(text3, pattern3)
        
        print(f"\n测试3 - 文本: {text3}")
        print(f"       模式: {pattern3}")
        print(f"       结果: {matches3}")
        # 验证每个匹配位置
        valid = all(text3[i:i+3][0] == 'a' and text3[i:i+3][2] == 'b' 
                   for i in matches3)
        
        if valid:
            print("       ✅ 通过")
            test3_pass = True
        else:
            print("       ❌ 失败")
            test3_pass = False
        
        if test1_pass and test2_pass and test3_pass:
            print("\n✅ 题目2测试通过！")
            return True
        else:
            print("\n❌ 题目2部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 题目2运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_problem4():
    """测试题目4：逻辑回归"""
    print("\n" + "="*50)
    print("测试题目4：逻辑回归")
    print("="*50)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'problem4'))
        
        from logistic_regression import LogisticRegression, StockDataManager
        import numpy as np
        
        # 测试逻辑回归基础功能
        print("测试1: Sigmoid函数")
        model = LogisticRegression()
        assert abs(model.sigmoid(0) - 0.5) < 1e-6
        assert model.sigmoid(10) > 0.99
        assert model.sigmoid(-10) < 0.01
        print("  ✅ Sigmoid正确")
        
        # 测试简单二分类
        print("\n测试2: 简单二分类")
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        model = LogisticRegression(learning_rate=0.1, iterations=500)
        model.fit(X_train, y_train)
        
        accuracy = np.mean(model.predict(X_train) == y_train)
        print(f"  训练集准确率: {accuracy:.2f}")
        
        if accuracy > 0.7:
            print("  ✅ 模型训练正常")
            test1_pass = True
        else:
            print("  ❌ 模型训练效果差")
            test1_pass = False
        
        # 测试数据管理器
        print("\n测试3: 数据管理器")
        db_path = 'test_stock.db'
        if os.path.exists(db_path):
            os.remove(db_path)
        
        manager = StockDataManager(db_path)
        df = manager.fetch_and_save_data('000001')
        
        if df is not None and len(df) > 0:
            print(f"  ✅ 成功获取 {len(df)} 条数据")
            test2_pass = True
        else:
            print("  ❌ 数据获取失败")
            test2_pass = False
        
        # 测试特征工程
        print("\n测试4: 特征工程")
        df_features = manager.feature_engineering(df)
        required_cols = ['return', 'ma5', 'ma10', 'target']
        
        if all(col in df_features.columns for col in required_cols):
            print(f"  ✅ 特征工程完成，包含 {len(df_features.columns)} 个特征")
            test3_pass = True
        else:
            print("  ❌ 特征工程缺少必要列")
            test3_pass = False
        
        manager.close()
        os.remove(db_path)
        
        if test1_pass and test2_pass and test3_pass:
            print("\n✅ 题目4测试通过！")
            return True
        else:
            print("\n❌ 题目4部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 题目4运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_project_structure():
    """检查项目结构"""
    print("\n" + "="*50)
    print("检查项目结构")
    print("="*50)
    
    required_files = [
        'README.md',
        '.gitignore',
        'requirements.txt',
        'problem1/solution.md',
        'problem2/fft_pattern_matching.py',
        'problem4/logistic_regression.py'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 缺失")
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    import os
    
    print("\n" + "#"*50)
    print("# 作业完整性检查")
    print("#"*50)
    
    # 检查文件结构
    structure_ok = check_project_structure()
    
    # 运行测试
    results = {
        '题目1': test_problem1(),
        '题目2': test_problem2(),
        '题目4': test_problem4()
    }
    
    # 总结
    print("\n" + "="*50)
    print("测试总结")
    print("="*50)
    print(f"文件结构: {'✅ 完整' if structure_ok else '❌ 不完整'}")
    for name, result in results.items():
        print(f"{name}: {'✅ 通过' if result else '❌ 失败'}")
    
    all_pass = structure_ok and all(results.values())
    
    if all_pass:
        print("\n🎉 所有测试通过！可以提交了")
    else:
        print("\n⚠️  部分测试失败，请检查代码")
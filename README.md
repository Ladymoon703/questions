# 算法作业

这是算法课的4个题目，在branch 001里完成。

**注意**: 这个分支是算法作业，主分支(main)是另一个项目（用户登录系统）。

## 文件说明

```
problem1/  反向对称矩阵那题
problem2/  FFT字符串匹配
problem3/  就是整理了下代码结构
problem4/  逻辑回归预测股票
```

## 运行方法

先装依赖：
```bash
pip install -r requirements.txt
```

然后分别运行：
```bash
# 题目2
python problem2/fft_pattern_matching.py

# 题目4
python problem4/logistic_regression.py
```

题目1是理论分析，看solution.md就行。

题目3就是这个整体的项目结构。

## 环境

- Python 3.8+
- 用到的库：numpy, pandas, akshare

## 说明

题目4的股票数据用的是模拟数据（因为akshare要联网，有时候会超时）。如果要用真实数据，把代码里注释的akshare那部分打开就行。

数据库用的sqlite，会自动创建stock_data.db文件。

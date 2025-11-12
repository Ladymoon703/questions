"""
题目4: 逻辑回归股票预测
使用akshare获取A股数据，实现逻辑回归模型
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os


class LogisticRegression:
    """逻辑回归模型实现"""
    
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.reg = regularization
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.iterations):
            # 前向传播
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # 添加L2正则化
            dw += (self.reg / n_samples) * self.weights
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 记录损失
            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                self.losses.append(loss)
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def compute_loss(self, X, y):
        """计算交叉熵损失"""
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        
        # 交叉熵损失
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # 添加L2正则化项
        loss += (self.reg / (2 * len(y))) * np.sum(self.weights ** 2)
        
        return loss
    
    def predict_proba(self, X):
        """预测概率"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """评估模型"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        # 计算混淆矩阵
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }


class StockDataManager:
    """股票数据管理器"""
    
    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                date TEXT,
                stock_code TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (date, stock_code)
            )
        ''')
        
        self.conn.commit()
        print(f"数据库初始化完成: {self.db_path}")
    
    def fetch_and_save_data(self, stock_code='000001', start_date=None, end_date=None):
        """
        获取并保存股票数据
        
        注意：由于akshare需要网络连接，这里提供模拟数据生成
        实际使用时，取消注释akshare相关代码
        """
        try:
            # 实际使用时的代码：
            # import akshare as ak
            # df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, 
            #                         end_date=end_date, adjust="qfq")
            
            # 模拟数据（用于演示）
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            print(f"生成模拟数据: {stock_code}, {start_date} 至 {end_date}")
            df = self._generate_mock_data(start_date, end_date)
            
            # 保存到数据库
            cursor = self.conn.cursor()
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_data 
                    (date, stock_code, open, high, low, close, volume, amount)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['date'], stock_code, row['open'], row['high'], 
                      row['low'], row['close'], row['volume'], row['amount']))
            
            self.conn.commit()
            print(f"成功保存 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None
    
    def _generate_mock_data(self, start_date, end_date):
        """生成模拟股票数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日
        
        n = len(dates)
        base_price = 10.0
        
        # 使用随机游走生成价格
        returns = np.random.randn(n) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            open_price = close * (1 + np.random.randn() * 0.01)
            high = max(open_price, close) * (1 + abs(np.random.randn() * 0.02))
            low = min(open_price, close) * (1 - abs(np.random.randn() * 0.02))
            volume = np.random.randint(1000000, 10000000)
            amount = volume * close
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'amount': round(amount, 2)
            })
        
        return pd.DataFrame(data)
    
    def load_data(self, stock_code='000001'):
        """从数据库加载数据"""
        query = f"SELECT * FROM daily_data WHERE stock_code = '{stock_code}' ORDER BY date"
        df = pd.read_sql_query(query, self.conn)
        print(f"加载了 {len(df)} 条数据")
        return df
    
    def feature_engineering(self, df):
        """特征工程"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 计算技术指标
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 价格动量
        df['momentum'] = df['close'] - df['close'].shift(5)
        
        # 波动率
        df['volatility'] = df['return'].rolling(window=5).std()
        
        # 成交量变化
        df['volume_change'] = df['volume'].pct_change()
        
        # 目标变量：明天涨跌（1=涨，0=跌）
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 删除缺失值
        df = df.dropna()
        
        return df
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")


def main():
    """主函数"""
    print("=== 股票预测逻辑回归模型 ===\n")
    
    # 1. 初始化数据管理器
    data_manager = StockDataManager('stock_data.db')
    
    # 2. 获取数据
    print("\n步骤1: 获取股票数据")
    df = data_manager.fetch_and_save_data(stock_code='000001')
    
    # 3. 特征工程
    print("\n步骤2: 特征工程")
    df = data_manager.feature_engineering(df)
    print(f"特征列: {df.columns.tolist()}")
    
    # 4. 准备训练数据
    print("\n步骤3: 准备训练数据")
    feature_cols = ['return', 'ma5', 'ma10', 'ma20', 'momentum', 'volatility', 'volume_change']
    X = df[feature_cols].values
    y = df['target'].values
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 5. 训练模型
    print("\n步骤4: 训练逻辑回归模型")
    model = LogisticRegression(learning_rate=0.01, iterations=1000, regularization=0.01)
    model.fit(X_train, y_train)
    
    # 6. 评估模型
    print("\n步骤5: 模型评估")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n训练集表现:")
    print(f"  准确率: {train_metrics['accuracy']:.4f}")
    print(f"  精确率: {train_metrics['precision']:.4f}")
    print(f"  召回率: {train_metrics['recall']:.4f}")
    print(f"  F1分数: {train_metrics['f1_score']:.4f}")
    
    print("\n测试集表现:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1_score']:.4f}")
    
    # 7. 特征重要性
    print("\n特征重要性:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {model.weights[i]:.4f}")
    
    # 关闭数据库
    data_manager.close()
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
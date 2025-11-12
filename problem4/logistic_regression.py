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
    """股票数据管理器 - 支持事务回滚"""
    
    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.isolation_level = None  # 手动控制事务
        cursor = self.conn.cursor()
        
        # 开启事务
        cursor.execute('BEGIN')
        
        try:
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
            
            # 创建索引加速查询
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stock_date 
                ON daily_data(stock_code, date)
            ''')
            
            cursor.execute('COMMIT')
            print(f"✓ 数据库初始化成功: {self.db_path}")
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            print(f"✗ 数据库初始化失败: {e}")
            raise
    
    def fetch_and_save_data(self, stock_code='000001', start_date=None, end_date=None):
        """
        获取并保存股票数据（带事务回滚）
        """
        cursor = self.conn.cursor()
        
        try:
            # 生成数据（实际使用时替换为akshare）
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            print(f"正在获取数据: {stock_code}, {start_date} 至 {end_date}")
            df = self._generate_mock_data(start_date, end_date)
            
            if df is None or len(df) == 0:
                print("✗ 没有获取到数据")
                return None
            
            # 开启事务
            cursor.execute('BEGIN')
            
            # 批量插入数据
            success_count = 0
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_data 
                        (date, stock_code, open, high, low, close, volume, amount)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['date'], stock_code, row['open'], row['high'], 
                          row['low'], row['close'], row['volume'], row['amount']))
                    success_count += 1
                except sqlite3.IntegrityError as e:
                    print(f"⚠ 数据插入警告 {row['date']}: {e}")
                    continue
            
            # 提交事务
            cursor.execute('COMMIT')
            print(f"✓ 成功保存 {success_count}/{len(df)} 条数据")
            return df
            
        except Exception as e:
            # 回滚事务
            cursor.execute('ROLLBACK')
            print(f"✗ 数据保存失败，已回滚: {e}")
            return None
    
    def _generate_mock_data(self, start_date, end_date):
        """生成模拟股票数据"""
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = [d for d in dates if d.weekday() < 5]  # 工作日
            
            n = len(dates)
            if n == 0:
                return None
            
            base_price = 10.0
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
            
        except Exception as e:
            print(f"✗ 模拟数据生成失败: {e}")
            return None
    
    def load_data(self, stock_code='000001'):
        """从数据库加载数据（带错误处理）"""
        try:
            query = f"""
                SELECT * FROM daily_data 
                WHERE stock_code = '{stock_code}' 
                ORDER BY date
            """
            df = pd.read_sql_query(query, self.conn)
            
            if len(df) == 0:
                print(f"⚠ 没有找到股票 {stock_code} 的数据")
                return None
            
            print(f"✓ 加载了 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return None
    
    def delete_data(self, stock_code=None, date_range=None):
        """
        删除数据（带事务回滚）
        
        参数:
            stock_code: 股票代码，None表示删除所有
            date_range: (start_date, end_date) 日期范围
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('BEGIN')
            
            if stock_code and date_range:
                cursor.execute('''
                    DELETE FROM daily_data 
                    WHERE stock_code = ? AND date BETWEEN ? AND ?
                ''', (stock_code, date_range[0], date_range[1]))
                
            elif stock_code:
                cursor.execute('''
                    DELETE FROM daily_data WHERE stock_code = ?
                ''', (stock_code,))
                
            else:
                cursor.execute('DELETE FROM daily_data')
            
            deleted_count = cursor.rowcount
            cursor.execute('COMMIT')
            
            print(f"✓ 成功删除 {deleted_count} 条数据")
            return deleted_count
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            print(f"✗ 删除失败，已回滚: {e}")
            return 0
    
    def backup_database(self, backup_path='stock_data_backup.db'):
        """备份数据库"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"✓ 数据库备份成功: {backup_path}")
            return True
        except Exception as e:
            print(f"✗ 数据库备份失败: {e}")
            return False
    
    def restore_database(self, backup_path='stock_data_backup.db'):
        """恢复数据库"""
        try:
            import shutil
            self.close()
            shutil.copy2(backup_path, self.db_path)
            self.conn = sqlite3.connect(self.db_path)
            print(f"✓ 数据库恢复成功")
            return True
        except Exception as e:
            print(f"✗ 数据库恢复失败: {e}")
            return False
    
    def feature_engineering(self, df):
        """特征工程（带错误处理）"""
        try:
            if df is None or len(df) < 20:
                print("⚠ 数据不足，无法进行特征工程")
                return None
            
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 计算技术指标
            df['return'] = df['close'].pct_change()
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['momentum'] = df['close'] - df['close'].shift(5)
            df['volatility'] = df['return'].rolling(window=5).std()
            df['volume_change'] = df['volume'].pct_change()
            
            # 目标变量
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # 删除缺失值
            df = df.dropna()
            
            if len(df) == 0:
                print("⚠ 特征工程后没有有效数据")
                return None
            
            print(f"✓ 特征工程完成，剩余 {len(df)} 条有效数据")
            return df
            
        except Exception as e:
            print(f"✗ 特征工程失败: {e}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✓ 数据库连接已关闭")


# 测试回滚机制
def test_rollback():
    """测试事务回滚"""
    print("\n" + "="*50)
    print("测试数据库回滚机制")
    print("="*50)
    
    import os
    test_db = 'test_rollback.db'
    
    # 清理测试数据库
    if os.path.exists(test_db):
        os.remove(test_db)
    
    manager = StockDataManager(test_db)
    
    # 测试1: 正常插入
    print("\n测试1: 正常插入数据")
    df = manager.fetch_and_save_data('000001')
    
    # 测试2: 备份
    print("\n测试2: 备份数据库")
    manager.backup_database('test_backup.db')
    
    # 测试3: 删除数据（会回滚如果出错）
    print("\n测试3: 删除部分数据")
    manager.delete_data('000001', ('2024-01-01', '2024-06-01'))
    
    # 测试4: 恢复
    print("\n测试4: 从备份恢复")
    manager.restore_database('test_backup.db')
    
    # 清理
    manager.close()
    os.remove(test_db)
    if os.path.exists('test_backup.db'):
        os.remove('test_backup.db')
    
    print("\n✓ 回滚机制测试完成")


if __name__ == "__main__":
    test_rollback()

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
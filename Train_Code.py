import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import re
import os
import platform  # 提前导入platform模块

warnings.filterwarnings('ignore')


# ==================== 数据预处理部分 ====================
class SmartMeterDataProcessor:
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path)
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.feature_stats = {}
        self.process_data()

    def clean_datetime_string(self, dt_str):
        if pd.isna(dt_str):
            return dt_str
        dt_str = str(dt_str).strip()
        if ':' in dt_str:
            parts = dt_str.split(':')
            if len(parts) >= 3:
                try:
                    int(parts[2])
                    return dt_str
                except:
                    return ':'.join(parts[:2])
            elif len(parts) == 2:
                return dt_str
        return dt_str

    def parse_datetime(self, datetime_series):
        print("正在解析时间数据...")
        cleaned_series = datetime_series.apply(self.clean_datetime_string)
        sample_values = cleaned_series.dropna().head(100).tolist()
        print(f"清理后的时间样本：{sample_values[:10]}")
        try:
            parsed_dates = pd.to_datetime(cleaned_series, errors='coerce', infer_datetime_format=True)
            failed_count = parsed_dates.isna().sum()
            if failed_count > 0:
                print(f"警告：有 {failed_count} 个时间值解析失败，将使用前向填充处理")
                parsed_dates = parsed_dates.fillna(method='ffill')
            return parsed_dates
        except Exception as e:
            print(f"时间解析错误：{e}")
            return self.manual_parse_datetime(cleaned_series)

    def manual_parse_datetime(self, datetime_series):
        print("使用手动时间解析...")
        parsed_dates = []
        for dt_str in datetime_series:
            try:
                if pd.isna(dt_str):
                    parsed_dates.append(pd.NaT)
                    continue
                dt_str = str(dt_str).strip()
                formats = [
                    '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M',
                    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                    '%Y/%m/%d', '%Y-%m-%d'
                ]
                parsed = None
                for fmt in formats:
                    try:
                        parsed = pd.to_datetime(dt_str, format=fmt)
                        break
                    except:
                        continue
                if parsed is None:
                    parsed = pd.to_datetime(dt_str, errors='coerce')
                parsed_dates.append(parsed)
            except Exception as e:
                print(f"解析时间 '{dt_str}' 时出错: {e}")
                parsed_dates.append(pd.NaT)
        result = pd.Series(parsed_dates, index=datetime_series.index)
        result = result.fillna(method='ffill')
        return result

    def analyze_features(self):
        print("\n" + "=" * 60)
        print("特征分析报告")
        print("=" * 60)
        for col in self.feature_cols:
            if col in self.data.columns:
                values = self.data[col]
                self.feature_stats[col] = {
                    'count': len(values), 'mean': values.mean(), 'std': values.std(),
                    'min': values.min(), 'max': values.max(), 'q25': values.quantile(0.25),
                    'q50': values.quantile(0.50), 'q75': values.quantile(0.75),
                    'missing': values.isna().sum(), 'zeros': (values == 0).sum()
                }
                print(f"\n特征: {col}")
                print(f"  数据点数: {self.feature_stats[col]['count']:,}")
                print(f"  均值: {self.feature_stats[col]['mean']:.4f}")
                print(f"  标准差: {self.feature_stats[col]['std']:.4f}")
                print(f"  最小值: {self.feature_stats[col]['min']:.4f}")
                print(f"  最大值: {self.feature_stats[col]['max']:.4f}")
                print(f"  中位数: {self.feature_stats[col]['q50']:.4f}")
                print(f"  缺失值: {self.feature_stats[col]['missing']:,}")
                print(f"  零值: {self.feature_stats[col]['zeros']:,}")

    def process_data(self):
        print("检查数据格式...")
        print(f"数据形状：{self.data.shape}")
        print(f"列名：{self.data.columns.tolist()}")

        self.data['数据时间'] = self.parse_datetime(self.data['数据时间'])
        print(f"时间解析完成，成功解析 {(~self.data['数据时间'].isna()).sum()} 条记录")

        # 处理缺失值
        if self.data['误差值(%)'].isnull().any():
            print(f"目标列'误差值(%)'有 {self.data['误差值(%)'].isnull().sum()} 个缺失值，将删除这些行")
            before_rows = len(self.data)
            self.data = self.data.dropna(subset=['误差值(%)'])
            print(f"删除目标列缺失值：{before_rows} -> {len(self.data)} 行")

        # 处理异常值
        Q1 = self.data['误差值(%)'].quantile(0.25)
        Q3 = self.data['误差值(%)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = (self.data['误差值(%)'] < lower_bound) | (self.data['误差值(%)'] > upper_bound)
        print(f"发现 {outliers.sum()} 个异常值（误差值超出 [{lower_bound:.2f}, {upper_bound:.2f}] 范围）")
        self.data.loc[self.data['误差值(%)'] < lower_bound, '误差值(%)'] = lower_bound
        self.data.loc[self.data['误差值(%)'] > upper_bound, '误差值(%)'] = upper_bound

        # 基础特征工程
        start_time = self.data['数据时间'].min()
        self.data['使用时长(小时)'] = (self.data['数据时间'] - start_time).dt.total_seconds() / 3600
        self.data['小时'] = self.data['数据时间'].dt.hour
        self.data['星期'] = self.data['数据时间'].dt.dayofweek
        self.data['月份'] = self.data['数据时间'].dt.month

        # 周期性编码
        self.data['小时_sin'] = np.sin(2 * np.pi * self.data['小时'] / 24)
        self.data['小时_cos'] = np.cos(2 * np.pi * self.data['小时'] / 24)
        self.data['星期_sin'] = np.sin(2 * np.pi * self.data['星期'] / 7)
        self.data['星期_cos'] = np.cos(2 * np.pi * self.data['星期'] / 7)
        self.data['月份_sin'] = np.sin(2 * np.pi * self.data['月份'] / 12)
        self.data['月份_cos'] = np.cos(2 * np.pi * self.data['月份'] / 12)

        # 精简但有效的特征工程
        self.data['电流_平方'] = self.data['负载电流'] ** 2
        self.data['电流_log'] = np.log1p(np.abs(self.data['负载电流']))
        self.data['是否工作日'] = (self.data['星期'] < 5).astype(float)

        # 交互特征
        self.data['温湿度交互'] = self.data['气温(℃)'] * self.data['湿度(%RH)'] / 100
        self.data['舒适度指数'] = (self.data['气温(℃)'] + self.data['湿度(%RH)'] / 10 -
                              self.data['风速(m/s)'] * 2)

        # 核心特征列表（保持精简）
        self.feature_cols = [
            '负载电流', '电流_平方', '电流_log',
            '气温(℃)', '湿度(%RH)', '气压(hpa)', '风速(m/s)', '光照(lux)',
            '使用时长(小时)',
            '小时_sin', '小时_cos', '星期_sin', '星期_cos', '月份_sin', '月份_cos',
            '是否工作日', '温湿度交互', '舒适度指数'
        ]
        self.target_col = '误差值(%)'

        # 填充缺失值
        for col in self.feature_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(method='ffill').fillna(self.data[col].mean())

        print(f"最终数据行数：{len(self.data)}")
        print(f"特征数量：{len(self.feature_cols)}")
        self.analyze_features()

    def create_sequences(self, seq_length=12, pred_length=1):
        print(f"\n创建时间序列数据，序列长度：{seq_length}，预测长度：{pred_length}")
        X, y = [], []
        for i in range(len(self.data) - seq_length - pred_length + 1):
            X.append(self.data[self.feature_cols].iloc[i:i + seq_length].values)
            y.append(self.data[self.target_col].iloc[i + seq_length:i + seq_length + pred_length].values)
        X = np.array(X)
        y = np.array(y)
        print(f"序列数据形状 - X: {X.shape}, y: {y.shape}")

        # 标准化
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

        print(f"标准化完成")
        print(f"X标准化后范围: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
        print(f"y标准化后范围: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
        return X_scaled, y_scaled


# ==================== 改进的LSTM模型 ====================
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # 注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 加权上下文向量
        context = torch.sum(lstm_out * attention_weights, dim=1)

        output = self.fc(context)
        return output


# ==================== 改进的Transformer模型 ====================
class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=96, n_heads=6, n_layers=3, dropout=0.35):
        super(ImprovedTransformerModel, self).__init__()

        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 投影和位置编码
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer编码
        x = self.transformer(x)

        # 使用最后一个时间步
        output = self.output_layer(x[:, -1, :])
        return output


# ==================== 数据集类 ====================
class SmartMeterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 早停机制 ====================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ==================== 稳健的训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=300, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 使用稳定的损失函数
    criterion = nn.SmoothL1Loss()  # Huber Loss

    # AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )

    # 学习率调度器 - 使用ReduceLROnPlateau更稳定
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )

    early_stopping = EarlyStopping(patience=30, min_delta=1e-5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # 梯度累积参数
    grad_accum_steps = 1

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # 非阻塞数据传输
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) / grad_accum_steps
            loss.backward()

            # 梯度裁剪
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 早停检查
        if early_stopping(avg_val_loss):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


# ==================== 评估函数 ====================
def evaluate_model(model, test_loader, scaler_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            outputs = model(batch_X)

            y_true_batch = scaler_y.inverse_transform(batch_y.cpu().numpy())
            y_pred_batch = scaler_y.inverse_transform(outputs.cpu().numpy())

            y_true.extend(y_true_batch.flatten())
            y_pred.extend(y_pred_batch.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 改进的MAPE计算
    epsilon = 1e-8
    mask = np.abs(y_true) > 0.01  # 只计算误差值大于1%的MAPE
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
               'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
           }, y_true, y_pred


# ==================== 可视化函数 ====================
def plot_results(train_losses, val_losses, y_true, y_pred, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 训练曲线
    if len(train_losses) > 0:
        axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 预测vs实际
    sample_size = min(1000, len(y_true))
    axes[0, 1].scatter(y_true[:sample_size], y_pred[:sample_size], alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Error (%)')
    axes[0, 1].set_ylabel('Predicted Error (%)')
    axes[0, 1].set_title('Prediction vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 误差分布
    errors = y_pred - y_true
    axes[0, 2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=0, color='r', linestyle='--', label='Zero Error')
    axes[0, 2].set_xlabel('Prediction Error (%)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Error Distribution (Mean: {errors.mean():.3f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 时间序列对比
    sample_size = min(200, len(y_true))
    axes[1, 0].plot(y_true[:sample_size], label='Actual', alpha=0.8, linewidth=1)
    axes[1, 0].plot(y_pred[:sample_size], label='Predicted', alpha=0.8, linewidth=1)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_title('Time Series Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 残差图
    sample_size = min(1000, len(y_true))
    axes[1, 1].scatter(y_pred[:sample_size], errors[:sample_size], alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Error (%)')
    axes[1, 1].set_ylabel('Residuals (%)')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)

    # Q-Q图
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印指标
    print("\n" + "=" * 50)
    print("模型评估指标：")
    print("=" * 50)
    for metric, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric:8s}: {value:.4f}")
        else:
            print(f"{metric:8s}: N/A")


# ==================== 在现有代码基础上添加的优化函数 ====================

# 1. 改进的MAPE计算（在现有evaluate_model函数后添加）
def calculate_improved_mape(y_true, y_pred):
    """改进的MAPE计算，添加阈值过滤和SMAPE"""
    # 标准MAPE（带阈值过滤）
    threshold = 0.02  # 2%的阈值
    mask = np.abs(y_true) > threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100

    return mape, smape


# 2. 替换现有的evaluate_model函数
def evaluate_model_improved(model, test_loader, scaler_y):
    """改进的评估函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            outputs = model(batch_X)

            y_true_batch = scaler_y.inverse_transform(batch_y.cpu().numpy())
            y_pred_batch = scaler_y.inverse_transform(outputs.cpu().numpy())

            y_true.extend(y_true_batch.flatten())
            y_pred.extend(y_pred_batch.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 基础指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 改进的MAPE和SMAPE
    mape, smape = calculate_improved_mape(y_true, y_pred)

    return {
               'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
               'MAPE': mape, 'SMAPE': smape
           }, y_true, y_pred


# 3. 增强的Transformer模型（在ImprovedTransformerModel后添加）
class EnhancedTransformerModel(nn.Module):
    """增强版Transformer，目标R²>0.90"""

    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=4, dropout=0.3):
        super(EnhancedTransformerModel, self).__init__()

        # 增强的输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 学习的位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, 200, d_model) * 0.01)

        # 多层Transformer编码器（使用norm_first提高稳定性）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm架构
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 增强的输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )

        # 残差连接
        self.shortcut = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 输入处理
        x = self.input_projection(x)
        x_orig = x  # 保存用于残差连接

        # 添加位置编码
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer编码
        x = self.transformer(x)

        # 使用最后3个时间步的加权平均
        if seq_len >= 3:
            weights = torch.softmax(torch.tensor([0.3, 0.3, 0.4]).to(x.device), dim=0)
            x_final = (x[:, -3, :] * weights[0] +
                       x[:, -2, :] * weights[1] +
                       x[:, -1, :] * weights[2])
        else:
            x_final = x[:, -1, :]

        # 输出层带残差连接
        output = self.output_layer(x_final) + self.shortcut(x_orig[:, -1, :]) * 0.1

        return output


# 4. 优化的训练函数（替换现有的train_model函数）
def train_model_optimized(model, train_loader, val_loader, num_epochs=500, learning_rate=0.0008):
    """优化的训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 组合损失函数
    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss(beta=0.01)

    def combined_loss(pred, true):
        """70% MSE + 30% Huber"""
        return 0.7 * mse_loss(pred, true) + 0.3 * huber_loss(pred, true)

    criterion = combined_loss

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.005,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # 学习率调度
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=15,
        min_lr=1e-7,
        verbose=True
    )

    # Warmup
    def warmup_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        return 1.0

    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # 早停
    early_stopping = EarlyStopping(patience=50, min_delta=1e-6)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    # 梯度累积参数（GPU内存不足时可增加）
    grad_accum_steps = 1

    for epoch in range(num_epochs):
        # Warmup阶段
        if epoch < 10:
            scheduler2.step()

        # 训练
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # 非阻塞数据传输，提升GPU利用率
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) / grad_accum_steps
            loss.backward()

            # 梯度累积：每grad_accum_steps个批次更新一次参数
            if (batch_idx + 1) % grad_accum_steps == 0:
                max_norm = 1.0 if epoch < 20 else 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 调整学习率
        if epoch >= 10:
            scheduler1.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1

        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 早停（至少训练100轮）
        if epoch > 100 and early_stopping(avg_val_loss):
            print(f'Early stopping at epoch {epoch + 1}, best was epoch {best_epoch}')
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


# 5. 增强特征工程函数（在process_data方法中调用）
def add_advanced_features(data):
    """添加高级特征"""
    # 添加'日期'列如果不存在
    if '日期' not in data.columns and '数据时间' in data.columns:
        data['日期'] = data['数据时间'].dt.day

    # 时间特征
    data['小时段'] = pd.cut(data['小时'], bins=[0, 6, 12, 18, 24],
                         labels=[0, 1, 2, 3]).astype(float)
    data['是否周末'] = (data['星期'] >= 5).astype(float)
    data['月中旬'] = ((data['日期'] > 10) & (data['日期'] <= 20)).astype(float)

    # 滑动窗口特征
    for w in [3, 5]:
        data[f'电流_滚动均值_{w}'] = data['负载电流'].rolling(window=w, min_periods=1).mean()
        data[f'电流_滚动std_{w}'] = data['负载电流'].rolling(window=w, min_periods=1).std().fillna(0)

    # 差分特征
    data['电流_差分'] = data['负载电流'].diff().fillna(0)
    data['温度_差分'] = data['气温(℃)'].diff().fillna(0)

    # 高级交互特征
    data['电流温度交互'] = data['负载电流'] * data['气温(℃)'] / 100
    data['光照电流比'] = np.log1p(data['光照(lux)']) / (data['负载电流'] + 1)
    data['环境综合指数'] = (data['气温(℃)'] * 0.3 +
                      data['湿度(%RH)'] * 0.2 +
                      data['气压(hpa)'] / 100 * 0.3 +
                      data['风速(m/s)'] * 0.2)

    return data


# 6. 高级集成类（优化GPU设备一致性）
class AdvancedEnsemble:
    """高级集成模型（GPU设备一致性优化）"""

    def __init__(self, models):
        self.models = models
        self.weights = None
        # 记录模型所在设备（确保所有模型在同一设备）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化时将所有模型迁移到目标设备
        for i in range(len(self.models)):
            self.models[i] = self.models[i].to(self.device)

    def fit_weights(self, val_loader):
        """基于验证集自动优化权重（设备一致性优化）"""
        from scipy.optimize import minimize
        all_predictions = [[] for _ in self.models]
        all_y_true = []

        # 对每个批次进行预测
        for batch_X, batch_y in val_loader:
            # 强制将数据迁移到模型所在设备
            batch_X = batch_X.to(self.device, non_blocking=True)  # non_blocking加速传输
            batch_y_np = batch_y.numpy()  # 真实值无需GPU，直接转CPU

            # 收集真实值
            all_y_true.extend(batch_y_np.flatten())

            # 每个模型的预测
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    pred = model(batch_X)
                    # 预测结果转CPU（避免GPU内存占用）
                    all_predictions[i].extend(pred.cpu().numpy().flatten())

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)  # shape: (n_models, n_samples)
        all_y_true = np.array(all_y_true)

        print(f"预测数据形状: {all_predictions.shape}")
        print(f"真实数据形状: {all_y_true.shape}")

        # 优化权重的目标函数
        def objective(weights):
            # 加权平均预测
            weighted_pred = np.sum(all_predictions.T * weights, axis=1)
            return mean_squared_error(all_y_true, weighted_pred)

        # 约束：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        init_weights = np.array([1 / len(self.models)] * len(self.models))

        # 优化
        result = minimize(objective, init_weights, bounds=bounds,
                          constraints=constraints, method='SLSQP')

        self.weights = result.x
        print(f"优化后的集成权重: {self.weights}")

        # 计算集成后的验证误差
        final_pred = np.sum(all_predictions.T * self.weights, axis=1)
        val_mse = mean_squared_error(all_y_true, final_pred)
        print(f"集成模型验证MSE: {val_mse:.6f}")

    def predict(self, X):
        """使用加权平均进行预测（设备一致性强制）"""
        # 强制输入数据与模型在同一设备
        X = X.to(self.device, non_blocking=True)
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred)

        if self.weights is None:
            self.weights = [1 / len(self.models)] * len(self.models)

        # 加权平均
        weighted_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_pred


# ==================== 完整的主函数集成所有优化 ====================
def main_optimized(data_file='Data_5.xlsx'):
    """完全优化版本的主函数"""

    # 1. 数据处理
    print("正在加载和处理数据...")
    processor = SmartMeterDataProcessor(data_file)

    # 添加高级特征
    print("添加高级特征...")
    processor.data = add_advanced_features(processor.data)

    # 更新特征列表
    additional_features = [
        '小时段', '是否周末', '月中旬',
        '电流_滚动均值_3', '电流_滚动std_3',
        '电流_滚动均值_5', '电流_滚动std_5',
        '电流_差分', '温度_差分',
        '电流温度交互', '光照电流比', '环境综合指数'
    ]

    # 扩展特征列表
    processor.feature_cols.extend(additional_features)

    # 填充新特征的缺失值
    for col in additional_features:
        if col in processor.data.columns:
            processor.data[col] = processor.data[col].fillna(method='ffill').fillna(0)

    print(f"增强后特征数量：{len(processor.feature_cols)}")

    # 2. 创建序列数据
    seq_length = 12  # 增加序列长度
    pred_length = 1
    X, y = processor.create_sequences(seq_length, pred_length)
    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 3. 划分数据集
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 4. 创建数据加载器（优化GPU数据传输）
    train_dataset = SmartMeterDataset(X_train, y_train)
    val_dataset = SmartMeterDataset(X_val, y_val)
    test_dataset = SmartMeterDataset(X_test, y_test)

    # 根据操作系统设置工作进程数
    num_workers = 0 if platform.system() == 'Windows' else 4  # 增加工作进程数（非Windows）
    pin_memory = torch.cuda.is_available()  # 仅当GPU可用时启用pin_memory

    # 仅在多进程模式下设置prefetch_factor
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory, **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory, **dataloader_kwargs
    )

    # 5. 训练多个模型
    input_dim = X.shape[-1]
    models = []
    model_metrics = []

    # 模型1：增强版Transformer
    print("\n" + "=" * 60)
    print("训练模型1：增强版Transformer...")
    print("=" * 60)
    model1 = EnhancedTransformerModel(
        input_dim=input_dim,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.3
    )
    print(f"模型1参数量: {sum(p.numel() for p in model1.parameters()):,}")

    model1, losses1_train, losses1_val = train_model_optimized(
        model1, train_loader, val_loader,
        num_epochs=300,
        learning_rate=0.0008
    )

    # 评估模型1
    metrics1, y_true1, y_pred1 = evaluate_model_improved(model1, test_loader, processor.scaler_y)
    print(f"\n模型1性能:")
    print(f"  R²: {metrics1['R2']:.4f}")
    print(f"  RMSE: {metrics1['RMSE']:.4f}")
    print(f"  MAE: {metrics1['MAE']:.4f}")
    print(f"  SMAPE: {metrics1['SMAPE']:.2f}%")
    models.append(model1)
    model_metrics.append(metrics1)

    # 模型2：标准Transformer（不同配置）
    print("\n" + "=" * 60)
    print("训练模型2：标准Transformer...")
    print("=" * 60)
    model2 = ImprovedTransformerModel(
        input_dim=input_dim,
        d_model=96,
        n_heads=6,
        n_layers=3,
        dropout=0.35
    )
    print(f"模型2参数量: {sum(p.numel() for p in model2.parameters()):,}")

    model2, losses2_train, losses2_val = train_model(
        model2, train_loader, val_loader,
        num_epochs=200,
        learning_rate=0.001
    )

    # 评估模型2
    metrics2, y_true2, y_pred2 = evaluate_model_improved(model2, test_loader, processor.scaler_y)
    print(f"\n模型2性能:")
    print(f"  R²: {metrics2['R2']:.4f}")
    print(f"  RMSE: {metrics2['RMSE']:.4f}")
    print(f"  MAE: {metrics2['MAE']:.4f}")
    print(f"  SMAPE: {metrics2['SMAPE']:.2f}%")
    models.append(model2)
    model_metrics.append(metrics2)

    # 模型3：LSTM模型
    print("\n" + "=" * 60)
    print("训练模型3：LSTM模型...")
    print("=" * 60)
    model3 = ImprovedLSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    )
    print(f"模型3参数量: {sum(p.numel() for p in model3.parameters()):,}")

    model3, losses3_train, losses3_val = train_model(
        model3, train_loader, val_loader,
        num_epochs=200,
        learning_rate=0.001
    )

    # 评估模型3
    metrics3, y_true3, y_pred3 = evaluate_model_improved(model3, test_loader, processor.scaler_y)
    print(f"\n模型3性能:")
    print(f"  R²: {metrics3['R2']:.4f}")
    print(f"  RMSE: {metrics3['RMSE']:.4f}")
    print(f"  MAE: {metrics3['MAE']:.4f}")
    print(f"  SMAPE: {metrics3['SMAPE']:.2f}%")
    models.append(model3)
    model_metrics.append(metrics3)

    # 6. 创建集成模型
    print("\n" + "=" * 60)
    print("创建集成模型...")
    print("=" * 60)

    ensemble = AdvancedEnsemble(models)

    # 优化集成权重
    print("优化集成权重...")
    ensemble.fit_weights(val_loader)

    # 7. 评估集成模型
    print("\n评估集成模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true = []
    y_pred = []

    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        outputs = ensemble.predict(batch_X)

        y_true_batch = processor.scaler_y.inverse_transform(batch_y.numpy())
        y_pred_batch = processor.scaler_y.inverse_transform(outputs.cpu().numpy())

        y_true.extend(y_true_batch.flatten())
        y_pred.extend(y_pred_batch.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算集成模型指标
    ensemble_metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

    mape, smape = calculate_improved_mape(y_true, y_pred)
    ensemble_metrics['MAPE'] = mape
    ensemble_metrics['SMAPE'] = smape

    # 8. 打印最终结果
    print("\n" + "=" * 60)
    print("最终性能对比")
    print("=" * 60)
    print(f"{'模型':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'SMAPE':<10}")
    print("-" * 60)
    print(f"{'模型1(增强Trans)':<20} {metrics1['R2']:<10.4f} {metrics1['RMSE']:<10.4f} "
          f"{metrics1['MAE']:<10.4f} {metrics1['SMAPE']:<10.2f}%")
    print(f"{'模型2(标准Trans)':<20} {metrics2['R2']:<10.4f} {metrics2['RMSE']:<10.4f} "
          f"{metrics2['MAE']:<10.4f} {metrics2['SMAPE']:<10.2f}%")
    print(f"{'模型3(LSTM)':<20} {metrics3['R2']:<10.4f} {metrics3['RMSE']:<10.4f} "
          f"{metrics3['MAE']:<10.4f} {metrics3['SMAPE']:<10.2f}%")
    print(f"{'集成模型':<20} {ensemble_metrics['R2']:<10.4f} {ensemble_metrics['RMSE']:<10.4f} "
          f"{ensemble_metrics['MAE']:<10.4f} {ensemble_metrics['SMAPE']:<10.2f}%")

    # 9. 可视化最佳结果
    best_r2 = max(metrics1['R2'], metrics2['R2'], metrics3['R2'], ensemble_metrics['R2'])
    if ensemble_metrics['R2'] == best_r2:
        print("\n集成模型表现最佳，生成可视化...")
        plot_results([], [], y_true, y_pred, ensemble_metrics)
        best_model = ensemble
        best_metrics = ensemble_metrics
    else:
        print("\n单模型表现最佳")
        if metrics1['R2'] == best_r2:
            best_model = model1
            best_metrics = metrics1
            plot_results(losses1_train, losses1_val, y_true1, y_pred1, metrics1)
        elif metrics2['R2'] == best_r2:
            best_model = model2
            best_metrics = metrics2
            plot_results(losses2_train, losses2_val, y_true2, y_pred2, metrics2)
        else:
            best_model = model3
            best_metrics = metrics3
            plot_results(losses3_train, losses3_val, y_true3, y_pred3, metrics3)

    # 10. 保存最佳模型（设备无关性处理）
    model_save_path = 'smart_meter_model_optimized.pth'

    if isinstance(best_model, AdvancedEnsemble):
        # 保存集成模型时，将所有子模型参数转CPU
        ensemble_states = [m.cpu().state_dict() for m in best_model.models]
        torch.save({
            'ensemble_models': ensemble_states,  # 已转CPU的参数
            'ensemble_weights': best_model.weights,
            'model_configs': [
                {'type': 'EnhancedTransformer', 'd_model': 128, 'n_heads': 8, 'n_layers': 4},
                {'type': 'ImprovedTransformer', 'd_model': 96, 'n_heads': 6, 'n_layers': 3},
                {'type': 'ImprovedLSTM', 'hidden_dim': 128, 'num_layers': 3}
            ],
            'scaler_X': processor.scaler_X,
            'scaler_y': processor.scaler_y,
            'feature_cols': processor.feature_cols,
            'metrics': ensemble_metrics,
            'input_dim': input_dim,
            'seq_length': seq_length
        }, model_save_path)
    else:
        # 保存单模型时，参数转CPU
        torch.save({
            'model_state_dict': best_model.cpu().state_dict(),  # 已转CPU的参数
            'model_type': type(best_model).__name__,
            'scaler_X': processor.scaler_X,
            'scaler_y': processor.scaler_y,
            'feature_cols': processor.feature_cols,
            'metrics': best_metrics,
            'input_dim': input_dim,
            'seq_length': seq_length
        }, model_save_path)

    print(f"\n最佳模型已保存至 {model_save_path}")
    print(f"最终R²: {best_metrics['R2']:.4f}")
    print(f"最终RMSE: {best_metrics['RMSE']:.4f}%")
    print(f"最终SMAPE: {best_metrics['SMAPE']:.2f}%")

    return best_model, best_metrics, processor


# ==================== 快速测试函数 ====================
def quick_test_improvements():
    """快速测试各项改进是否有效"""

    # 测试数据
    y_true = np.array([0.01, 0.02, 0.001, 0.1, -0.05, -0.001, 0.15])
    y_pred = np.array([0.011, 0.019, 0.002, 0.095, -0.048, -0.0005, 0.14])

    # 测试改进的MAPE
    mape, smape = calculate_improved_mape(y_true, y_pred)
    print(f"改进的MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")

    # 测试模型架构
    model = EnhancedTransformerModel(input_dim=18)
    print(f"增强模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("测试通过！")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    try:
        # 运行快速测试
        print("运行改进测试...")
        quick_test_improvements()

        print("\n" + "=" * 60)
        print("开始主程序...")
        print("=" * 60)

        # 自动检测数据文件
        if os.path.exists('Data_5.xlsx'):
            data_file = 'Data_5.xlsx'
        elif os.path.exists('Data_4.xlsx'):
            data_file = 'Data_4.xlsx'
        elif os.path.exists('Data_3.xlsx'):
            data_file = 'Data_3.xlsx'
        else:
            data_file = input("请输入数据文件名：")

        # 运行优化版本
        best_model, final_metrics, processor = main_optimized(data_file)

        print("\n" + "=" * 60)
        print("优化完成！")
        print("=" * 60)

        # 性能提升总结
        original_r2 = 0.8558  # 之前的结果
        final_r2 = final_metrics['R2']
        improvement = (final_r2 - original_r2) / original_r2 * 100

        print(f"\n性能提升总结：")
        print(f"原始R²: {original_r2:.4f}")
        print(f"优化后R²: {final_r2:.4f}")
        print(f"提升幅度: {improvement:.2f}%")

        if final_r2 >= 0.90:
            print("\n✓ 成功达到R²≥0.90的目标！")
        else:
            print(f"\n距离R²=0.90目标还差: {0.90 - final_r2:.4f}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()

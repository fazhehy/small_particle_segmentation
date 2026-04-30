import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

k = 0.12718545
csv_path = './results/3.23-1/03.27-12:05:45/03.27-12:05:45.csv'

df = pd.read_csv(csv_path)

if 'area' not in df.columns:
    raise ValueError(f"CSV中未找到'area'列，可用列：{df.columns.tolist()}")
if 'frame_id' not in df.columns:
    raise ValueError(f"CSV中未找到'frame_id'列，可用列：{df.columns.tolist()}")

# 将 area 等效为圆面积：A = πr^2 -> r = sqrt(A/π)
df['equivalent_radius_px'] = np.sqrt(df['area'] / np.pi)

# 若 k 为像素到实际长度的比例（例如 mm/px），则可得到实际半径
df['equivalent_radius_real'] = df['equivalent_radius_px'] * k

# 自动阈值：假设 log(半径) 更接近正态分布，使用 3σ 迭代裁剪
radius_real = df['equivalent_radius_real'].copy()
log_r = np.log1p(radius_real)
mask = np.ones(len(log_r), dtype=bool)

for _ in range(3):
    mu = log_r[mask].mean()
    sigma = log_r[mask].std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        break
    new_mask = (log_r >= (mu - 3 * sigma)) & (log_r <= (mu + 3 * sigma))
    if new_mask.sum() == mask.sum():
        break
    mask = new_mask

upper_cap = np.expm1(mu + 3 * sigma) if np.isfinite(sigma) and sigma > 0 else radius_real.max()
df_outlier = df[df['equivalent_radius_real'] > upper_cap][['frame_id', 'area', 'equivalent_radius_real']].copy()
df = df[df['equivalent_radius_real'] <= upper_cap].copy()

print(f'自动阈值(对数半径3σ) upper_cap = {upper_cap:.4f} mm, 过滤 {len(df_outlier)} 条')
if not df_outlier.empty:
    print('被过滤的大半径样本（按半径降序）：')
    print(df_outlier.sort_values('equivalent_radius_real', ascending=False).reset_index(drop=True))

# 以 0.25 mm 作为区间长度划分半径区间，并重新统计
bin_width = 0.25  # mm
r_min = df['equivalent_radius_real'].min()
r_max = df['equivalent_radius_real'].max()

# 将起止点对齐到区间长度的整数倍
bin_start = np.floor(r_min / bin_width) * bin_width
bin_end = np.ceil(r_max / bin_width) * bin_width

# 处理极端情况：所有半径都相同
if bin_start == bin_end:
    bin_end = bin_start + bin_width

# 构建分箱
bins = np.arange(bin_start, bin_end + bin_width, bin_width)

df['radius_bin'] = pd.cut(df['equivalent_radius_real'], bins=bins, right=False, include_lowest=True)

radius_stats = (
    df.groupby('radius_bin', observed=True, as_index=False)
      .agg(
          area_sum=('area', 'sum'),
          count=('area', 'size')
      )
)
radius_stats['area_ratio'] = radius_stats['area_sum'] / radius_stats['area_sum'].sum()

# 统一横坐标标签为“>区间右值”
def format_bin_label(interval):
    if pd.isna(interval):
        return 'NA'
    return f'>{interval.right:.2f}'

radius_stats['radius_interval_mm'] = radius_stats['radius_bin'].apply(format_bin_label)

print(radius_stats[['radius_interval_mm', 'count', 'area_sum', 'area_ratio']])

# 绘制柱状图（x: 半径区间，y: 面积占比）
plt.figure(figsize=(14, 6))
sns.barplot(data=radius_stats, x='radius_interval_mm', y='area_ratio')
plt.xlabel('等效半径区间（mm）')
plt.ylabel('面积占比')
plt.title('不同半径区间（0.25 mm）的面积占比柱状图')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



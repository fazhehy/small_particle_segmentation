import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# csv_path = './results/3-23-1/03-23_18-12-28/results_03-23_18-12-28.csv'
csv_path = './results/temp/results_03-23_19-42-44.csv'
save_path = './results/images/'+os.path.basename(os.path.dirname(csv_path))
if not os.path.exists(save_path):
    os.makedirs(save_path)

truth_value = [3.305, 6.327, 12.067, 13.583, 16.150, 6.759, 0.986]

truth_value = [0, 0, 0, 0, 0, 0, 0]

df = pd.read_csv(csv_path)

# split
df = df.drop(['Unnamed: 0', 'n'], axis=1)
df_area = df[['total_area', '<0.15_area', '>0.15_area', '>0.25_area', '>0.5_area', '>1_area', '>3_area', '>5_area']]
df_num = df.drop(df_area.columns, axis = 1)

# statistic area
df_area = df_area.sum(axis=0).reset_index()

df_area.columns = ['raw_label', 'area']
total_area = df_area['area'][0]
df_area.drop(index=[0], inplace=True)
df_area[['diameters', 'unit']] = df_area['raw_label'].str.split('_', expand=True)
df_area.drop(['raw_label','unit'], axis=1, inplace=True)
df_area.reset_index(drop=True, inplace=True)
df_area = df_area[['diameters', 'area']]
df_area['diameters'] = df_area['diameters'].astype('str') + 'mm'

precent = (df_area['area']/total_area*100).round(2)
df_area['precent'] = precent
diff = 100-precent.sum()
df_area.loc[0, 'precent'] -= diff
df_area.drop(['area'], axis=1, inplace=True)
# print(df_area)

# turth value
truth_df = pd.DataFrame({'diameters':df_area['diameters'], 'value': truth_value})
precent = (truth_df['value']/truth_df['value'].sum()*100).round(2)
truth_df['precent'] = precent
diff = 100-precent.sum()
truth_df.loc[0, 'precent'] -= diff
truth_df.drop(['value'], axis=1, inplace=True)
# print(truth_df)

# mean
temp = pd.DataFrame()
temp['truth'] = truth_df['precent']
temp['area'] = df_area['precent']
temp['reference'] = [0.075, 0.2, 0.375, 0.75, 2, 4, 5]
mean_truth = (temp['truth']*temp['reference']/100).sum().round(2)
mean_area = (temp['area']*temp['reference']/100).sum().round(2)
# print(temp)

# statistic num
df_num_avg = df_num.loc[:, ['frame_num', 'frame_mean_diameter']]
df_num.drop(df_num_avg.columns, axis=1, inplace=True)

df_num_avg['total'] = df_num_avg['frame_num']*df_num_avg['frame_mean_diameter']
total_num = df_num_avg.drop(['frame_mean_diameter'], axis=1)['frame_num'].astype(int).sum()
total_diameters = df_num_avg.drop(['frame_mean_diameter'], axis=1)['total'].astype(float).sum()
mean_num = (total_diameters/total_num).round(2)

df_num = df_num.sum()
df_num = df_num.reset_index()
df_num.columns = ['raw_label', 'num']
df_num[['diameters', 'unit']] = df_num['raw_label'].str.split('_', expand=True)
df_num.drop(['raw_label','unit'], axis=1, inplace=True)
df_num.reset_index(drop=True, inplace=True)
df_num = df_num[['diameters', 'num']]
df_num['diameters'] = df_num['diameters'].astype('str') + 'mm'
precent = (df_num['num']/total_num*100).round(2)
df_num['precent'] = precent
diff = 100-precent.sum()
df_num.loc[0, 'precent'] -= diff
df_num.drop(['num'], axis=1, inplace=True)
# print(df_num)

# save the area result
x = np.arange(len(df_area['diameters'].iloc[2:]))   # 横坐标位置
width = 0.35                           # 柱子宽度

fig, ax = plt.subplots(figsize=(8,5))
# 绘制两组数据
bars2 = ax.bar(x - width/2, df_area['precent'].iloc[2:], width, color='skyblue')
# bars2 = ax.bar(x - width/2, df_area['precent'].iloc[2:], width, label='计算值', color='skyblue')
# bars1 = ax.bar(x + width/2, truth_df['precent'].iloc[2:], width, label='真实值', color='orange')
# 添加数值标签
# for bar in bars1:
#     height = bar.get_height()
#     bar.set_edgecolor('black')
#     bar.set_linewidth(0.5)
#     ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}%',
#             ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    bar.set_edgecolor('black')
    bar.set_linewidth(0.5)
    ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}%',
            ha='center', va='bottom', fontsize=9)

# 设置标签
ax.set_xticks(x)
ax.set_xticklabels(df_area['diameters'].iloc[2:], rotation=45)
ax.set_xlabel('直径区间')
ax.set_ylabel('数值')
ax.set_title(f'面积对比图 总数:{total_num}, 平均粒径:{mean_area} mm')
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f'{save_path}/面积占比', dpi=300)

# save the num result
# x 轴位置
x = np.arange(len(df_num['diameters'].iloc[2:]))

fig, ax = plt.subplots(figsize=(8,5))

# 绘制柱状图
bars = ax.bar(x, df_num['precent'].iloc[2:], color='skyblue', width=0.6)

# 在柱子上方显示数值
for bar in bars:
    height = bar.get_height()
    bar.set_edgecolor('black')
    bar.set_linewidth(0.5)
    ax.text(
        bar.get_x() + bar.get_width()/2,  # x 位置
        height,                           # y 位置
        f'{height}',                      # 显示的文字
        ha='center', va='bottom', fontsize=10
    )

# 设置坐标轴 & 标题
ax.set_xticks(x)
ax.set_xticklabels(df_num['diameters'].iloc[2:], rotation=45)
ax.set_ylabel('数量')
ax.set_title(f'数量对比图 总数:{total_num}, 平均粒径:{mean_num}')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
fig.savefig(f'{save_path}/数量占比.png', dpi=300)

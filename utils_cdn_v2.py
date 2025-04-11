# # 文件名: utils_cdn_v2.py
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# import seaborn as sns
# import math
# import torch
# plt.style.use('seaborn-v0_8-whitegrid')
#
# # --------------------------------------------
# #            指标计算函数
# # --------------------------------------------
# def calculate_regression_metrics(y_true, y_pred):
#     """计算回归指标 (R2, RMSE, MAE, MAPE)"""
#     metrics = {}
#     # Ensure inputs are numpy arrays
#     y_true = np.asarray(y_true).flatten()
#     y_pred = np.asarray(y_pred).flatten()
#
#     # Remove NaNs or Infs if present
#     valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
#     if not np.all(valid_idx):
#         print(f"Warning: Removing {np.sum(~valid_idx)} non-finite values before metric calculation.")
#         y_true = y_true[valid_idx]
#         y_pred = y_pred[valid_idx]
#
#     if len(y_true) == 0:
#         print("Warning: No valid samples for metric calculation.")
#         return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
#
#     metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
#     metrics['MAE'] = mean_absolute_error(y_true, y_pred)
#     # MAPE Calculation with check for zeros in true values
#     mask = y_true != 0
#     if np.any(mask):
#         metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
#     else:
#         metrics['MAPE'] = np.nan
#     metrics['R2'] = r2_score(y_true, y_pred)
#     return metrics
#
# def calculate_ceg_zones_mmol(ref_values, pred_values):
#     """计算克拉克误差网格区域百分比 (mmol/L)"""
#     zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
#     total_valid_count = 0
#
#     # mmol/L 阈值
#     zone_a_low = 3.9  # ~70 mg/dL
#     zone_b_up_low = 10.0 # ~180 mg/dL
#     zone_a_abs_thresh = 0.83 # ~15 mg/dL
#
#     for ref, pred in zip(ref_values, pred_values):
#         if not (np.isfinite(ref) and np.isfinite(pred)) or ref <= 0 or pred <= 0:
#             continue
#         total_valid_count += 1
#         in_zone = '?'
#
#         # Zone E: Gross errors
#         if (pred < zone_a_low and ref > zone_b_up_low) or \
#            (pred > zone_b_up_low and ref < zone_a_low):
#             in_zone = 'E'
#         # Zone D: Failure to detect hypo/hyperglycemia
#         elif (pred >= zone_a_low and ref < zone_a_low) or \
#              (pred < zone_a_low and ref >= zone_a_low and abs(pred-ref)>zone_a_abs_thresh): # Added check for large error in low range
#              in_zone = 'D'
#         # Zone C: Poor prediction in hyperglycemia
#         elif pred > zone_b_up_low and ref > zone_b_up_low:
#             if abs(pred - ref) > ref * 0.20: # Large relative error in high range
#                  in_zone = 'C'
#             else:
#                  in_zone = 'B' # Acceptable high range error (moves to B)
#         # Zone B: Errors outside 20% or abs threshold, but not dangerous
#         else: # Neither E, D, nor C - must be A or B
#             err = abs(pred - ref)
#             is_a = False
#             if ref < zone_a_low: # Low reference range
#                 if err <= zone_a_abs_thresh: is_a = True
#             elif ref >= zone_a_low: # Normal/High reference range
#                 if err <= ref * 0.20: is_a = True
#
#             in_zone = 'A' if is_a else 'B'
#
#         if in_zone in zone_counts:
#             zone_counts[in_zone] += 1
#         else: # Should not happen with logic above
#              print(f"Warning: Point ({ref:.1f}, {pred:.1f}) didn't fall into any zone?")
#
#     zone_percentages = {k: (v / total_valid_count * 100) if total_valid_count > 0 else 0 for k, v in zone_counts.items()}
#     print(f"CEG Zone Distribution (% - based on {total_valid_count} valid points): " +
#           " | ".join([f"{k}: {v:.1f}%" for k, v in zone_percentages.items()]))
#     return zone_percentages
#
#
# # --------------------------------------------
# #            绘图函数
# # --------------------------------------------
#
# def plot_clarke_error_grid(ref_values, pred_values, title_str, save_path=None, unit='mg/dL'):
#     """
#     生成符合标准定义的克拉克误差网格图 (CEG) 并计算各区域百分比。
#
#     Args:
#         ref_values (array-like): 参考血糖值数组。
#         pred_values (array-like): 预测血糖值数组。
#         title_str (str): 图表标题。
#         save_path (str, optional): 保存图表的路径。默认为 None (不保存)。
#         unit (str, optional): 血糖值的单位 ('mg/dL' 或 'mmol/L')。
#                                 如果是 'mmol/L', 数值会被转换为 'mg/dL' 进行绘图。
#                                 默认为 'mg/dL'。
#     Returns:
#         tuple: 包含各区域百分比的字典和 matplotlib 的 Axes 对象。
#                如果数据无效则返回 (None, None)。
#     """
#     ref_values = np.asarray(ref_values)
#     pred_values = np.asarray(pred_values)
#
#     # --- 单位转换 (如果需要) ---
#     if unit == 'mmol/L':
#         # 注意： mmol/L 转换为 mg/dL 的因子约为 18.018 或 18
#         conversion_factor = 18.018
#         ref_values = ref_values * conversion_factor
#         pred_values = pred_values * conversion_factor
#         axis_unit = 'mg/dL (由 mmol/L 转换)'
#         print("注意：已将 mmol/L 单位转换为 mg/dL 用于绘制 CEG 图。")
#     elif unit == 'mg/dL':
#         axis_unit = 'mg/dL'
#     else:
#         raise ValueError("单位必须是 'mg/dL' 或 'mmol/L'")
#
#     # --- 数据有效性检查 ---
#     # 确保数值为非负数，CEG 通常用于血糖值
#     mask = (ref_values >= 0) & (pred_values >= 0)
#     ref_values = ref_values[mask]
#     pred_values = pred_values[mask]
#
#     if len(ref_values) == 0:
#         print("警告：没有有效的非负数据点来绘制 CEG 图。")
#         return None, None
#
#     # --- 区域计算 (根据标准定义，分层逻辑) ---
#     zone_a, zone_b, zone_c, zone_d, zone_e = 0, 0, 0, 0, 0
#     points_in_zones = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []} # 用于按区域存储点
#
#     for ref, pred in zip(ref_values, pred_values):
#         point = (ref, pred)
#         assigned = False # 标记点是否已被分配到某个区域
#
#         # 1. 检查 A 区 (临床准确)
#         #    标准: |预测值 - 参考值| <= 15 (如果 参考值 <= 70)
#         #          |预测值 - 参考值| / 参考值 <= 0.20 (如果 参考值 > 70)
#         if (ref <= 70 and abs(ref - pred) <= 15) or \
#            (ref > 70 and abs(ref - pred) <= 0.20 * ref):
#             zone_a += 1
#             points_in_zones['A'].append(point)
#             assigned = True
#
#         # 2. 检查 E 区 (危险错误)
#         #    标准: (参考值 <= 70 且 预测值 >= 180) 或 (参考值 >= 180 且 预测值 <= 70)
#         if not assigned:
#             if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
#                 zone_e += 1
#                 points_in_zones['E'].append(point)
#                 assigned = True
#
#         # 3. 检查 C 区 (过度校正)
#         #    标准: (参考值 >= 70 且 预测值 > 上 A 界 且 预测值 >= 180)  (上 C 区)
#         #          (参考值 < 70  且 预测值 < 下 A 界 且 预测值 <= 70)   (下 C 区)
#         if not assigned:
#             is_c_upper = (ref >= 70 and pred > ref * 1.2 and pred >= 180)
#             is_c_lower = (ref < 70 and pred < ref - 15 and pred <= 70)
#             if is_c_upper or is_c_lower:
#                 zone_c += 1
#                 points_in_zones['C'].append(point)
#                 assigned = True
#
#         # 4. 检查 D 区 (潜在检测失败) - 位于角落，非A/C/E
#         #    标准: (参考值 >= 180 且 70 < 预测值 < 下 A 界)  (右下 D 区)
#         #          (参考值 <= 70 且 180 > 预测值 > 上 A 界)  (左上 D 区)
#         if not assigned:
#             # is_d_lower_right = (ref >= 180 and pred > 70 and pred < ref * 0.8) # 检查是否严格在下A界以下
#             is_d_lower_right = (ref >= 180 and pred > 70 and (pred < ref - 15 if ref <= 70 else pred < ref * 0.8)) # 使用精确的A区边界
#             # is_d_upper_left = (ref <= 70 and pred < 180 and pred > ref + 15) # 检查是否严格在上A界以上
#             is_d_upper_left = (ref <= 70 and pred < 180 and (pred > ref + 15 if ref <= 70 else pred > ref * 1.2)) # 使用精确的A区边界
#
#             # 简化D区判断：如果点在E区定义的“危险”角落区域，但不满足E区的严格条件，且不在A区，则归为D区
#             # E区定义了两个矩形区域：(Ref<=70, Pred>=180) 和 (Ref>=180, Pred<=70)
#             # D区可以认为是这两个区域附近、但不那么极端的点
#             # (Ref >= 180 and pred <= 70)是E区，如果 Ref >= 180 但 pred > 70 且显著低于 Ref (例如 < 0.8*Ref)，可能是D区
#             # (Ref <= 70 and pred >= 180)是E区，如果 Ref <= 70 但 pred < 180 且显著高于 Ref (例如 > Ref+15)，可能是D区
#             if (ref >= 180 and pred > 70 and pred < ref * 0.8) or \
#                (ref <= 70 and pred < 180 and pred > ref + 15):
#                  # 确保这一点没有被之前的区域捕获 (例如 A 区)
#                  # 因为我们用了 if not assigned 结构，这里自动排除了 A, E, C
#                  zone_d += 1
#                  points_in_zones['D'].append(point)
#                  assigned = True
#
#
#         # 5. 剩余的点属于 B 区 (临床可接受误差)
#         if not assigned:
#             zone_b += 1
#             points_in_zones['B'].append(point)
#
#     total_points = len(ref_values)
#     if total_points == 0:
#         percentages = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
#     else:
#         percentages = {
#             'A': zone_a / total_points * 100,
#             'B': zone_b / total_points * 100,
#             'C': zone_c / total_points * 100,
#             'D': zone_d / total_points * 100,
#             'E': zone_e / total_points * 100,
#         }
#     print(f"CEG Percentages: A={percentages['A']:.1f}%, B={percentages['B']:.1f}%, C={percentages['C']:.1f}%, D={percentages['D']:.1f}%, E={percentages['E']:.1f}%")
#
#     # --- 绘图 ---
#     fig, ax = plt.subplots(figsize=(8, 8))
#
#     # 确定绘图范围 (可以基于数据范围，但至少包含 0-400 mg/dL)
#     max_val_data = max(np.max(ref_values) if len(ref_values)>0 else 400,
#                        np.max(pred_values) if len(pred_values)>0 else 400)
#     plot_limit = max(400, max_val_data * 1.05) # 留一点边距
#     min_val = 0
#     ax.set_xlim(min_val, plot_limit)
#     ax.set_ylim(min_val, plot_limit)
#     ax.set_aspect('equal', adjustable='box') # 保证 x, y 轴比例相同
#
#     # 绘制不同区域的点 (使用不同颜色和标签)
#     colors = {'A': '#006400', 'B': '#1E90FF', 'C': '#FFA500', 'D': '#FF4500', 'E': '#8B0000'} # 深绿, 蓝, 橙, 橙红, 深红
#     labels = {
#         'A': f'A ({percentages["A"]:.1f}%)', 'B': f'B ({percentages["B"]:.1f}%)',
#         'C': f'C ({percentages["C"]:.1f}%)', 'D': f'D ({percentages["D"]:.1f}%)',
#         'E': f'E ({percentages["E"]:.1f}%)'
#     }
#
#     for zone, pts in points_in_zones.items():
#         if pts: # 检查列表非空
#              refs = [p[0] for p in pts]
#              preds = [p[1] for p in pts]
#              # 调整 s (大小) 和 alpha (透明度) 以获得更好效果
#              ax.scatter(refs, preds, color=colors[zone], s=12, alpha=0.7, label=labels[zone], zorder=3)
#
#     # 绘制标准的 CEG 网格线
#     t = np.linspace(min_val, plot_limit, 200) # 更平滑的线
#
#     # y = x 线
#     ax.plot(t, t, 'k-', linewidth=1, zorder=1)
#
#     # A 区边界线
#     t_low = np.linspace(min_val, 70, 50)
#     t_high = np.linspace(70, plot_limit, 150)
#
#     # 上边界: y = x + 15 (x <= 70), y = 1.2x (x > 70)
#     ax.plot(t_low, t_low + 15, 'k-', linewidth=1, zorder=2)
#     ax.plot(t_high, 1.2 * t_high, 'k-', linewidth=1, zorder=2)
#
#     # 下边界: y = x - 15 (x <= 70), y = 0.8x (x > 70)
#     ax.plot(t_low, np.maximum(0, t_low - 15), 'k-', linewidth=1, zorder=2) # 确保不低于0
#     ax.plot(t_high, 0.8 * t_high, 'k-', linewidth=1, zorder=2)
#
#     # 关键阈值线 (虚线)
#     ax.plot([70, 70], [min_val, plot_limit], 'k:', linewidth=0.8, zorder=1) # x=70
#     ax.plot([min_val, plot_limit], [70, 70], 'k:', linewidth=0.8, zorder=1) # y=70
#     ax.plot([180, 180], [min_val, plot_limit], 'k:', linewidth=0.8, zorder=1) # x=180
#     ax.plot([min_val, plot_limit], [180, 180], 'k:', linewidth=0.8, zorder=1) # y=180
#
#     # 添加区域标签 (大致位置)
#     # 使用 transform=ax.transAxes 使位置相对于坐标轴，避免数值问题
#     # (0,0) is bottom-left, (1,1) is top-right
#     ax.text(0.3, 0.3, 'A', fontsize=14, ha='center', va='center', alpha=0.5, transform=ax.transAxes)
#     ax.text(0.7, 0.7, 'A', fontsize=14, ha='center', va='center', alpha=0.5, transform=ax.transAxes)
#
#     ax.text(0.15, 0.5, 'B', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Upper B left
#     ax.text(0.5, 0.85, 'B', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Upper B top
#     ax.text(0.85, 0.5, 'B', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Lower B right
#     ax.text(0.5, 0.15, 'B', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Lower B bottom
#
#     ax.text(0.1, 0.9, 'C', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Top-left C
#     ax.text(0.9, 0.1, 'C', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Bottom-right C
#
#     ax.text(0.1, 0.7, 'D', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Top-left D
#     ax.text(0.7, 0.1, 'D', fontsize=12, ha='center', va='center', alpha=0.5, transform=ax.transAxes) # Bottom-right D
#
#     # E区通常点很少，且在角落，标签可以省略或放在D区附近
#
#     # --- 图表设置 ---
#     ax.set_xlabel(f'参考血糖值 ({axis_unit})', fontsize=12)
#     ax.set_ylabel(f'预测血糖值 ({axis_unit})', fontsize=12)
#     ax.set_title(title_str, fontsize=14)
#     ax.legend(loc='upper left', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.3, zorder=0) # 将网格置于底层
#
#     # --- 保存图表 ---
#     if save_path:
#         # 确保目录存在
#         save_dir = os.path.dirname(save_path)
#         if save_dir and not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             print(f"创建目录: {save_dir}")
#         try:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 避免标签被截断
#             print(f"CEG 图已保存至: {save_path}")
#         except Exception as e:
#             print(f"保存 CEG 图失败: {e}")
#
#     # plt.show() # 在脚本中调用时，通常不需要在这里 show()，除非明确需要交互显示
#
#     return percentages, ax # 返回百分比和 axes 对象
#
# def plot_scatter(targets, predictions, metrics, unit='mmol/L', save_path=None):
#     """绘制预测值 vs 真实值散点图"""
#     targets = np.asarray(targets).flatten()
#     predictions = np.asarray(predictions).flatten()
#     valid_idx = np.isfinite(targets) & np.isfinite(predictions)
#     targets, predictions = targets[valid_idx], predictions[valid_idx]
#     if len(targets) == 0: return
#
#     rmse = metrics.get('RMSE', np.nan)
#     mae = metrics.get('MAE', np.nan)
#
#     plt.figure(figsize=(8, 8))
#     min_val = min(0, np.min(targets), np.min(predictions)) * 0.95
#     max_val = max(np.max(targets), np.max(predictions)) * 1.05
#     if not (np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val):
#         min_val, max_val = 0, 25 # Default range for mmol/L if calculation fails
#
#     plt.scatter(targets, predictions, alpha=0.6, color='#0072BD', s=15, edgecolors='w', linewidths=0.5,
#                 label=f'RMSE = {rmse:.2f} {unit}\nMAE = {mae:.2f} {unit}')
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal y=x')
#     plt.xlabel(f"Actual Glucose ({unit})", fontsize=12)
#     plt.ylabel(f"Predicted Glucose ({unit})", fontsize=12)
#     plt.title("Actual vs Predicted Glucose Values", fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle=':')
#     plt.xlim([min_val, max_val])
#     plt.ylim([min_val, max_val])
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#
#     if save_path:
#         try:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Scatter plot saved to {save_path}")
#         except Exception as e: print(f"Error saving scatter plot: {e}")
#     plt.close()
#
# def plot_loss_curves(results_dict, save_path=None):
#     """绘制训练和验证损失曲线 (包括各分量)"""
#     epochs = range(1, len(results_dict.get('train_loss_total', [])) + 1)
#     if not epochs:
#         print("Warning: No loss data to plot.")
#         return
#
#     plt.figure(figsize=(12, 8))
#
#     # 主损失图 (Total Train vs Val)
#     plt.subplot(2, 2, 1)
#     plt.plot(epochs, results_dict.get('train_loss_total', []), 'o-', label='Train Total Loss', lw=1.5, ms=3)
#     plt.plot(epochs, results_dict.get('val_loss_total', []), 's-', label='Val Total Loss', lw=1.5, ms=3)
#     plt.title('Total Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle=':')
#
#     # 预测损失图
#     plt.subplot(2, 2, 2)
#     plt.plot(epochs, results_dict.get('train_loss_pred', []), 'o-', label='Train Pred Loss', lw=1.5, ms=3)
#     plt.plot(epochs, results_dict.get('val_loss_pred', []), 's-', label='Val Pred Loss', lw=1.5, ms=3)
#     plt.title('Prediction Loss (L1)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle=':')
#
#     # 重构损失图
#     plt.subplot(2, 2, 3)
#     plt.plot(epochs, results_dict.get('train_loss_recon', []), 'o-', label='Train Recon Loss', lw=1.5, ms=3)
#     # Validation recon loss might not be calculated/stored, check existence
#     if 'val_loss_recon' in results_dict:
#         plt.plot(epochs, results_dict['val_loss_recon'], 's-', label='Val Recon Loss', lw=1.5, ms=3)
#     plt.title('Reconstruction Loss (MSE)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle=':')
#
#     # 解耦损失图
#     plt.subplot(2, 2, 4)
#     plt.plot(epochs, results_dict.get('train_loss_disentangle', []), 'o-', label='Train Disentangle Loss', lw=1.5, ms=3)
#     if 'val_loss_disentangle' in results_dict:
#         plt.plot(epochs, results_dict['val_loss_disentangle'], 's-', label='Val Disentangle Loss', lw=1.5, ms=3)
#     plt.title('Disentanglement Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle=':')
#
#     plt.tight_layout()
#     if save_path:
#         try:
#             plt.savefig(save_path, dpi=300)
#             print(f"Loss curves plot saved to {save_path}")
#         except Exception as e: print(f"Error saving loss curves: {e}")
#     plt.close()
#
#
# def plot_factor_correlation_heatmap(latent_factors, factor_names=None, save_path=None):
#     """绘制潜因子之间的相关性热力图"""
#     if not latent_factors:
#         print("Warning: No latent factors provided for heatmap.")
#         return
#     # latent_factors is a list of tensors [(batch, dim), (batch, dim), ...]
#     # Concatenate and convert to numpy for correlation calculation
#     try:
#         z_all_np = torch.cat(latent_factors, dim=1).detach().cpu().numpy() # (batch, total_factor_dim)
#         if z_all_np.shape[0] < 2 :
#              print("Warning: Need at least 2 samples to calculate correlation.")
#              return
#         # Calculate correlation matrix
#         corr_matrix = np.corrcoef(z_all_np, rowvar=False) # Columns are variables
#
#         num_factors = len(latent_factors)
#         factor_dims = latent_factors[0].shape[1]
#         total_factor_dim = corr_matrix.shape[0]
#
#         # Create labels for heatmap axes
#         if factor_names and len(factor_names) == num_factors:
#             labels = [f"{name}_{i}" for name in factor_names for i in range(factor_dims)]
#         else:
#             labels = [f"F{f}_D{i}" for f in range(num_factors) for i in range(factor_dims)]
#         if len(labels) != total_factor_dim: # Fallback if calculation is complex
#              labels = [f"Dim_{i}" for i in range(total_factor_dim)]
#
#
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, # Annotations might be too dense
#                     xticklabels=labels if total_factor_dim<50 else False, # Hide labels if too many
#                     yticklabels=labels if total_factor_dim<50 else False,
#                     linewidths=0.5, linecolor='lightgray', vmin=-1, vmax=1)
#         plt.title('Correlation Heatmap of Latent Factor Dimensions', fontsize=14)
#         plt.xticks(rotation=90, fontsize=8)
#         plt.yticks(rotation=0, fontsize=8)
#         plt.tight_layout()
#
#         if save_path:
#             try:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"Latent factor correlation heatmap saved to {save_path}")
#             except Exception as e: print(f"Error saving heatmap: {e}")
#         plt.close()
#
#     except Exception as e:
#         print(f"Error generating factor correlation heatmap: {e}")
#
#
# # --------------------------------------------
# #             结果保存辅助函数
# # --------------------------------------------
# def save_results(results_dir, args, metrics, zones, targets=None, predictions=None, losses_dict=None):
#     """保存所有实验结果"""
#     os.makedirs(results_dir, exist_ok=True)
#
#     # 1. 保存参数
#     params_path = os.path.join(results_dir, 'parameters.txt')
#     try:
#         with open(params_path, 'w') as f:
#             for key, value in vars(args).items():
#                 f.write(f"{key}: {value}\n")
#             f.write("\n--- Test Metrics ---\n")
#             for key, value in metrics.items():
#                 f.write(f"{key}: {value:.4f}\n")
#             f.write("\n--- CEG Zones ---\n")
#             for key, value in zones.items():
#                  f.write(f"Zone {key} (%): {value:.2f}\n")
#         print(f"Parameters and metrics saved to {params_path}")
#     except Exception as e: print(f"Error saving parameters/metrics: {e}")
#
#     # 2. 保存预测值
#     if targets is not None and predictions is not None:
#         pred_path = os.path.join(results_dir, 'test_predictions.csv')
#         try:
#             results_df = pd.DataFrame({'Actual_Glucose': targets.flatten(), 'Predicted_Glucose': predictions.flatten()})
#             results_df.to_csv(pred_path, index=False, float_format='%.4f')
#             print(f"Test predictions saved to {pred_path}")
#         except Exception as e: print(f"Error saving predictions: {e}")
#
#     # 3. 保存损失历史 (可选)
#     if losses_dict:
#         loss_path = os.path.join(results_dir, 'loss_history.csv')
#         try:
#             loss_df = pd.DataFrame(losses_dict)
#             loss_df.to_csv(loss_path, index_label='Epoch')
#             print(f"Loss history saved to {loss_path}")
#         except Exception as e: print(f"Error saving loss history: {e}")

# 文件名: utils_cdn_v2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
import math
import seaborn as sns # 用于热力图

# --- 保留原有的辅助函数 (如果需要) ---
def calculate_regression_metrics(y_true, y_pred):
    """Calculates standard regression metrics."""
    # Ensure inputs are numpy arrays and handle potential NaNs/Infs
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        logging.warning("No valid points found for regression metrics calculation.")
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE carefully to avoid division by zero
    # Use a mask for non-zero true values
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
         mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
         mape = np.nan # Or 0 if appropriate, depends on definition

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

def plot_scatter(y_true, y_pred, metrics, unit='Unknown', title_prefix='Scatter Plot', save_path=None):
    """Generates a scatter plot of true vs predicted values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, label='Predictions') # Smaller points
    min_val = min(np.min(y_true), np.min(y_pred)) * 0.95 if len(y_true)>0 else 0
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05 if len(y_true)>0 else 10
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlabel(f'True Values ({unit})')
    plt.ylabel(f'Predicted Values ({unit})')
    plt.title(f'{title_prefix}\nRMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, R2={metrics["R2"]:.4f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Ensure equal aspect ratio after setting limits maybe?
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)


    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Scatter plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save scatter plot: {e}")
    # plt.show() # Usually commented out in scripts
    plt.close() # Close the figure to free memory

def plot_loss_curves(history, save_path=None):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss_total']) + 1)

    # Plot Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss_total'], label='Train Total Loss')
    if 'val_loss_total' in history and history['val_loss_total']: # Check if val loss exists and is not empty
       plt.plot(epochs, history['val_loss_total'], label='Val Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot Prediction Loss and maybe RMSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss_pred'], label='Train Prediction Loss')
    if 'val_loss_pred' in history and history['val_loss_pred']:
       plt.plot(epochs, history['val_loss_pred'], label='Val Prediction Loss')
    if 'val_rmse' in history and history['val_rmse']:
        ax2 = plt.gca().twinx() # Create a second y-axis for RMSE
        ax2.plot(epochs, history['val_rmse'], 'g--', label='Val RMSE')
        ax2.set_ylabel('RMSE', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        # Combine legends if needed or place separately
        # lines, labels = plt.gca().get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Loss')
    plt.title('Prediction Loss & Val RMSE')
    plt.gca().legend(loc='upper left') # For the loss curves
    if 'val_rmse' in history and history['val_rmse']: ax2.legend(loc='upper right') # For RMSE curve
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Loss curves plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save loss curves plot: {e}")
    # plt.show()
    plt.close()

def plot_factor_correlation_heatmap(factors_list, save_path=None):
    """Plots a heatmap of the correlation between different latent factors."""
    if not factors_list:
        logging.warning("No factors provided for correlation heatmap.")
        return

    try:
        # Assuming factors_list is a list of tensors [factor0_batch, factor1_batch, ...]
        num_factors = len(factors_list)
        factor_dim = factors_list[0].shape[1] # Get dimension of factors

        # Combine factors: average over the factor dimension for each sample
        # Or take the first element? Averaging seems more robust.
        avg_factors = [f.mean(dim=1).numpy() for f in factors_list] # List of (num_samples,) arrays
        factor_data = np.stack(avg_factors, axis=1) # Shape (num_samples, num_factors)

        if factor_data.shape[0] < 2: # Need at least 2 samples for correlation
            logging.warning("Not enough samples to calculate factor correlation.")
            return

        # Create a pandas DataFrame for easier correlation calculation and plotting
        df_factors = pd.DataFrame(factor_data, columns=[f'Factor_{i}' for i in range(num_factors)])

        # Calculate correlation matrix
        corr_matrix = df_factors.corr()

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Latent Factor Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Factor correlation heatmap saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save factor correlation heatmap: {e}")
        # plt.show()
        plt.close()

    except Exception as e:
        logging.error(f"Error generating factor correlation heatmap: {e}", exc_info=True)

def save_results(results_dir, args, test_metrics, # ceg_zones_perc, # Removed - calculated/printed internally now
                 test_targets, test_preds, history):
    """Saves hyperparameters, metrics, predictions, and loss history."""
    try:
        # Save hyperparameters and test metrics
        params_path = os.path.join(results_dir, 'parameters.txt')
        with open(params_path, 'w') as f:
            f.write("Hyperparameters:\n")
            for key, value in vars(args).items():
                f.write(f"  {key}: {value}\n")
            f.write("\nTest Set Metrics:\n")
            for key, value in test_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            # Add CEG results here if needed, though they are printed by the plot func
            # f.write("\nCEG Zone Distribution (%, A,B,C,D,E):\n")
            # f.write(f"  {ceg_zones_perc.tolist()}\n") # Assuming it's a list/array
        logging.info(f"Parameters and metrics saved to {params_path}")

        # Save test predictions
        preds_df = pd.DataFrame({'y_true': test_targets.flatten(), 'y_pred': test_preds.flatten()})
        preds_path = os.path.join(results_dir, 'test_predictions.csv')
        preds_df.to_csv(preds_path, index=False)
        logging.info(f"Test predictions saved to {preds_path}")

        # Save loss history
        history_df = pd.DataFrame(history)
        history_path = os.path.join(results_dir, 'loss_history.csv')
        history_df.to_csv(history_path, index=False)
        logging.info(f"Loss history saved to {history_path}")

    except Exception as e:
        logging.error(f"Failed to save results: {e}", exc_info=True)


# ============================================================================
#  克拉克误差网格分析 (MATLAB 样式复现) - 添加的新函数
# ============================================================================
def plot_clarke_ega_matlab_style(y_ref, y_pred, title_str="", save_path=None):
    """
    执行克拉克误差网格分析 (Clarke Error Grid Analysis)，
    复现提供的 MATLAB 脚本 'clarke.m' 的逻辑和绘图样式。

    Args:
        y_ref (array-like): 参考血糖值 (单位必须是 mg/dL)。
        y_pred (array-like): 预测血糖值 (单位必须是 mg/dL)。
        title_str (str, optional): 图表的标题。
        save_path (str, optional): 保存绘图图片的路径 (例如, 'clarke_ega.png')。
                                   默认为 None (不保存)。

    Returns:
        tuple: (total, percentage)
            - total (numpy.ndarray): 包含 A, B, C, D, E 五个区域计数的 5 元素数组。
                                     索引 0=A, 1=B, 2=C, 3=D, 4=E。
            - percentage (numpy.ndarray): 包含每个区域百分比的 5 元素数组。
    Raises:
        ValueError: 如果输入长度不一致。
    """
    y_ref = np.asarray(y_ref)
    y_pred = np.asarray(y_pred)

    # --- 输入验证 (类似 MATLAB 脚本) ---
    if y_ref.shape != y_pred.shape:
        raise ValueError("参考值 (y_ref) 和预测值 (y_pred) 向量长度必须相同。")
    if np.any(y_ref < 0) or np.any(y_pred < 0) or np.any(y_ref > 400) or np.any(y_pred > 400):
        print("警告：输入值理想情况下应在生理血糖范围 [0, 400] mg/dL 内。")

    n = len(y_ref)
    if n == 0:
        print("警告：输入数组为空。")
        return np.zeros(5), np.zeros(5)

    # --- 绘图 (复现 MATLAB plot 命令) ---
    fig, ax = plt.subplots(figsize=(6, 6)) # 调整图像大小

    # 绘制数据散点图
    ax.scatter(y_ref, y_pred, c='black', s=10, marker='o', edgecolors='black', facecolors='black', zorder=5)

    ax.set_xlabel('Reference Concentration [mg/dl]')
    ax.set_ylabel('Prediction Concentration [mg/dl]')
    ax.set_title(title_str)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal', adjustable='box') # 使坐标轴方形
    ax.grid(False) # 不显示网格

    # 绘制边界线 (严格按照 MATLAB 代码)
    ax.plot([0, 400], [0, 400], 'k:', linewidth=1)  # 对角线 y=x
    ax.plot([0, 175/3], [70, 70], 'k-', linewidth=1)
    ax.plot([175/3, 400/1.2], [70, 400], 'k-', linewidth=1)
    ax.plot([0, 70], [180, 180], 'k-', linewidth=1)
    ax.plot([70, 290], [180, 400], 'k-', linewidth=1) # MATLAB 源码中修正过的上 B-C 边界
    ax.plot([70, 70], [0, 56], 'k-', linewidth=1) # 调整过的下 A 区边界交点
    ax.plot([70, 400], [56, 320], 'k-', linewidth=1) # 下 A/B 区边界线 第2部分
    ax.plot([180, 180], [0, 70], 'k-', linewidth=1)
    ax.plot([180, 400], [70, 70], 'k-', linewidth=1)
    ax.plot([240, 240], [70, 180], 'k-', linewidth=1)
    ax.plot([240, 400], [180, 180], 'k-', linewidth=1)
    ax.plot([130, 180], [0, 70], 'k-', linewidth=1) # MATLAB 源码中的下 B-C 边界

    # 添加区域文本标签 (位置来自 MATLAB 代码)
    ax.text(30, 20, 'A', fontsize=12)
    ax.text(30, 150, 'D', fontsize=12)
    ax.text(30, 380, 'E', fontsize=12)
    ax.text(150, 380, 'C', fontsize=12)
    ax.text(160, 20, 'C', fontsize=12)
    ax.text(380, 20, 'E', fontsize=12)
    ax.text(380, 120, 'D', fontsize=12)
    ax.text(380, 260, 'B', fontsize=12)
    ax.text(280, 380, 'B', fontsize=12)

    # --- 区域计算 (复现 MATLAB 的统计逻辑) ---
    total = np.zeros(5) # 索引 0=A, 1=B, 2=C, 3=D, 4=E

    for i in range(n):
        ref = y_ref[i]
        pred = y_pred[i]

        # A 区条件 (MATLAB 版本)
        if (pred <= 70 and ref <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
            total[0] += 1 # A 区
        else:
            # E 区条件
            if ((ref >= 180) and (pred <= 70)) or ((ref <= 70) and pred >= 180):
                total[4] += 1 # E 区
            else:
                # C 区条件 (MATLAB 版本)
                safe_ref = max(ref, 1e-9) # 避免除零
                cond_c1 = ((ref >= 70 and ref <= 290) and (pred >= ref + 110))
                cond_c2 = ((ref >= 130 and ref <= 180) and (pred <= (7/5) * safe_ref - 182))
                if cond_c1 or cond_c2:
                     total[2] += 1 # C 区
                else:
                    # D 区条件 (MATLAB 版本)
                    cond_d1 = ((ref >= 240) and (pred >= 70 and pred <= 180))
                    cond_d2 = (ref <= 175/3 and (pred <= 180 and pred >= 70))
                    cond_d3 = ((ref >= 175/3 and ref <= 70) and (pred >= (6/5) * safe_ref))
                    if cond_d1 or cond_d2 or cond_d3:
                        total[3] += 1 # D 区
                    else:
                        # B 区 (所有剩余的点)
                        total[1] += 1 # B 区

    # 计算百分比
    percentage = (total / n) * 100 if n > 0 else np.zeros(5)

    print("-" * 30)
    print(f"克拉克误差网格区域分布 (MATLAB 样式):")
    print(f"  A 区: {total[0]:>5} 个点 ({percentage[0]:>6.2f}%)")
    print(f"  B 区: {total[1]:>5} 个点 ({percentage[1]:>6.2f}%)")
    print(f"  C 区: {total[2]:>5} 个点 ({percentage[2]:>6.2f}%)")
    print(f"  D 区: {total[3]:>5} 个点 ({percentage[3]:>6.2f}%)")
    print(f"  E 区: {total[4]:>5} 个点 ({percentage[4]:>6.2f}%)")
    print(f"  总计: {n:>5} 个点")
    print("-" * 30)

    # --- 保存绘图 (可选) ---
    if save_path:
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"克拉克误差网格图 (MATLAB 样式) 已保存至: {save_path}")
        except Exception as e:
            print(f"保存绘图至 {save_path} 时出错: {e}")

    # plt.show()
    plt.close() # 关闭图像，释放内存

    return total, percentage

# --- 可能存在的旧 CEG 函数 (如果不再需要可以删除) ---
# def calculate_ceg_zones_mmol(ref_values_mmol, pred_values_mmol):
#     # ... (旧代码或标准代码) ...
#     pass
#
# def plot_clarke_error_grid_mmol(ref_values_mmol, pred_values_mmol, save_path=None):
#     # ... (旧代码或标准代码) ...
#     pass
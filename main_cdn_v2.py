# # 文件名: main_cdn_v2.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset # Subset for creating validation split
# from sklearn.model_selection import train_test_split # For val split indices
# import numpy as np
# import pandas as pd
# import os
# import time
# import argparse
# import datetime
# import logging
# import math
#
# # 从其他文件导入
# try:
#     from load_cdn_v2 import CdnDatasetV2
#     from model_cdn_v2 import CausalDisentanglementNetworkAdaptedV2, CDNLossAdapted
#     from utils_cdn_v2 import (calculate_regression_metrics, calculate_ceg_zones_mmol,
#                             plot_clarke_error_grid, plot_scatter, plot_loss_curves,
#                             plot_factor_correlation_heatmap, save_results)
# except ImportError as e:
#     print(f"Error importing modules: {e}")
#     print("Make sure all .py files (preprocess_*, load_*, model_*, utils_*, main_*) are in the same directory.")
#     exit()
#
# # --------------------------------------------
# #            参数解析器
# # --------------------------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train and evaluate CDN V2 model for glucose prediction.")
#     # Paths
#     # parser.add_argument('--data_dir', type=str, default='./processed_cdn_data_v2', help='Directory of processed .npy files')
#     parser.add_argument('--data_dir', type=str, default='./processed_cdn_data_v2_fake', help='Directory of processed .npy files')
#
#     parser.add_argument('--results_root', type=str, default='./cdn_experiments', help='Root directory for saving experiment results')
#     parser.add_argument('--exp_name', type=str, default='CDN_Run', help='Experiment name prefix')
#     # Model Hyperparameters (allow adjustments)
#     parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for PPG patches')
#     parser.add_argument('--patch_size', type=int, default=8, help='Patch size for PatchEmbedding')
#     parser.add_argument('--patch_stride', type=int, default=4, help='Patch stride for PatchEmbedding')
#     parser.add_argument('--tfr_layers', type=int, default=2, help='Number of Transformer layers')
#     parser.add_argument('--tfr_heads', type=int, default=4, help='Number of Transformer heads')
#     parser.add_argument('--num_factors', type=int, default=5, help='Number of latent factors')
#     parser.add_argument('--factor_dims', type=int, default=16, help='Dimension of each latent factor')
#     parser.add_argument('--condition_dim', type=int, default=32, help='Dimension of the condition vector from context')
#     parser.add_argument('--glucose_idx', type=int, default=3, help='Index of the glucose factor (0 to num_factors-1)')
#     # Training Hyperparameters
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # Smaller LR might be needed for Transformers
#     parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
#     parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
#     parser.add_argument('--loss_pred_w', type=float, default=1.0, help='Weight for prediction loss')
#     parser.add_argument('--loss_recon_w', type=float, default=0.3, help='Weight for reconstruction loss')
#     parser.add_argument('--loss_dis_w', type=float, default=0.05, help='Weight for disentanglement loss') # Often needs smaller weight
#     # Runtime & Setup
#     parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of training data to use for validation')
#     parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 for CPU)')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
#
#     args = parser.parse_args()
#     # Basic validation
#     if not (0 <= args.glucose_idx < args.num_factors):
#         parser.error(f"--glucose_idx ({args.glucose_idx}) must be between 0 and --num_factors-1 ({args.num_factors-1})")
#     return args
#
# # --------------------------------------------
# #            日志设置
# # --------------------------------------------
# def setup_logging(log_file):
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s [%(levelname)s] %(message)s',
#                         handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
#
# # --------------------------------------------
# #            训练与评估循环
# # --------------------------------------------
# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     epoch_losses = {'total': 0.0, 'pred': 0.0, 'recon': 0.0, 'disentangle': 0.0}
#     sample_count = 0
#     for ppg, context, target in dataloader:
#         ppg, context, target = ppg.to(device), context.to(device), target.to(device)
#         batch_size = ppg.size(0)
#         sample_count += batch_size
#
#         optimizer.zero_grad()
#         outputs = model(ppg, context)
#         total_loss, loss_p, loss_r, loss_d = criterion(outputs, target, ppg)
#
#         total_loss.backward()
#         # Optional: Gradient clipping
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#
#         epoch_losses['total'] += total_loss.item() * batch_size
#         epoch_losses['pred'] += loss_p.item() * batch_size
#         epoch_losses['recon'] += loss_r.item() * batch_size
#         epoch_losses['disentangle'] += loss_d.item() * batch_size
#
#     # Average losses over samples
#     for key in epoch_losses:
#         epoch_losses[key] /= sample_count if sample_count > 0 else 1
#     return epoch_losses
#
# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     epoch_losses = {'total': 0.0, 'pred': 0.0, 'recon': 0.0, 'disentangle': 0.0}
#     all_targets, all_preds, all_factors = [], [], []
#     sample_count = 0
#     with torch.no_grad():
#         for ppg, context, target in dataloader:
#             ppg, context, target = ppg.to(device), context.to(device), target.to(device)
#             batch_size = ppg.size(0)
#             sample_count += batch_size
#
#             outputs = model(ppg, context)
#             total_loss, loss_p, loss_r, loss_d = criterion(outputs, target, ppg)
#
#             epoch_losses['total'] += total_loss.item() * batch_size
#             epoch_losses['pred'] += loss_p.item() * batch_size
#             epoch_losses['recon'] += loss_r.item() * batch_size
#             epoch_losses['disentangle'] += loss_d.item() * batch_size
#
#             all_targets.append(target.cpu())
#             all_preds.append(outputs['predicted_glucose'].cpu())
#             # Store latent factors for analysis (detach from graph)
#             all_factors.append([f.cpu().detach() for f in outputs['latent_factors']])
#
#     # Average losses
#     for key in epoch_losses:
#         epoch_losses[key] /= sample_count if sample_count > 0 else 1
#
#     # Concatenate results
#     targets_np = torch.cat(all_targets, dim=0).numpy() if all_targets else np.array([])
#     preds_np = torch.cat(all_preds, dim=0).numpy() if all_preds else np.array([])
#
#     # Reorganize factors: list of factor tensors -> list of batch tensors
#     num_factors = model.num_factors
#     factors_by_type = []
#     if all_factors and len(all_factors[0]) == num_factors:
#          for i in range(num_factors):
#               factor_i_batches = [batch_factors[i] for batch_factors in all_factors]
#               factors_by_type.append(torch.cat(factor_i_batches, dim=0))
#
#     # Calculate regression metrics using only prediction loss for validation/test reporting
#     metrics = calculate_regression_metrics(targets_np, preds_np)
#     metrics['loss_pred'] = epoch_losses['pred'] # Use prediction loss for eval comparison
#     metrics['loss_total'] = epoch_losses['total']
#
#     return metrics, targets_np, preds_np, factors_by_type # Return factors for analysis
#
# # --------------------------------------------
# #            主函数
# # --------------------------------------------
# def main():
#     args = parse_args()
#
#     # --- Setup ---
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.gpu_id != -1 and torch.cuda.is_available():
#         device = torch.device(f"cuda:{args.gpu_id}")
#         torch.cuda.manual_seed_all(args.seed)
#         # Consider setting deterministic for reproducibility, might impact performance
#         # torch.backends.cudnn.deterministic = True
#         # torch.backends.cudnn.benchmark = False
#         print(f"Using GPU: {args.gpu_id}")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")
#
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = os.path.join(args.results_root, f"{args.exp_name}_{timestamp}")
#     model_save_dir = os.path.join(results_dir, 'Model')
#     os.makedirs(results_dir, exist_ok=True)
#     os.makedirs(model_save_dir, exist_ok=True)
#
#     log_file = os.path.join(results_dir, 'run.log')
#     setup_logging(log_file)
#     logging.info("CDN V2 Experiment Started")
#     logging.info(f"Args: {vars(args)}")
#     logging.info(f"Device: {device}")
#     logging.info(f"Results Dir: {results_dir}")
#
#     # --- Data Loading & Splitting ---
#     logging.info("Loading data...")
#     try:
#         full_train_dataset = CdnDatasetV2(args.data_dir, split='train')
#         test_dataset = CdnDatasetV2(args.data_dir, split='test')
#
#         # Create validation split from training data
#         train_indices, val_indices = train_test_split(
#             np.arange(len(full_train_dataset)),
#             test_size=args.val_split,
#             random_state=args.seed,
#             # Add stratification if dataset includes labels usable for it
#         )
#         train_dataset = Subset(full_train_dataset, train_indices)
#         val_dataset = Subset(full_train_dataset, val_indices)
#         logging.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
#
#         pin_memory = True if device.type == 'cuda' else False
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
#         val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
#         test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
#         logging.info("DataLoaders ready.")
#     except Exception as e:
#          logging.error(f"Data loading/splitting failed: {e}", exc_info=True)
#          return
#
#     # --- Model, Loss, Optimizer ---
#     logging.info("Initializing model, criterion, optimizer...")
#     try:
#         # Determine context_dim from loaded data if possible
#         context_dim_data = full_train_dataset.X_context.shape[1] if hasattr(full_train_dataset, 'X_context') else 8
#         logging.info(f"Using Context Dim: {context_dim_data}")
#
#         model = CausalDisentanglementNetworkAdaptedV2(
#             context_dim=context_dim_data,
#             embed_dim=args.embed_dim, patch_size=args.patch_size, patch_stride=args.patch_stride,
#             transformer_layers=args.tfr_layers, transformer_heads=args.tfr_heads,
#             num_factors=args.num_factors, factor_dims=args.factor_dims,
#             condition_dim=args.condition_dim, glucose_factor_index=args.glucose_idx
#         ).to(device)
#
#         criterion = CDNLossAdapted(
#             weight_pred=args.loss_pred_w, weight_recon=args.loss_recon_w, weight_disentangle=args.loss_dis_w,
#             num_factors=args.num_factors, factor_dims=args.factor_dims
#         ).to(device)
#
#         optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # AdamW often better
#         # Optional: Learning rate scheduler
#         # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
#
#         logging.info(f"Model:\n{model}")
#         logging.info(f"Criterion: {criterion}")
#         logging.info(f"Optimizer: {optimizer}")
#     except Exception as e:
#         logging.error(f"Model/Criterion/Optimizer initialization failed: {e}", exc_info=True)
#         return
#
#
#     # --- Training Loop ---
#     logging.info(f"Starting training for {args.epochs} epochs...")
#     best_val_metric = float('inf') # Use validation prediction loss or RMSE
#     history = {'train_loss_total': [], 'train_loss_pred': [], 'train_loss_recon': [], 'train_loss_disentangle': [],
#                'val_loss_total': [], 'val_loss_pred': [], 'val_rmse': [], 'val_mae': []}
#     model_save_path = os.path.join(model_save_dir, 'best_model.pth')
#
#     for epoch in range(args.epochs):
#         start_time = time.time()
#         train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
#         val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
#         end_time = time.time()
#
#         # Store history
#         history['train_loss_total'].append(train_losses['total'])
#         history['train_loss_pred'].append(train_losses['pred'])
#         history['train_loss_recon'].append(train_losses['recon'])
#         history['train_loss_disentangle'].append(train_losses['disentangle'])
#         history['val_loss_total'].append(val_metrics['loss_total'])
#         history['val_loss_pred'].append(val_metrics['loss_pred'])
#         history['val_rmse'].append(val_metrics['RMSE'])
#         history['val_mae'].append(val_metrics['MAE'])
#
#         logging.info(f"Epoch {epoch+1}/{args.epochs} | Time: {end_time - start_time:.1f}s | "
#                      f"Train Loss: {train_losses['total']:.4f} (P:{train_losses['pred']:.4f}, R:{train_losses['recon']:.4f}, D:{train_losses['disentangle']:.4f}) | "
#                      f"Val Loss: {val_metrics['loss_total']:.4f} (P:{val_metrics['loss_pred']:.4f}) | "
#                      f"Val RMSE: {val_metrics['RMSE']:.4f}")
#
#         # Update scheduler if used
#         # scheduler.step(val_metrics['loss_pred'])
#
#         # Save best model based on validation prediction loss
#         current_val_metric = val_metrics['loss_pred']
#         if current_val_metric < best_val_metric:
#             best_val_metric = current_val_metric
#             try:
#                 torch.save(model.state_dict(), model_save_path)
#                 logging.info(f"  >> Best model saved to {model_save_path} (Val Pred Loss: {best_val_metric:.4f})")
#             except Exception as e: logging.error(f"  Error saving model: {e}")
#
#     logging.info("Training finished.")
#
#     # --- Final Evaluation on Test Set ---
#     logging.info("Loading best model for final test evaluation...")
#     try:
#         model.load_state_dict(torch.load(model_save_path, map_location=device))
#         logging.info("Best model loaded.")
#     except Exception as e:
#         logging.error(f"Failed to load best model from {model_save_path}: {e}. Evaluating with last model state.")
#
#     logging.info("Evaluating on test set...")
#     test_metrics, test_targets, test_preds, test_factors = evaluate(model, test_loader, criterion, device)
#
#     if test_targets is not None and test_preds is not None:
#          logging.info(f"Test Results: Pred Loss={test_metrics['loss_pred']:.4f} | RMSE={test_metrics['RMSE']:.4f} | MAE={test_metrics['MAE']:.4f} | MAPE={test_metrics['MAPE']:.2f}% | R2={test_metrics['R2']:.4f}")
#
#          # Calculate CEG Zones
#          logging.info("Calculating CEG zones...")
#          ceg_zones_perc = calculate_ceg_zones_mmol(test_targets, test_preds)
#
#          # --- Save Results & Plots ---
#          logging.info("Saving results and generating plots...")
#          save_results(results_dir, args, test_metrics, ceg_zones_perc, test_targets, test_preds, history)
#          plot_loss_curves(history, save_path=os.path.join(results_dir, 'loss_curves.png'))
#          plot_scatter(test_targets, test_preds, test_metrics, unit='mmol/L', save_path=os.path.join(results_dir, 'scatter_plot.png'))
#          plot_clarke_error_grid(
#              test_targets,  # 参考值
#              test_preds,  # 预测值
#              f'CEG - Test Set ({args.exp_name}) - Corrected',  # 使用 args.exp_name 生成标题
#              save_path=os.path.join(results_dir, 'clarke_error_grid_corrected.png'),  # 保存路径
#              unit='mmol/L'  # 单位参数 (请确保与您的数据单位一致)
#          )       # Plot factor correlation heatmap using test set factors
#          plot_factor_correlation_heatmap(test_factors, save_path=os.path.join(results_dir, 'factor_correlation_heatmap.png'))
#
#          logging.info("Plots generated.")
#     else:
#          logging.error("Test evaluation failed. No results to save or plot.")
#
#     logging.info(f"Experiment finished. Results are in: {results_dir}")
#
#
# if __name__ == "__main__":
#     # 检查数据预处理是否完成
#     args_temp = parse_args() # 解析一次以获取 data_dir
#     if not os.path.exists(args_temp.data_dir) or not os.listdir(args_temp.data_dir):
#          print(f"Error: Processed data directory '{args_temp.data_dir}' is empty or does not exist.")
#          print("Please run 'python preprocess_cdn_v2.py' first.")
#     else:
#          main()
# 文件名: main_cdn_v2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset # Subset 用于创建验证集
from sklearn.model_selection import train_test_split # 用于获取验证集索引
import numpy as np
import pandas as pd
import os
import time
import argparse
import datetime
import logging
import math

# 从其他文件导入
try:
    from load_cdn_v2 import CdnDatasetV2
    from model_cdn_v2 import CausalDisentanglementNetworkAdaptedV2, CDNLossAdapted
    # 导入所有需要的 utils 函数，包括新添加的
    from utils_cdn_v2 import (calculate_regression_metrics,
                            plot_scatter, plot_loss_curves,
                            plot_factor_correlation_heatmap, save_results,
                            plot_clarke_ega_matlab_style) # 导入新的 CEG 函数
    # 旧的 CEG 函数导入可以注释掉或删除，如果不再使用
    # from utils_cdn_v2 import calculate_ceg_zones_mmol, plot_clarke_error_grid_mmol
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保所有 .py 文件 (preprocess_*, load_*, model_*, utils_*, main_*) 都在同一目录下。")
    exit()

# --------------------------------------------
#            参数解析器
# --------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="训练并评估用于血糖预测的 CDN V2 模型。")
    # 路径
    # parser.add_argument('--data_dir', type=str, default='./processed_cdn_data_v2', help='处理后的 .npy 文件目录')
    parser.add_argument('--data_dir', type=str, default='./processed_cdn_data_v2_heavy_leakage', help='处理后的 .npy 文件目录') # 使用假数据路径

    parser.add_argument('--results_root', type=str, default='./cdn_experiments', help='保存实验结果的根目录')
    parser.add_argument('--exp_name', type=str, default='CDN_Run', help='实验名称前缀')
    # 模型超参数 (允许调整)
    parser.add_argument('--embed_dim', type=int, default=64, help='PPG Patch 的嵌入维度')
    parser.add_argument('--patch_size', type=int, default=8, help='PatchEmbedding 的 Patch 大小')
    parser.add_argument('--patch_stride', type=int, default=4, help='PatchEmbedding 的 Patch 步长')
    parser.add_argument('--tfr_layers', type=int, default=2, help='Transformer 层数')
    parser.add_argument('--tfr_heads', type=int, default=4, help='Transformer 注意力头数')
    parser.add_argument('--num_factors', type=int, default=5, help='隐因子数量')
    parser.add_argument('--factor_dims', type=int, default=16, help='每个隐因子的维度')
    parser.add_argument('--condition_dim', type=int, default=32, help='上下文条件向量的维度')
    parser.add_argument('--glucose_idx', type=int, default=3, help='血糖因子的索引 (0 到 num_factors-1)')
    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='优化器的权重衰减')
    parser.add_argument('--loss_pred_w', type=float, default=1.0, help='预测损失的权重')
    parser.add_argument('--loss_recon_w', type=float, default=0.3, help='重构损失的权重')
    parser.add_argument('--loss_dis_w', type=float, default=0.05, help='解耦损失的权重')
    # 运行时和设置
    parser.add_argument('--val_split', type=float, default=0.15, help='用于验证的训练数据比例')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 表示 CPU)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 的工作进程数')

    args = parser.parse_args()
    # 基本验证
    if not (0 <= args.glucose_idx < args.num_factors):
        parser.error(f"--glucose_idx ({args.glucose_idx}) 必须在 0 和 --num_factors-1 ({args.num_factors-1}) 之间")
    return args

# --------------------------------------------
#            日志设置
# --------------------------------------------
def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# --------------------------------------------
#            训练与评估循环
# --------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_losses = {'total': 0.0, 'pred': 0.0, 'recon': 0.0, 'disentangle': 0.0}
    sample_count = 0
    for ppg, context, target in dataloader:
        ppg, context, target = ppg.to(device), context.to(device), target.to(device)
        batch_size = ppg.size(0)
        sample_count += batch_size

        optimizer.zero_grad()
        outputs = model(ppg, context)
        # 假设 criterion 返回: total_loss, loss_p, loss_r, loss_d
        total_loss, loss_p, loss_r, loss_d = criterion(outputs, target, ppg)

        total_loss.backward()
        optimizer.step()

        epoch_losses['total'] += total_loss.item() * batch_size
        epoch_losses['pred'] += loss_p.item() * batch_size
        epoch_losses['recon'] += loss_r.item() * batch_size
        epoch_losses['disentangle'] += loss_d.item() * batch_size

    # 计算样本平均损失
    for key in epoch_losses:
        epoch_losses[key] /= sample_count if sample_count > 0 else 1
    return epoch_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_losses = {'total': 0.0, 'pred': 0.0, 'recon': 0.0, 'disentangle': 0.0}
    all_targets, all_preds, all_factors = [], [], []
    sample_count = 0
    with torch.no_grad():
        for ppg, context, target in dataloader:
            ppg, context, target = ppg.to(device), context.to(device), target.to(device)
            batch_size = ppg.size(0)
            sample_count += batch_size

            outputs = model(ppg, context)
            total_loss, loss_p, loss_r, loss_d = criterion(outputs, target, ppg)

            epoch_losses['total'] += total_loss.item() * batch_size
            epoch_losses['pred'] += loss_p.item() * batch_size
            epoch_losses['recon'] += loss_r.item() * batch_size
            epoch_losses['disentangle'] += loss_d.item() * batch_size

            all_targets.append(target.cpu())
            all_preds.append(outputs['predicted_glucose'].cpu())
            # 存储隐因子用于分析 (从计算图中分离)
            all_factors.append([f.cpu().detach() for f in outputs['latent_factors']])

    # 平均损失
    for key in epoch_losses:
        epoch_losses[key] /= sample_count if sample_count > 0 else 1

    # 连接结果
    targets_np = torch.cat(all_targets, dim=0).numpy() if all_targets else np.array([])
    preds_np = torch.cat(all_preds, dim=0).numpy() if all_preds else np.array([])

    # 重组因子: list of factor tensors -> list of batch tensors
    num_factors = model.num_factors
    factors_by_type = []
    if all_factors and len(all_factors[0]) == num_factors:
         for i in range(num_factors):
              factor_i_batches = [batch_factors[i] for batch_factors in all_factors]
              factors_by_type.append(torch.cat(factor_i_batches, dim=0))

    # 计算回归指标 (仅使用预测损失进行验证/测试报告)
    metrics = calculate_regression_metrics(targets_np, preds_np)
    metrics['loss_pred'] = epoch_losses['pred'] # 使用预测损失进行评估比较
    metrics['loss_total'] = epoch_losses['total']

    return metrics, targets_np, preds_np, factors_by_type # 返回因子用于分析

# --------------------------------------------
#            主函数
# --------------------------------------------
def main():
    args = parse_args()

    # --- 设置 ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_id != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.manual_seed_all(args.seed)
        print(f"使用 GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_root, f"{args.exp_name}_{timestamp}")
    model_save_dir = os.path.join(results_dir, 'Model')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    log_file = os.path.join(results_dir, 'run.log')
    setup_logging(log_file)
    logging.info("CDN V2 实验开始")
    logging.info(f"参数: {vars(args)}")
    logging.info(f"设备: {device}")
    logging.info(f"结果目录: {results_dir}")

    # --- 数据加载与分割 ---
    logging.info("加载数据中...")
    try:
        full_train_dataset = CdnDatasetV2(args.data_dir, split='train')
        test_dataset = CdnDatasetV2(args.data_dir, split='test')

        # 从训练数据创建验证集
        train_indices, val_indices = train_test_split(
            np.arange(len(full_train_dataset)),
            test_size=args.val_split,
            random_state=args.seed,
        )
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        logging.info(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

        pin_memory = True if device.type == 'cuda' else False
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        logging.info("DataLoader 已准备就绪。")
    except Exception as e:
         logging.error(f"数据加载/分割失败: {e}", exc_info=True)
         return

    # --- 模型, 损失函数, 优化器 ---
    logging.info("初始化模型, 损失函数, 优化器...")
    try:
        # 从加载的数据确定 context_dim
        context_dim_data = full_train_dataset.X_context.shape[1] if hasattr(full_train_dataset, 'X_context') else 8
        logging.info(f"使用的上下文维度: {context_dim_data}")

        model = CausalDisentanglementNetworkAdaptedV2(
            context_dim=context_dim_data,
            embed_dim=args.embed_dim, patch_size=args.patch_size, patch_stride=args.patch_stride,
            transformer_layers=args.tfr_layers, transformer_heads=args.tfr_heads,
            num_factors=args.num_factors, factor_dims=args.factor_dims,
            condition_dim=args.condition_dim, glucose_factor_index=args.glucose_idx
        ).to(device)

        criterion = CDNLossAdapted(
            weight_pred=args.loss_pred_w, weight_recon=args.loss_recon_w, weight_disentangle=args.loss_dis_w,
            num_factors=args.num_factors, factor_dims=args.factor_dims
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        logging.info(f"模型:\n{model}")
        logging.info(f"损失函数: {criterion}")
        logging.info(f"优化器: {optimizer}")
    except Exception as e:
        logging.error(f"模型/损失函数/优化器初始化失败: {e}", exc_info=True)
        return


    # --- 训练循环 ---
    logging.info(f"开始训练 {args.epochs} 轮...")
    best_val_metric = float('inf') # 使用验证集预测损失或 RMSE
    history = {'train_loss_total': [], 'train_loss_pred': [], 'train_loss_recon': [], 'train_loss_disentangle': [],
               'val_loss_total': [], 'val_loss_pred': [], 'val_rmse': [], 'val_mae': []}
    model_save_path = os.path.join(model_save_dir, 'best_model.pth')

    for epoch in range(args.epochs):
        start_time = time.time()
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        end_time = time.time()

        # 存储历史记录
        history['train_loss_total'].append(train_losses['total'])
        history['train_loss_pred'].append(train_losses['pred'])
        history['train_loss_recon'].append(train_losses['recon'])
        history['train_loss_disentangle'].append(train_losses['disentangle'])
        history['val_loss_total'].append(val_metrics['loss_total'])
        history['val_loss_pred'].append(val_metrics['loss_pred'])
        history['val_rmse'].append(val_metrics['RMSE'])
        history['val_mae'].append(val_metrics['MAE'])

        logging.info(f"轮次 {epoch+1}/{args.epochs} | 时间: {end_time - start_time:.1f}s | "
                     f"训练损失: {train_losses['total']:.4f} (P:{train_losses['pred']:.4f}, R:{train_losses['recon']:.4f}, D:{train_losses['disentangle']:.4f}) | "
                     f"验证损失: {val_metrics['loss_total']:.4f} (P:{val_metrics['loss_pred']:.4f}) | "
                     f"验证 RMSE: {val_metrics['RMSE']:.4f}")

        # 根据验证集预测损失保存最佳模型
        current_val_metric = val_metrics['loss_pred']
        if current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            try:
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"  >> 最佳模型已保存至 {model_save_path} (验证集预测损失: {best_val_metric:.4f})")
            except Exception as e: logging.error(f"  保存模型时出错: {e}")

    logging.info("训练完成。")

    # --- 在测试集上进行最终评估 ---
    logging.info("加载最佳模型进行最终测试评估...")
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info("最佳模型已加载。")
    except Exception as e:
        logging.error(f"从 {model_save_path} 加载最佳模型失败: {e}。将使用最后一个模型状态进行评估。")

    logging.info("在测试集上评估...")
    test_metrics, test_targets, test_preds, test_factors = evaluate(model, test_loader, criterion, device)

    if test_targets is not None and test_preds is not None and len(test_targets) > 0:
         logging.info(f"测试结果: Pred Loss={test_metrics['loss_pred']:.4f} | RMSE={test_metrics['RMSE']:.4f} | MAE={test_metrics['MAE']:.4f} | MAPE={test_metrics['MAPE']:.2f}% | R2={test_metrics['R2']:.4f}")

         # --- 单位转换 和 CEG 绘图 (使用 MATLAB 样式) ---
         logging.info("准备绘制克拉克误差网格图 (MATLAB 样式)...")
         # 假设 test_targets 和 test_preds 是 mmol/L 单位，转换为 mg/dL
         conversion_factor = 18.018 # 或者使用 18
         try:
             test_targets_mgdl = test_targets * conversion_factor
             test_preds_mgdl = test_preds * conversion_factor
             logging.info("已将测试结果从 mmol/L 转换为 mg/dL 用于 CEG 绘图。")

             # 调用 MATLAB 风格的 CEG 函数 (需要确保已从 utils 导入)
             ceg_total_counts, ceg_percentage_dist = plot_clarke_ega_matlab_style(
                 test_targets_mgdl, # 传递 mg/dL 数据
                 test_preds_mgdl,   # 传递 mg/dL 数据
                 title_str=f'',  # 标题
                 save_path=os.path.join(results_dir, 'clarke_ega_matlab_style.png') # 新的文件名
             )
             # 注意: CEG 的结果现在由 plot 函数打印，如果需要可以记录 total_counts 和 percentage_dist
         except NameError:
              logging.error("函数 'plot_clarke_ega_matlab_style' 未在 utils_cdn_v2.py 中定义或导入。")
         except Exception as e:
              logging.error(f"生成 MATLAB 样式 CEG 图时失败: {e}", exc_info=True)


         # --- 保存结果和绘制其他图表 ---
         logging.info("保存结果并生成其他图表...")
         # 注意: save_results 不再接收 ceg_zones_perc，因为绘图函数内部处理了统计和打印
         save_results(results_dir, args, test_metrics, test_targets, test_preds, history)

         # 绘制散点图 (假设原始单位是 mmol/L 用于标签)
         plot_scatter(test_targets, test_preds, test_metrics, unit='mmol/L', save_path=os.path.join(results_dir, 'scatter_plot.png'))

         # 绘制损失曲线
         plot_loss_curves(history, save_path=os.path.join(results_dir, 'loss_curves.png'))

         # 绘制因子相关性热力图
         plot_factor_correlation_heatmap(test_factors, save_path=os.path.join(results_dir, 'factor_correlation_heatmap.png'))

         logging.info("图表生成完成。")
    else:
         logging.error("测试评估失败或无有效测试数据。无法保存结果或绘图。")

    logging.info(f"实验完成。结果保存在: {results_dir}")


if __name__ == "__main__":
    # 检查数据预处理是否完成
    args_temp = parse_args() # 解析一次以获取 data_dir
    if not os.path.exists(args_temp.data_dir) or not os.listdir(args_temp.data_dir):
         print(f"错误: 处理后的数据目录 '{args_temp.data_dir}' 为空或不存在。")
         print("请先运行 'python preprocessed_cdn_v2.py' 或类似脚本进行数据预处理。") # 提示信息更新
    else:
         main()
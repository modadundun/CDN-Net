# 文件名: preprocess_cdn_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
import torch
def preprocess_for_cdn_v2(file_path, output_dir, test_size=0.2, random_state=42):
    """
    为 CDN 模型加载、预处理、划分数据 (V2: 包含 Vitals)。
    输入列: 血糖值, Gender, Height, Weight, Age, HeartRate, SpO2, BPSys, BPDia, PPG 1..64
    输出: X_ppg, X_context, y (mmol/L)
    """
    print(f"--- Running CDN Preprocessing V2 ---")
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path, sep='\t')
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        print(f"Original columns found: {data.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # --- 1. 定义并检查所需列 ---
    target_col = '血糖值'
    demographic_cols = ['Gender', 'Height', 'Weight', 'Age']
    vital_cols = ['HeartRate', 'SpO2', 'BPSys', 'BPDia'] # 新增生命体征列
    context_cols = demographic_cols + vital_cols # 所有非PPG特征
    ppg_cols = [f'PPG_{i}' for i in range(1, 65)]
    alt_ppg_cols = [f'PPG {i}' for i in range(1, 65)]

    # 检查 PPG
    if all(col in data.columns for col in ppg_cols):
        print("Using 'PPG_X' columns.")
    elif all(col in data.columns for col in alt_ppg_cols):
        print("Using 'PPG X' columns.")
        ppg_cols = alt_ppg_cols
    else:
        print(f"Error: Cannot find all 64 PPG columns.")
        return False

    all_needed_cols = [target_col] + context_cols + ppg_cols
    missing_cols = [col for col in all_needed_cols if col not in data.columns]
    if missing_cols:
        # **关键警告**
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: Missing required columns: {missing_cols}")
        print("Please ensure your data file contains ALL required columns:")
        print(f"  Target: {target_col}")
        print(f"  Demographics: {demographic_cols}")
        print(f"  Vitals: {vital_cols}")
        print(f"  PPG: PPG 1 to PPG 64")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return False

    # --- 2. 类型转换和提取 ---
    try:
        target = data[target_col].astype(np.float32).values
        print(f"Target '{target_col}' loaded (mmol/L assumed). Shape: {target.shape}")
        if np.max(target) > 30: # mmol/L 不应远超30
             print(f"Warning: Max glucose value ({np.max(target):.1f}) seems high for mmol/L. Please double-check units.")

        context_features_df = data[context_cols].copy()
        print(f"Context features loaded. Shape: {context_features_df.shape}")

        ppg_features = data[ppg_cols].astype(np.float32).values
        print(f"PPG features loaded. Shape: {ppg_features.shape}")
    except ValueError as e:
        print(f"Error converting columns to numeric: {e}. Check data.")
        return False
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return False

    # --- 3. 预处理上下文特征 (X_context) ---
    # 处理缺失值 (例如，用均值填充)
    if context_features_df.isnull().values.any():
        print("Warning: Missing values found in context features. Filling with mean.")
        for col in context_features_df.columns:
             if context_features_df[col].isnull().any():
                  if pd.api.types.is_numeric_dtype(context_features_df[col]):
                      mean_val = context_features_df[col].mean()
                      context_features_df[col].fillna(mean_val, inplace=True)
                      print(f"  Filled NaNs in '{col}' with mean ({mean_val:.2f})")
                  else: # 非数值列尝试用 'unknown' 或最常见值
                      mode_val = context_features_df[col].mode()[0]
                      context_features_df[col].fillna(mode_val, inplace=True)
                      print(f"  Filled NaNs in '{col}' with mode ('{mode_val}')")


    # (a) 性别编码
    le = LabelEncoder()
    gender_col = 'Gender'
    try:
        if pd.api.types.is_string_dtype(context_features_df[gender_col]) or pd.api.types.is_object_dtype(context_features_df[gender_col]):
            context_features_df.loc[:, gender_col] = le.fit_transform(context_features_df[gender_col])
            print(f"Gender encoded. Classes: {le.classes_}")
        elif not pd.api.types.is_numeric_dtype(context_features_df[gender_col]):
             print(f"Warning: Gender column type is {context_features_df[gender_col].dtype}. Attempting encoding.")
             context_features_df.loc[:, gender_col] = le.fit_transform(context_features_df[gender_col])
    except Exception as e:
        print(f"Warning: Could not robustly encode Gender: {e}.")
        # 尝试强制转为数值，失败则报错
        try:
            context_features_df[gender_col] = pd.to_numeric(context_features_df[gender_col])
        except ValueError:
             print(f"Error: Failed to process Gender column.")
             return False

    # (b) 确保所有 Context 列为数值
    for col in context_features_df.columns:
         if not pd.api.types.is_numeric_dtype(context_features_df[col]):
             try:
                 context_features_df[col] = pd.to_numeric(context_features_df[col])
             except ValueError:
                 print(f"Error: Context feature '{col}' has non-numeric values and cannot be converted.")
                 return False

    # (c) 标准化所有 Context 特征
    context_scaler = StandardScaler()
    context_features_scaled = context_scaler.fit_transform(context_features_df).astype(np.float32)
    print(f"Context features scaled. Shape: {context_features_scaled.shape}") # (n_samples, 8)

    # --- 4. 预处理 PPG 特征 (X_ppg) ---
    ppg_mean = np.mean(ppg_features, axis=1, keepdims=True)
    ppg_std = np.std(ppg_features, axis=1, keepdims=True)
    ppg_features_scaled = np.divide(ppg_features - ppg_mean, ppg_std,
                                    out=np.zeros_like(ppg_features, dtype=np.float32),
                                    where=ppg_std != 0)
    ppg_features_final = np.expand_dims(ppg_features_scaled, axis=1).astype(np.float32) # (n_samples, 1, 64)
    print(f"PPG features scaled and reshaped. Shape: {ppg_features_final.shape}")

    # --- 5. 数据集划分 ---
    stratify_opt = None
    if len(np.unique(target)) > 5:
        try:
            stratify_opt = pd.cut(target, bins=10, labels=False, duplicates='drop')
            print("Using stratified split based on glucose levels.")
        except Exception as e:
            print(f"Warning: Could not create strata ({e}). Using random split.")

    indices = np.arange(len(target))
    try:
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify_opt)
    except ValueError as e:
         print(f"Warning: Stratified split failed ({e}). Using random split.")
         train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)


    X_ppg_train, X_ppg_test = ppg_features_final[train_indices], ppg_features_final[test_indices]
    X_context_train, X_context_test = context_features_scaled[train_indices], context_features_scaled[test_indices]
    y_train, y_test = target[train_indices], target[test_indices]

    print(f"Data split into Train ({len(y_train)} samples) and Test ({len(y_test)} samples).")

    # --- 6. 保存处理后的数据 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed data to '{output_dir}'...")
    try:
        np.save(os.path.join(output_dir, 'X_ppg_train.npy'), X_ppg_train)
        np.save(os.path.join(output_dir, 'X_context_train.npy'), X_context_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_ppg_test.npy'), X_ppg_test)
        np.save(os.path.join(output_dir, 'X_context_test.npy'), X_context_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        # 保存 scaler
        joblib.dump(context_scaler, os.path.join(output_dir, 'context_scaler.joblib'))
        if 'le' in locals() and hasattr(le, 'classes_'): # 只有当 le 创建并使用后才保存
             joblib.dump(le, os.path.join(output_dir, 'gender_encoder.joblib'))
        print("Preprocessing and saving complete.")
        return True
    except Exception as e:
        print(f"Error saving files: {e}")
        return False

# --- 使用示例 ---
if __name__ == "__main__":
    input_file = 'processed_output_十进制.txt'
    output_directory = './processed_cdn_data_v2' # 新目录
    success = preprocess_for_cdn_v2(input_file, output_directory)
    if success:
        print(f"\nCDN V2 preprocessing successful. Data saved in '{output_directory}'")
    else:
        print("\nCDN V2 preprocessing failed. Please check errors and data file.")
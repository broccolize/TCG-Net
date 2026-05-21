#!/usr/bin/env python3
"""
UK Biobank 个体化因子评分计算器 (V3)

新版方法：
1. 厚度先对采集/站点（Centre）、性别（Sex）、ICV 等协变量残差化
2. 再对每一个脑区全样本 z-score 标准化
3. 使用加权求和（点积）计算 risk score

注意：ICV 从 UKB_BasicMRI.csv 的 Vol_WB_TIV_2_0 列读取

改进点：
- 先控制协变量（残差化），去除采集/站点（Centre）、性别（Sex）、ICV 的影响
- ICV 从 UKB_BasicMRI.csv 的 Vol_WB_TIV_2_0 列读取
- 再对残差进行全样本 z-score 标准化
- 最后计算 risk score
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 添加code目录到路径，以便导入config
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

print("=" * 70)
print("UK Biobank 个体化因子评分计算器 (V3)")
print("方法：协变量残差化（Centre/Sex/ICV） + 全样本 Z-score + 加权求和")
print("=" * 70)

# 输出目录
OUTPUT_DIR = Path("results_v3/01_ukb_risk_scores")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")

# ==============================================================================
# 1. 加载因子模板 (Weights)
# ==============================================================================
print("\n[步骤 1/6] 加载因子模板...")

# 优先从 results_v2 文件夹的第一步分析结果加载
factor_scores_paths = [
    Path('results_v2/01_factor_analysis/factor_scores.csv'),  # 第一步：因子分析结果（优先）
    Path('results_v2/02_disease_specific/factor_scores.csv'),  # 第二步：疾病特异性分析结果
    Path('factor_scores.csv')  # 根目录（备用）
]

factor_scores = None
factor_scores_source = None

for path in factor_scores_paths:
    if path.exists():
        try:
            factor_scores = pd.read_csv(path)
            factor_scores_source = str(path)
            print(f"  ✓ 从 {path} 加载因子得分数据")
            break
        except Exception as e:
            print(f"  ⚠ 警告: 无法读取 {path}: {e}")
            continue

if factor_scores is None:
    print("  ✗ 未找到 factor_scores.csv")
    print("    请检查以下位置:")
    for path in factor_scores_paths:
        print(f"      - {path}")
    exit()

# 处理因子得分数据
if 'Unnamed: 0' in factor_scores.columns:
    factor_scores = factor_scores.drop(columns=['Unnamed: 0'])

factor_cols = [c for c in factor_scores.columns if 'Factor' in c]
factor_scores = factor_scores[factor_cols]

# 确保只有68个脑区
if len(factor_scores) > 68:
    factor_scores = factor_scores.iloc[:68]
    print(f"  ⚠ 注意: 截取前68行")
elif len(factor_scores) < 68:
    print(f"  ⚠ 警告: 因子数据只有 {len(factor_scores)} 行，少于68个脑区")

print(f"  ✓ 因子模板: {factor_scores.shape} (脑区数: {len(factor_scores)}, 因子数: {len(factor_cols)})")
print(f"     因子: {factor_cols}")
print(f"     数据来源: {factor_scores_source}")

# ==============================================================================
# 2. 加载 UKB 个体数据
# ==============================================================================
print("\n[步骤 2/6] 加载 UK Biobank 数据...")

ukb_file = Path('UKB_FsDKT(2).csv')
if not ukb_file.exists():
    print(f"  ✗ 未找到 UKB 数据文件: {ukb_file}")
    exit()

print(f"  ✓ 正在读取: {ukb_file}")
ukb_data = pd.read_csv(ukb_file)
print(f"  ✓ 数据加载完成: {len(ukb_data)} 个个体, {len(ukb_data.columns)} 列")

# 提取所有CT列（皮层厚度，_2_0表示第二次扫描）
ct_cols = [c for c in ukb_data.columns if 'CT' in c and '_2_0' in c]
print(f"  ✓ 找到 {len(ct_cols)} 个CT列（皮层厚度）")

if len(ct_cols) != 68:
    print(f"  ⚠ 警告: CT列数 ({len(ct_cols)}) 不等于68，可能影响匹配")

# 显示前几个CT列名
print(f"  前5个CT列: {ct_cols[:5]}")

# ==============================================================================
# 3. 创建脑区名称映射（根据名称匹配）
# ==============================================================================
print("\n[步骤 3/6] 创建脑区名称映射（根据名称匹配）...")

# 标准DK68脑区名称（与factor_scores.csv的行顺序对应）
dk68_standard_names = [
    # 左半球 (1-34)
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal',
    'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital',
    'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal',
    'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
    'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal',
    'frontalpole', 'temporalpole', 'transversetemporal', 'insula',
    # 右半球 (35-68)
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal',
    'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital',
    'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal',
    'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
    'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal',
    'frontalpole', 'temporalpole', 'transversetemporal', 'insula'
]

# UKB列名缩写到标准DK68名称的映射
ukb_abbrev_to_region = {
    'BST': 'bankssts',
    'CACg': 'caudalanteriorcingulate',
    'CMF': 'caudalmiddlefrontal',
    'Cu': 'cuneus',
    'En': 'entorhinal',
    'Fp': 'frontalpole',
    'Fu': 'fusiform',
    'IP': 'inferiorparietal',
    'IT': 'inferiortemporal',
    'Ins': 'insula',
    'IstCg': 'isthmuscingulate',
    'LO': 'lateraloccipital',
    'LOrF': 'lateralorbitofrontal',
    'Lg': 'lingual',
    'MOrF': 'medialorbitofrontal',
    'MT': 'middletemporal',
    'Op': 'parsopercularis',
    'Or': 'parsorbitalis',
    'Tr': 'parstriangularis',
    'PaC': 'paracentral',
    'PaH': 'parahippocampal',
    'PerCa': 'pericalcarine',
    'PoC': 'postcentral',
    'PoCg': 'posteriorcingulate',
    'PreC': 'precentral',
    'PreCu': 'precuneus',
    'RoACg': 'rostralanteriorcingulate',
    'RoMF': 'rostralmiddlefrontal',
    'SF': 'superiorfrontal',
    'SP': 'superiorparietal',
    'ST': 'superiortemporal',
    'SM': 'supramarginal',
    'TrT': 'temporalpole',
}

def extract_region_from_ukb_col(col_name):
    """从UKB列名提取脑区信息"""
    parts = col_name.replace('CT_', '').replace('_2_0', '').split('_')
    if len(parts) >= 2:
        hemisphere = parts[0]  # L or R
        abbrev = parts[1]  # BST, CACg, 等
        region_name = ukb_abbrev_to_region.get(abbrev, None)
        return hemisphere, region_name, abbrev
    return None, None, None

# 创建映射：标准DK68顺序索引 -> UKB列名
ordered_ukb_cols = [None] * len(dk68_standard_names)
matched_regions = []
unmatched_regions = []

for i, std_region in enumerate(dk68_standard_names):
    # 确定半球
    if i < 34:
        hem = 'L'
    else:
        hem = 'R'
    
    # 在UKB列中查找匹配的列
    for ukb_col in ct_cols:
        if 'Total' in ukb_col:
            continue
        
        hem_ukb, region_ukb, abbrev_ukb = extract_region_from_ukb_col(ukb_col)
        
        if hem_ukb == hem and region_ukb == std_region:
            ordered_ukb_cols[i] = ukb_col
            matched_regions.append((i, std_region, ukb_col))
            break
    
    if ordered_ukb_cols[i] is None:
        unmatched_regions.append((i, std_region, hem))

# 过滤掉None值，得到实际匹配的列
ordered_ukb_cols = [c for c in ordered_ukb_cols if c is not None]

print(f"  ✓ 成功匹配 {len(matched_regions)}/{len(dk68_standard_names)} 个脑区")
print(f"     左半球匹配: {len([m for m in matched_regions if m[0] < 34])} 个")
print(f"     右半球匹配: {len([m for m in matched_regions if m[0] >= 34])} 个")

if unmatched_regions:
    print(f"  ⚠ 未匹配的脑区 ({len(unmatched_regions)} 个):")
    for idx, region, hem in unmatched_regions[:10]:
        print(f"     索引 {idx}: {hem}_{region}")
    if len(unmatched_regions) > 10:
        print(f"     ... 还有 {len(unmatched_regions) - 10} 个未匹配")

# 创建因子得分到UKB列的映射索引
factor_indices = [i for i, c in enumerate(ordered_ukb_cols) if c is not None]
matched_factor_indices = [m[0] for m in matched_regions]

print(f"  ✓ 最终使用 {len(ordered_ukb_cols)} 个UKB列进行匹配")

# ==============================================================================
# 4. 加载协变量数据
# ==============================================================================
print("\n[步骤 4/6] 加载协变量数据...")

# 加载基础 MRI 数据（包含年龄、性别等）
basic_mri_file = Path('UKB_BasicMRI.csv')
covariates_data = None
covariate_cols = []

if basic_mri_file.exists():
    try:
        print(f"  加载协变量数据用于残差化...")
        covariate_data_full = pd.read_csv(basic_mri_file, low_memory=False)
        
        # 提取需要的协变量：只包括 Centre（采集/站点）、Sex（性别）、ICV
        required_covars = ['Centre_0_0', 'Sex_0_0']
        
        # 查找Sex列
        sex_col = None
        for possible_sex_col in ['Sex_0_0', 'Sex_2_0', 'Sex', 'sex', 'Sex_1_0']:
            if possible_sex_col in covariate_data_full.columns:
                sex_col = possible_sex_col
                break
        
        if sex_col:
            if 'Sex_0_0' in required_covars:
                required_covars = [c if c != 'Sex_0_0' else sex_col for c in required_covars]
            print(f"    ✓ 找到Sex列: {sex_col}")
        else:
            print(f"    ⚠ 警告: 未找到Sex列，将跳过sex协变量")
            required_covars = [c for c in required_covars if c != 'Sex_0_0']
        
        # 检查 ICV 列（Vol_WB_TIV_2_0）
        icv_col = 'Vol_WB_TIV_2_0'
        if icv_col in covariate_data_full.columns:
            required_covars.append(icv_col)
            print(f"    ✓ 找到ICV列: {icv_col}")
        else:
            print(f"    ⚠ 警告: 未找到ICV列 ({icv_col})，将跳过ICV协变量")
        
        # 检查必需的协变量
        missing_covars = [c for c in required_covars if c not in covariate_data_full.columns]
        if missing_covars:
            print(f"    ⚠ 警告: 缺少协变量: {missing_covars}，将跳过这些协变量")
            required_covars = [c for c in required_covars if c not in missing_covars]
        
        # 提取协变量数据
        covar_data = covariate_data_full[['eid'] + required_covars].copy()
        
        # 重命名 ICV 列为 'ICV'（如果存在）
        if icv_col in covar_data.columns:
            covar_data = covar_data.rename(columns={icv_col: 'ICV'})
            # 更新 required_covars 中的名称
            required_covars = [c if c != icv_col else 'ICV' for c in required_covars]
        
        # 处理Centre缺失值
        if 'Centre_0_0' in covar_data.columns:
            if covar_data['Centre_0_0'].isna().any():
                n_missing = covar_data['Centre_0_0'].isna().sum()
                if covar_data['Centre_0_0'].notna().any():
                    most_common = covar_data['Centre_0_0'].mode()[0] if len(covar_data['Centre_0_0'].mode()) > 0 else 0
                    covar_data['Centre_0_0'] = covar_data['Centre_0_0'].fillna(most_common)
                else:
                    covar_data['Centre_0_0'] = covar_data['Centre_0_0'].fillna('Missing')
        
        # 将Centre转化为one-hot编码
        if 'Centre_0_0' in covar_data.columns:
            centre_onehot = pd.get_dummies(covar_data['Centre_0_0'], prefix='Centre', drop_first=True)
            covar_data = pd.concat([covar_data.drop(columns=['Centre_0_0']), centre_onehot], axis=1)
        
        # 确定协变量列（不包括eid）
        covariate_cols = [c for c in covar_data.columns if c != 'eid']
        covariates_data = covar_data
        print(f"    ✓ 协变量列: {len(covariate_cols)} 个")
        print(f"      使用的协变量: {', '.join(covariate_cols[:10])}... (共{len(covariate_cols)}个)")
        if 'ICV' in covariate_cols:
            print(f"      ICV 来源: UKB_BasicMRI.csv 的 Vol_WB_TIV_2_0 列")
        
    except Exception as e:
        print(f"  ⚠ 加载协变量数据失败: {e}")

if covariates_data is None or len(covariate_cols) == 0:
    print("  ⚠ 警告: 未找到协变量数据，将跳过残差化步骤")
    print("    将直接进行全样本 Z-score 标准化")

# ==============================================================================
# 5. 协变量残差化 + 全样本 Z-score 标准化
# ==============================================================================
print("\n[步骤 5/6] 协变量残差化 + 全样本 Z-score 标准化...")

# 提取CT数据
ukb_ct_data = ukb_data[ordered_ukb_cols].copy()

# 确定eid列
eid_col_ukb = 'eid' if 'eid' in ukb_data.columns else ukb_data.columns[0]
ukb_eid = ukb_data[eid_col_ukb].values

# 处理缺失值：如果某个个体的某个脑区数据缺失，使用该脑区的均值填充
missing_count = 0
for col in ordered_ukb_cols:
    if ukb_ct_data[col].isna().any():
        n_missing = ukb_ct_data[col].isna().sum()
        if n_missing > 0:
            mean_val = ukb_ct_data[col].mean()
            ukb_ct_data[col] = ukb_ct_data[col].fillna(mean_val)
            missing_count += n_missing

if missing_count > 0:
    print(f"  ⚠ 总共填充了 {missing_count} 个缺失值")
else:
    print(f"  ✓ 无缺失值")

# 合并协变量数据
if covariates_data is not None and len(covariate_cols) > 0:
    # 合并数据
    merged_data = pd.DataFrame({
        'eid': ukb_eid
    })
    merged_data = merged_data.merge(covariates_data, on='eid', how='left')
    
    # 准备协变量矩阵（处理缺失值）
    X_covars = merged_data[covariate_cols].copy()
    for col in covariate_cols:
        if X_covars[col].isna().any():
            if X_covars[col].dtype in ['object', 'string']:
                # 分类变量：使用众数填充
                mode_val = X_covars[col].mode()[0] if len(X_covars[col].mode()) > 0 else 0
                X_covars[col] = X_covars[col].fillna(mode_val)
            else:
                # 数值变量：使用均值填充
                mean_val = X_covars[col].mean()
                X_covars[col] = X_covars[col].fillna(mean_val)
    
    X_covars = X_covars.astype(float).values
    
    print(f"  ✓ 协变量数据已准备: {X_covars.shape}")
    print(f"     开始对每个脑区进行残差化...")
    
    # 对每个脑区进行残差化
    ct_residuals = ukb_ct_data.copy()
    
    for i, col in enumerate(ordered_ukb_cols):
        if (i + 1) % 10 == 0:
            print(f"     处理脑区 {i+1}/{len(ordered_ukb_cols)}...", end='\r')
        
        y_ct = ukb_ct_data[col].values
        
        # 线性回归：CT ~ 协变量
        reg = LinearRegression()
        reg.fit(X_covars, y_ct)
        y_pred = reg.predict(X_covars)
        
        # 计算残差
        residuals = y_ct - y_pred
        ct_residuals[col] = residuals
    
    print(f"  ✓ 残差化完成: {len(ordered_ukb_cols)} 个脑区")
    
    # 使用残差进行后续分析
    ct_data_for_zscore = ct_residuals
else:
    # 如果没有协变量，直接使用原始数据
    print(f"  ⚠ 未进行残差化（无协变量数据）")
    ct_data_for_zscore = ukb_ct_data

# 步骤 B: 全样本 Z-score 标准化 (Column-wise Standardization)
print(f"\n  正在进行全样本 Z-score 标准化 (Column-wise Z-score)...")

# 计算全样本的均值和标准差（对每一列/每个脑区）
global_mean = ct_data_for_zscore.mean(axis=0)  # 每个脑区的全样本均值
global_std = ct_data_for_zscore.std(axis=0)   # 每个脑区的全样本标准差

print(f"  ✓ 全样本统计量计算完成")
print(f"     均值范围: [{global_mean.min():.3f}, {global_mean.max():.3f}]")
print(f"     标准差范围: [{global_std.min():.3f}, {global_std.max():.3f}]")

# 计算 Z-score (Column-wise Standardization)
# Z = (Value - Global_Mean) / Global_Std
z_data = (ct_data_for_zscore - global_mean) / global_std

# 处理可能的 NaN (如某脑区标准差为0或缺失)
z_data = z_data.fillna(0)

print(f"  ✓ 全样本Z-score标准化完成")
print(f"     数据形状: {z_data.shape} (个体数 x 脑区数)")
print(f"     Z-score范围: [{z_data.min().min():.3f}, {z_data.max().max():.3f}]")

# ==============================================================================
# 6. 加权求和步骤
# ==============================================================================
print("\n[步骤 6/6] 加权求和（点积）计算 Risk Score...")

# 准备因子模板矩阵（权重）
# 根据名称匹配的结果，只使用匹配的脑区
if 'matched_factor_indices' in locals() and len(matched_factor_indices) > 0:
    # 使用匹配的因子得分行
    weights_df = factor_scores.iloc[matched_factor_indices][factor_cols]
    n_regions_used = len(matched_factor_indices)
    print(f"  ✓ 使用名称匹配的 {n_regions_used} 个脑区")
else:
    # 如果匹配失败，回退到顺序匹配
    n_regions_used = min(len(ordered_ukb_cols), len(factor_scores))
    weights_df = factor_scores.iloc[:n_regions_used][factor_cols]
    print(f"  ⚠ 回退到顺序匹配: 使用前 {n_regions_used} 行")

if n_regions_used < len(factor_scores):
    print(f"  ⚠ 注意: 因子模板有 {len(factor_scores)} 行，但只使用 {n_regions_used} 行（匹配UKB列数）")

print(f"  使用 {n_regions_used} 个脑区计算评分")
print(f"  处理 {len(ukb_data)} 个个体...")

prs_results = pd.DataFrame(index=ukb_data.index)

for factor in factor_cols:
    w = weights_df[factor].values  # 权重向量 (n_regions_used,)
    
    # 确保权重和数据的维度匹配
    if len(w) != len(ordered_ukb_cols):
        print(f"  ⚠ 警告: {factor} 的权重维度 ({len(w)}) 与数据维度 ({len(ordered_ukb_cols)}) 不匹配")
        # 尝试截断或填充
        if len(w) > len(ordered_ukb_cols):
            w = w[:len(ordered_ukb_cols)]
        else:
            # 如果权重更少，只使用前 len(w) 个脑区
            z_data_subset = z_data.iloc[:, :len(w)]
            score = np.dot(z_data_subset.values, w)
            prs_results[f'{factor}_V3_Score'] = score / len(w)
            continue
    
    # 计算点积 (加权求和)
    score = np.dot(z_data.values, w)
    
    # 归一化（除以脑区数，使分数在 Z-score 量级）
    prs_results[f'{factor}_V3_Score'] = score / n_regions_used
    
    if len(prs_results) > 0:
        print(f"  ✓ {factor}: 评分范围 [{prs_results[f'{factor}_V3_Score'].min():.3f}, {prs_results[f'{factor}_V3_Score'].max():.3f}]")

# 合并结果
ukb_eid_col = 'eid' if 'eid' in ukb_data.columns else ukb_data.columns[0]
ukb_results = pd.concat([ukb_data[[ukb_eid_col]], prs_results], axis=1)

print(f"  ✓ 基于协变量残差化 + 全样本标准化的 Risk Score 计算完成！")

# ==============================================================================
# 7. 保存结果和可视化
# ==============================================================================
print("\n[步骤 7/7] 保存结果和可视化...")

# 保存完整结果
output_file = OUTPUT_DIR / 'ukb_individual_factor_scores_v3.csv'
ukb_results.to_csv(output_file, index=False)
print(f"  ✓ 结果已保存: {output_file}")
print(f"     个体数: {len(ukb_results)}")
score_cols = [f'{f}_V3_Score' for f in factor_cols]
print(f"     因子评分列: {score_cols}")

# 保存统计摘要
summary_stats = ukb_results[score_cols].describe()
summary_file = OUTPUT_DIR / 'ukb_factor_scores_v3_summary.csv'
summary_stats.to_csv(summary_file)
print(f"  ✓ 统计摘要已保存: {summary_file}")

# 保存处理信息
processing_info = pd.DataFrame({
    'Step': [
        'Covariate_Residualization',
        'Global_Z_Score_Standardization',
        'Weighted_Sum',
        'N_Regions_Used',
        'N_Covariates',
        'Covariates_Used'
    ],
    'Value': [
        'Yes' if (covariates_data is not None and len(covariate_cols) > 0) else 'No',
        'Yes',
        'Yes',
        n_regions_used,
        len(covariate_cols) if covariates_data is not None else 0,
        ', '.join(covariate_cols) if covariates_data is not None and len(covariate_cols) > 0 else 'None'
    ]
})
processing_info.to_csv(OUTPUT_DIR / 'ukb_risk_score_v3_processing_info.csv', index=False)
print(f"  ✓ 处理信息已保存: {OUTPUT_DIR / 'ukb_risk_score_v3_processing_info.csv'}")

# 可视化：因子评分的分布
fig, axes = plt.subplots(1, len(factor_cols), figsize=(6 * len(factor_cols), 5))
if len(factor_cols) == 1:
    axes = [axes]

for i, factor in enumerate(factor_cols):
    score_col = factor + '_V3_Score'
    axes[i].hist(ukb_results[score_col], bins=50, alpha=0.7, edgecolor='black')
    axes[i].axvline(ukb_results[score_col].mean(), color='red', linestyle='--', 
                    label=f'Mean: {ukb_results[score_col].mean():.3f}')
    axes[i].set_xlabel(f'{factor} Score (V3: Residualized + Global Z-score)', fontsize=11)
    axes[i].set_ylabel('Frequency', fontsize=11)
    axes[i].set_title(f'Distribution of {factor} Scores (V3)\n(n={len(ukb_results)})', fontsize=12)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ukb_factor_scores_v3_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 分布图已保存: {OUTPUT_DIR / 'ukb_factor_scores_v3_distribution.png'}")

# 可视化：因子评分之间的相关性
if len(factor_cols) > 1:
    corr_matrix = ukb_results[score_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation between Factor Scores (V3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ukb_factor_scores_v3_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 相关性图已保存: {OUTPUT_DIR / 'ukb_factor_scores_v3_correlation.png'}")

# 显示前10个结果
print("\n前10个个体的因子评分:")
print(ukb_results.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("✅ UK Biobank 个体化因子评分计算完成 (V3)!")
print("=" * 70)
print(f"\n方法说明:")
print(f"  1. 协变量残差化: {'是' if (covariates_data is not None and len(covariate_cols) > 0) else '否'}")
if covariates_data is not None and len(covariate_cols) > 0:
    print(f"     协变量: {', '.join(covariate_cols[:5])}... (共{len(covariate_cols)}个)")
print(f"  2. 全样本 Z-score 标准化: 是")
print(f"  3. 加权求和: 是")
print(f"\n输出文件:")
print(f"  - 完整结果: {output_file}")
print(f"  - 统计摘要: {summary_file}")
print(f"  - 处理信息: {OUTPUT_DIR / 'ukb_risk_score_v3_processing_info.csv'}")
print(f"  - 分布图: {OUTPUT_DIR / 'ukb_factor_scores_v3_distribution.png'}")
if len(factor_cols) > 1:
    print(f"  - 相关性图: {OUTPUT_DIR / 'ukb_factor_scores_v3_correlation.png'}")

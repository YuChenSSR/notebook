from multiprocessing import Pool, cpu_count
import pickle
import numpy as np
import pandas as pd
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from typing import Union, Dict, Any, Optional
from scipy.stats import spearmanr, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


# 2026-01-06 新增: 配置参数类，提升阈值可配置性
@dataclass
class FactorScopeConfig:
    """
    Factor Scope 配置参数类
    
    根据需求文档FR-001到FR-003的要求，将所有阈值参数化，
    提升系统的可配置性和灵活性
    """
    # 健康度检查阈值 (FR-001)
    nan_threshold: float = 0.25              # NaN比例阈值  0.3 -> 0.25
    zero_threshold: float = 0.5             # 零值比例阈值
    coverage_threshold: float = 0.5         # 样本覆盖率阈值
    max_nan_streak_threshold: float = 0.3   # 最大连续NaN比例阈值
    
    # 有效性检查阈值 (FR-002)
    min_rankic_mean: float = 0.005          # 最小RankIC均值
    min_rankic_ir: float = 0.1              # 最小信息比率 0.1 -> 0.15
    min_tail_signal: float = 0.01           # 最小尾部信号强度
    
    # 稳定性检查阈值 (FR-003)
    sign_stability_threshold: float = 0.55  # 符号稳定性阈值  0.55 -> 0.6
    positive_ratio_threshold: float = 0.45  # 正值比例阈值   0.45  -> 0.5
    
    # 2026-01-06 修改: 强化熵值筛选，提升权重
    entropy_threshold: float = 0.1          # 信息熵阈值，更严格筛选纯噪声特征
    
    # 冗余性检查参数 (FR-004)
    cluster_corr_threshold: float = 0.7     # 聚类相关性阈值    0.8 -> 0.75 -> 0.7
    max_per_cluster: int = 1                # 每个聚类最大保留因子数
    
    # PCA分析参数 (FR-005)
    pca_n_components: int = 5               # PCA主成分数量
    
    # 2026-01-07 新增: 统一所有硬编码阈值到配置类
    # 极端值分位数阈值
    extreme_quantile_low: float = 0.01      # 极端值下分位数
    extreme_quantile_high: float = 0.99     # 极端值上分位数
    
    # RankIC分位数阈值
    rankic_q25: float = 0.25                # RankIC第一四分位数
    rankic_q50: float = 0.5                 # RankIC中位数
    rankic_q75: float = 0.75                # RankIC第三四分位数
    
    # 信息比率计算防除零参数
    ir_epsilon: float = 1e-6                # 避免除零的小数值
    
    # 质量评估评分参数
    rankic_mean_multiplier_high: int = 2    # RankIC均值高分倍数
    rankic_mean_multiplier_low: int = 1     # RankIC均值低分倍数
    rankic_ir_multiplier_high: int = 3      # 信息比率高分倍数
    rankic_ir_multiplier_low: int = 1       # 信息比率低分倍数
    sign_stability_bonus: float = 0.05      # 符号稳定性奖励阈值
    positive_ratio_bonus: float = 0.1       # 正值比例奖励阈值
    entropy_multiplier_high: int = 5        # 熵值高分倍数
    entropy_multiplier_low: int = 2         # 熵值低分倍数
    quality_score_a_threshold: int = 5      # A级质量评分阈值
    quality_score_b_threshold: int = 3      # B级质量评分阈值
    
    # PCA分析最小因子数
    pca_min_factors: int = 2                # PCA分析最小因子数量
    
    # 其他参数
    min_cs: int = 10                        # 最小横截面样本数
    entropy_bins: int = 20                  # 熵值计算分箱数
    n_jobs: Optional[int] = None            # 并行处理进程数


class FactorAnalysis:
    """
    特征因子筛选系统核心类
    
    实现需求文档中定义的六大功能模块：
    - FR-001: 健康度检查模块
    - FR-002: IC/RankIC有效性检查模块  
    - FR-003: 稳定性与一致性检查模块
    - FR-004: 冗余性与相关结构检查模块
    - FR-005: PCA结构分析模块
    - FR-006: 分层决策框架模块
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[FactorScopeConfig] = None,
        date_col: str = "datetime",
        symbol_col: str = "instrument", 
        # label_col: str = "Ref($adjclose,-5)/Ref($adjclose,-1)-1",
        label_col: str = "Ref($adjclose, -5) / Ref($adjclose, -1) - 1",
        # 2026-01-06 修改: 移除硬编码参数，使用配置类
        n_jobs: Union[int, None] = None,
    ):
        """
        初始化因子分析器
        
        Args:
            df: 包含因子数据的DataFrame
            config: 配置参数对象，如为None则使用默认配置
            date_col: 日期列名
            symbol_col: 股票代码列名
            label_col: 标签列名
            n_jobs: 并行处理进程数
        """
        self.df = df.copy()
        self.DATE_COL = date_col
        self.SYMBOL_COL = symbol_col
        self.LABEL_COL = label_col
        
        # 2026-01-06 新增: 使用配置类管理参数
        self.config = config or FactorScopeConfig()
        
        # 2026-01-06 新增: 设置日志记录
        self.logger = self._setup_logger()
        
        # 并行处理参数
        self.n_jobs = n_jobs or max(cpu_count() - 1, 1)

        # 提取因子列表
        self.factors = [
            c for c in df.columns
            if c not in [self.DATE_COL, self.SYMBOL_COL, self.LABEL_COL]
        ]
        
        self.logger.info(f"初始化完成，共发现 {len(self.factors)} 个因子待分析")

        # 预计算日期分组，提升性能
        self._date_groups = list(self.df.groupby(self.DATE_COL))
        
    def _setup_logger(self) -> logging.Logger:
        """
        2026-01-06 新增: 设置日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(f"{__name__}.FactorAnalysis")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # =========================
    # 工具函数 (Utility Functions)
    # =========================
    
    def _calc_entropy(self, x: pd.Series) -> float:
        """
        计算信息熵 (FR-001.4相关)
        
        用于识别信息熵接近0的纯噪声特征，这是需求文档中
        强调的核心筛选目标之一
        
        Args:
            x: 输入序列
            
        Returns:
            信息熵值，越小表示信息量越少
        """
        x = x.dropna()
        if x.nunique() <= 1:
            return 0.0
        # 2026-01-06 修改: 使用配置参数
        hist = np.histogram(x, bins=self.config.entropy_bins)[0]
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def _max_consecutive_nan_ratio(self, x: pd.Series) -> float:
        """
        计算最大连续NaN比例 (FR-001.6)
        
        防止因子在时间上"断裂式可用"，确保时间维度的连续性
        
        Args:
            x: 输入序列
            
        Returns:
            最大连续NaN比例
        """
        is_nan = x.isna().values
        max_streak, cur = 0, 0
        for v in is_nan:
            if v:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return max_streak / len(x)

    # =========================
    # RankIC 时间序列计算 (FR-002相关)
    # =========================
    
    def _calc_rankic_ts(self, factor: str):
        """
        计算因子的RankIC时间序列 (FR-002)
        
        这是IC/RankIC有效性检查的核心函数，计算因子与标签的
        横截面秩相关系数时间序列
        
        Args:
            factor: 因子名称
            
        Returns:
            tuple: (RankIC时间序列, 平均覆盖率)
        """
        ic_dict = {}
        coverage = []

        for dt, g in self._date_groups:
            sub = g[[factor, self.LABEL_COL]].dropna()
            # 2026-01-06 修改: 使用配置参数
            if len(sub) < self.config.min_cs:
                continue
            if sub[factor].nunique() <= 1 or sub[self.LABEL_COL].nunique() <= 1:
                continue

            try:
                # 计算Spearman秩相关系数
                ic = spearmanr(sub[factor], sub[self.LABEL_COL])[0]
                if np.isfinite(ic):
                    ic_dict[dt] = ic
                    coverage.append(len(sub) / len(g))
            except Exception as e:
                # 2026-01-06 新增: 异常处理和日志记录
                self.logger.warning(f"计算因子 {factor} 在日期 {dt} 的RankIC时出错: {e}")
                continue

        if not ic_dict:
            self.logger.warning(f"因子 {factor} 无有效RankIC数据")
            return pd.Series(dtype=float), 0.0

        return pd.Series(ic_dict), float(np.mean(coverage))

    # =========================
    # 单因子评估 (综合FR-001到FR-003)
    # =========================
    
    def _evaluate_factor(self, factor: str):
        """
        单因子综合评估函数
        
        整合健康度检查(FR-001)、有效性检查(FR-002)和稳定性检查(FR-003)
        的所有指标，为每个因子生成完整的评估报告
        
        Args:
            factor: 因子名称
            
        Returns:
            dict: 包含所有评估指标的字典，如果因子无效则返回None
        """
        try:
            self.logger.debug(f"开始评估因子: {factor}")
            x = self.df[factor]

            # FR-001: 健康度检查
            nan_ratio = x.isna().mean()
            zero_ratio = (x == 0).mean()
            entropy = self._calc_entropy(x)

            # 统计特征计算
            if x.dropna().nunique() > 1:
                skewness = skew(x.dropna())
                kurt = kurtosis(x.dropna())
            else:
                skewness, kurt = 0.0, 0.0

            # 极端值比例计算
            # 2026-01-07 修改: 使用配置参数统一阈值
            q_low, q_high = x.quantile([self.config.extreme_quantile_low, self.config.extreme_quantile_high])
            extreme_ratio = ((x < q_low) | (x > q_high)).mean()
            continuity = self._max_consecutive_nan_ratio(x)

            # FR-002: RankIC有效性检查
            rankic_ts, coverage = self._calc_rankic_ts(factor)
            if rankic_ts.empty:
                self.logger.warning(f"因子 {factor} 无有效RankIC数据，跳过")
                return None

            # RankIC统计指标
            mean_ic = rankic_ts.mean()
            std_ic = rankic_ts.std()
            # 2026-01-07 修改: 使用配置参数统一阈值
            ir = mean_ic / (std_ic + self.config.ir_epsilon)  # 避免除零

            # RankIC分位数分析 (FR-002.4)
            # 2026-01-07 修改: 使用配置参数统一阈值
            q25, q50, q75 = rankic_ts.quantile([
                self.config.rankic_q25, 
                self.config.rankic_q50, 
                self.config.rankic_q75
            ])
            
            # FR-003: 稳定性检查
            sign_stability = (np.sign(rankic_ts) == np.sign(mean_ic)).mean()
            positive_ratio = (rankic_ts > 0).mean()

            result = {
                "factor": factor,
                # 健康度指标
                "nan_ratio": nan_ratio,
                "zero_ratio": zero_ratio,
                "coverage": coverage,
                "entropy": entropy,
                "skew": skewness,
                "kurtosis": kurt,
                "extreme_ratio": extreme_ratio,
                "max_nan_streak_ratio": continuity,
                # 有效性指标
                "rankic_mean": mean_ic,
                "rankic_std": std_ic,
                "rankic_ir": ir,
                "rankic_q25": q25,
                "rankic_q50": q50,
                "rankic_q75": q75,
                # 稳定性指标
                "sign_stability": sign_stability,
                "positive_ratio": positive_ratio,
                "rankic_ts": rankic_ts,
            }
            
            self.logger.debug(f"因子 {factor} 评估完成")
            return result
            
        except Exception as e:
            # 2026-01-06 新增: 异常处理
            self.logger.error(f"评估因子 {factor} 时发生错误: {e}")
            return None

    # =========================
    # 冗余性分析 (FR-004)
    # =========================
    
    def _cluster_factors(self, rankic_df: pd.DataFrame):
        """
        因子聚类分析 (FR-004.2)
        
        基于RankIC时间序列的相关性进行层次聚类，识别信息层面
        的同质因子，降低特征共线性
        
        Args:
            rankic_df: RankIC时间序列DataFrame
            
        Returns:
            pd.Series: 因子聚类标签
        """
        try:
            # 计算RankIC序列相关性 (FR-004.1)
            corr = rankic_df.corr().abs()
            dist = 1 - corr

            # 2026-01-06 修改: 使用配置参数
            clustering = AgglomerativeClustering(
                metric="precomputed",
                linkage="average",
                distance_threshold=1 - self.config.cluster_corr_threshold,
                n_clusters=None,
            )
            labels = clustering.fit_predict(dist)
            
            self.logger.info(f"聚类完成，共生成 {len(set(labels))} 个聚类")
            return pd.Series(labels, index=rankic_df.columns, name="cluster")
            
        except Exception as e:
            self.logger.error(f"因子聚类过程中发生错误: {e}")
            # 返回默认聚类（每个因子单独一类）
            return pd.Series(range(len(rankic_df.columns)), 
                           index=rankic_df.columns, name="cluster")

    # =========================
    # PCA结构分析 (FR-005)
    # =========================
    
    def _run_pca(self, rankic_df: pd.DataFrame):
        """
        PCA结构分析 (FR-005)
        
        对RankIC时间序列进行主成分分析，理解特征空间结构，
        作为结构诊断工具而非直接筛选工具
        
        Args:
            rankic_df: RankIC时间序列DataFrame
            
        Returns:
            tuple: (特征载荷DataFrame, 解释方差比例列表)
        """
        if rankic_df.shape[1] < self.config.pca_min_factors:
            self.logger.warning("因子数量不足，跳过PCA分析")
            return pd.DataFrame(), []

        try:
            # 2026-01-06 修改: 使用配置参数
            n_comp = min(self.config.pca_n_components, rankic_df.shape[1])
            pca = PCA(n_components=n_comp)
            
            # 填充NaN值进行PCA分析
            pca.fit(rankic_df.fillna(0))

            # FR-005.3: 特征载荷分析
            loading = pd.DataFrame(
                pca.components_.T,
                index=rankic_df.columns,
                columns=[f"PC{i+1}" for i in range(n_comp)],
            )

            # FR-005.2: 解释方差分析
            explained_var = pca.explained_variance_ratio_
            
            self.logger.info(f"PCA分析完成，前{n_comp}个主成分累计解释方差: "
                           f"{explained_var.sum():.3f}")
            
            return loading, explained_var
            
        except Exception as e:
            self.logger.error(f"PCA分析过程中发生错误: {e}")
            return pd.DataFrame(), []

    # =========================
    # 决策层 1：可用性判定 (FR-006.1)
    # =========================
    
    def _eligibility(self, r):
        """
        可用性层判定 (FR-006.1)
        
        基于健康度与基础有效性分析结果，判定因子是否有资格
        进入模型候选池
        
        Args:
            r: 因子评估结果行
            
        Returns:
            str: 可用性判定结果
        """
        # 2026-01-06 修改: 强化熵值筛选，符合需求重点
        if r.entropy < self.config.entropy_threshold:
            return "HARD_DROP_ENTROPY"  # 新增：信息熵过低，纯噪声特征
            
        # 2026-01-06 修改: 使用配置参数
        if r.nan_ratio > self.config.nan_threshold:
            return "HARD_DROP_NAN"
            
        if r.zero_ratio > self.config.zero_threshold:
            return "HARD_DROP_ZERO"
            
        if r.coverage < self.config.coverage_threshold:
            return "HARD_DROP_COVERAGE"
            
        if r.max_nan_streak_ratio > self.config.max_nan_streak_threshold:
            return "HARD_DROP_CONTINUITY"
            
        return "ELIGIBLE"

    # =========================
    # 决策层 2：信息质量评估 (FR-006.2)
    # =========================
    
    def _quality(self, r):
        """
        信息质量层评估 (FR-006.2)
        
        基于IC、IR、稳定性、Tail行为、entropy等综合评估，
        不采用单一hard rule，输出信息质量分层
        
        Args:
            r: 因子评估结果行
            
        Returns:
            str: 信息质量等级 (QUALITY_A/B/C)
        """
        score = 0

        # RankIC均值评分
        # 2026-01-07 修改: 使用配置参数统一评分阈值
        if abs(r.rankic_mean) > self.config.min_rankic_mean * self.config.rankic_mean_multiplier_high:
            score += 2
        elif abs(r.rankic_mean) > self.config.min_rankic_mean * self.config.rankic_mean_multiplier_low:
            score += 1

        # 信息比率评分
        if r.rankic_ir > self.config.min_rankic_ir * self.config.rankic_ir_multiplier_high:
            score += 2
        elif r.rankic_ir > self.config.min_rankic_ir * self.config.rankic_ir_multiplier_low:
            score += 1

        # FR-002.4: 尾部信号检查
        tail_signal = max(abs(r.rankic_q25), abs(r.rankic_q75))
        if tail_signal > self.config.min_tail_signal:
            score += 1

        # FR-003: 稳定性评分
        if r.sign_stability > self.config.sign_stability_threshold + self.config.sign_stability_bonus:
            score += 1

        if r.positive_ratio > self.config.positive_ratio_threshold + self.config.positive_ratio_bonus:
            score += 1

        # 2026-01-07 修改: 使用配置参数统一熵值评分阈值
        if r.entropy > self.config.entropy_threshold * self.config.entropy_multiplier_high:  # 信息量充足
            score += 1
        elif r.entropy < self.config.entropy_threshold * self.config.entropy_multiplier_low:  # 信息量不足
            score -= 2

        # 质量分层
        # 2026-01-07 修改: 使用配置参数统一质量评分阈值
        if score >= self.config.quality_score_a_threshold:
            return "QUALITY_A"
        elif score >= self.config.quality_score_b_threshold:
            return "QUALITY_B"
        else:
            return "QUALITY_C"

    # =========================
    # 结构角色分析 (FR-006.3)
    # =========================
    
    def apply_cluster_pca_role(self, report, loading):
        """
        结构角色层分析 (FR-006.3)
        
        基于RankIC聚类与PCA loading结果，判定因子在结构中的角色
        
        Args:
            report: 因子评估报告DataFrame
            loading: PCA载荷DataFrame
            
        Returns:
            pd.DataFrame: 添加结构角色信息的报告
        """
        report = report.copy()
        report["structure_role"] = "REDUNDANT"

        if loading.empty:
            self.logger.warning("PCA载荷为空，无法进行结构角色分析")
            report["structure_role"] = "UNDEFINED"
            return report

        # 为每个聚类确定主导主成分
        cluster_main_pc = {}
        for c, g in report.groupby("cluster"):
            facs = g["factor"].values
            load_sub = loading.loc[loading.index.intersection(facs)]
            if load_sub.empty:
                continue
            # 选择平均载荷最大的主成分作为该聚类的主导成分
            cluster_main_pc[c] = load_sub.abs().mean().idxmax()

        report["cluster_main_pc"] = report["cluster"].map(cluster_main_pc)

        def pca_score(row):
            """计算因子在其聚类主导主成分上的载荷分数"""
            if row["factor"] not in loading.index:
                return 0.0
            pc = row["cluster_main_pc"]
            if pd.isna(pc):
                return 0.0
            return abs(loading.loc[row["factor"], pc])

        report["pca_score"] = report.apply(pca_score, axis=1)

        # 为每个聚类分配角色
        for c, g in report.groupby("cluster"):
            # 按PCA分数和信息比率排序
            g_sorted = g.sort_values(
                ["pca_score", "rankic_ir"],
                ascending=False
            )
            
            # 选择核心因子（每个聚类根据配置保留数量）
            # 2026-01-07 修改: 使用配置参数统一聚类保留数量
            core = g_sorted.head(self.config.max_per_cluster)["factor"]
            
            # 2026-01-07 修改: 修正可选因子选择逻辑，移除硬编码
            # 如果配置允许保留多个因子，则除了核心因子外的其他因子作为可选因子
            if self.config.max_per_cluster > 1:
                # 当max_per_cluster > 1时，前面的是CORE，后面的是OPTIONAL
                optional_count = min(self.config.max_per_cluster, len(g_sorted) - self.config.max_per_cluster)
                if optional_count > 0:
                    optional = g_sorted.iloc[self.config.max_per_cluster:self.config.max_per_cluster + optional_count]["factor"]
                else:
                    optional = pd.Series([], dtype=str, name="factor")
            else:
                # 当max_per_cluster = 1时，没有可选因子
                optional = pd.Series([], dtype=str, name="factor")

            report.loc[report["factor"].isin(core), "structure_role"] = "CORE"
            report.loc[report["factor"].isin(optional), "structure_role"] = "OPTIONAL"

        self.logger.info(f"结构角色分析完成: "
                        f"CORE={sum(report['structure_role']=='CORE')}, "
                        f"OPTIONAL={sum(report['structure_role']=='OPTIONAL')}, "
                        f"REDUNDANT={sum(report['structure_role']=='REDUNDANT')}")
        
        return report

    # =========================
    # 最终建议生成 (FR-006.4)
    # =========================
    
    def _final_advice(self, r):
        """
        最终建议生成 (FR-006.4)
        
        综合三层分析结果，给出可解释、可操作的因子建议
        
        Args:
            r: 因子评估结果行
            
        Returns:
            str: 最终建议 (KEEP_CORE/KEEP_OPTIONAL/DROP_REDUNDANT/DROP_INVALID)
        """
        # 不可用因子直接淘汰
        if r.eligibility != "ELIGIBLE":
            return "DROP_INVALID"
            
        # 核心因子保留
        if r.structure_role == "CORE":
            return "KEEP_CORE"
            
        # 冗余且质量差的因子淘汰
        if r.structure_role == "REDUNDANT" and r.quality == "QUALITY_C":
            return "DROP_REDUNDANT"
            
        # 其他情况作为可选因子保留
        return "KEEP_OPTIONAL"
    
    # =========================
    # 2026-01-06 新增: 结果导出功能
    # =========================
    
    def export_results(self, report: pd.DataFrame, loading: pd.DataFrame, 
                      output_dir: str, format: str = 'both') -> Dict[str, str]:
        """
        导出分析结果 (符合需求NFR-004)
        
        支持JSON格式的筛选结果和CSV格式的详细分析报告
        
        Args:
            report: 因子分析报告
            loading: PCA载荷矩阵
            output_dir: 输出目录
            format: 输出格式 ('json', 'csv', 'both')
            
        Returns:
            dict: 输出文件路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        try:
            if format in ['json', 'both']:
                # JSON格式筛选结果
                json_result = {
                    "metadata": {
                        "timestamp": timestamp,
                        "total_factors": len(report),
                        "config": self.config.__dict__
                    },
                    "summary": {
                        "keep_core": len(report[report['final_advice'] == 'KEEP_CORE']),
                        "keep_optional": len(report[report['final_advice'] == 'KEEP_OPTIONAL']),
                        "drop_redundant": len(report[report['final_advice'] == 'DROP_REDUNDANT']),
                        "drop_invalid": len(report[report['final_advice'] == 'DROP_INVALID'])
                    },
                    "recommendations": {
                        advice: report[report['final_advice'] == advice]['factor'].tolist()
                        for advice in report['final_advice'].unique()
                    }
                }
                
                json_path = output_dir / f"factor_selection_result_{timestamp}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, ensure_ascii=False, indent=2)
                output_files['json'] = str(json_path)
                self.logger.info(f"JSON结果已保存至: {json_path}")
            
            if format in ['csv', 'both']:
                # CSV格式详细报告
                csv_path = output_dir / f"factor_analysis_report_{timestamp}.csv"
                # 移除rankic_ts列（无法序列化到CSV）
                report_csv = report.drop(columns=['rankic_ts'], errors='ignore')
                report_csv.to_csv(csv_path, index=False, encoding='utf-8')
                output_files['csv'] = str(csv_path)
                self.logger.info(f"CSV报告已保存至: {csv_path}")
                
                # PCA载荷矩阵
                if not loading.empty:
                    pca_path = output_dir / f"factor_pca_loading_{timestamp}.csv"
                    loading.to_csv(pca_path, encoding='utf-8')
                    output_files['pca'] = str(pca_path)
                    self.logger.info(f"PCA载荷已保存至: {pca_path}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"导出结果时发生错误: {e}")
            return {}

    # =========================
    # 主入口函数 (Main Entry Point)
    # =========================
    
    def run(self):
        """
        因子筛选主流程
        
        执行完整的因子筛选流程，整合所有六大功能模块：
        1. 健康度检查 (FR-001)
        2. IC/RankIC有效性检查 (FR-002)  
        3. 稳定性与一致性检查 (FR-003)
        4. 冗余性与相关结构检查 (FR-004)
        5. PCA结构分析 (FR-005)
        6. 分层决策框架 (FR-006)
        
        Returns:
            tuple: (因子分析报告DataFrame, PCA载荷DataFrame)
        """
        self.logger.info("开始因子筛选流程")
        start_time = datetime.now()
        
        # 并行评估所有因子
        self.logger.info(f"开始并行评估 {len(self.factors)} 个因子，使用 {self.n_jobs} 个进程")
        with Pool(self.n_jobs) as pool:
            results = pool.map(self._evaluate_factor, self.factors)

        # 过滤有效结果
        results = [r for r in results if r is not None]
        if not results:
            self.logger.error("没有有效的因子评估结果")
            return pd.DataFrame(), pd.DataFrame()
            
        self.logger.info(f"成功评估 {len(results)} 个因子")
        
        # 构建评估报告
        report = pd.DataFrame(results)

        # FR-004: 冗余性分析 - 构建RankIC时间序列矩阵
        self.logger.info("开始冗余性分析")
        rankic_df = pd.concat(
            {r["factor"]: r["rankic_ts"] for r in results},
            axis=1
        )

        # 因子聚类
        cluster = self._cluster_factors(rankic_df)
        report = report.merge(cluster, left_on="factor", right_index=True, how='left')

        # FR-005: PCA结构分析
        self.logger.info("开始PCA结构分析")
        loading, explained = self._run_pca(rankic_df)
        
        # 将解释方差比例添加到报告中
        for i, v in enumerate(explained):
            report[f"pca_explained_var_{i+1}"] = v

        # FR-006: 分层决策框架
        self.logger.info("开始分层决策分析")
        
        # 决策层1: 可用性判定
        report["eligibility"] = report.apply(self._eligibility, axis=1)
        
        # 决策层2: 信息质量评估
        report["quality"] = report.apply(self._quality, axis=1)

        # 决策层3: 结构角色分析
        report = self.apply_cluster_pca_role(report, loading)

        # 最终建议生成
        report["final_advice"] = report.apply(self._final_advice, axis=1)

        # 结果排序：按建议类型、质量、信息比率排序
        report = report.sort_values(
            ["final_advice", "quality", "rankic_ir"],
            ascending=[True, True, False]
        )

        # 统计结果
        advice_counts = report['final_advice'].value_counts()
        elapsed_time = datetime.now() - start_time
        
        self.logger.info("因子筛选完成")
        self.logger.info(f"处理时间: {elapsed_time}")
        self.logger.info(f"筛选结果统计: {dict(advice_counts)}")

        return report, loading


# =========================
# 调用示例和配置示例
# =========================
if __name__ == "__main__":
    """
    Factor Scope 使用示例
    
    展示如何使用配置类和新的导出功能
    """
    # 2026-01-06 修改: 使用相对路径，符合项目结构规范
    data_dir = "/home/a/notebook/zxf/data/Daily_data/Training_data/csi800"  
    # notebook/zxf/data/Daily_data/Training_data/csi800/csi800_self_dl_train.pkl
    
    try:
        # 加载数据
        with open(f"{data_dir}/csi800_self_dl_train.pkl", "rb") as f:
            dl = pickle.load(f)

        df = dl.data.droplevel(0, axis=1)
        df.reset_index(inplace=True)

        # 2026-01-06 新增: 使用自定义配置
        custom_config = FactorScopeConfig(
            entropy_threshold=0.1,  # 更严格的熵值筛选 0.05 -> 0.1
            nan_threshold=0.25,      # 更严格的NaN阈值
            cluster_corr_threshold=0.68,  # 更严格的聚类阈值 0.85 -> 0.88 -> 0.8 -> 0.75 -> 0.7
            max_per_cluster=1,       # 每个聚类保留更多因子 2-> 1
        )

        # 初始化分析器
        fa = FactorAnalysis(
            df=df, 
            config=custom_config,
            # n_jobs=max(cpu_count() - 1, 1)
            n_jobs=3
        )
        
        # 执行分析
        report, pca_loading = fa.run()

        # 2026-01-06 新增: 使用新的导出功能
        output_files = fa.export_results(
            report=report,
            loading=pca_loading,
            output_dir=data_dir,
            format='both'
        )
        
        print("=== Factor Scope 分析完成 ===")
        print(f"输出文件: {output_files}")
        print("\n=== 筛选结果摘要 ===")
        print(report['final_advice'].value_counts())
        print("\n=== 前10个推荐因子 ===")
        print(report[report['final_advice'].isin(['KEEP_CORE', 'KEEP_OPTIONAL'])].head(10)[
            ['factor', 'final_advice', 'quality', 'rankic_ir', 'entropy']
        ])
        
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        print("请确保数据文件路径正确")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

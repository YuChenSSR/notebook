"""
多种子预测文件交集处理系统
Multi-Seed Prediction Intersection Processing

功能：从多个不同种子训练的模型预测结果中，通过Top-N交集策略生成融合预测文件
作者：量化交易团队
日期：2026-01-16
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import fire


class PredictionIntersectionProcessor:
    """多种子预测文件交集处理器"""
    
    def __init__(self, top_n: int = 30, verbose: bool = True):
        """
        初始化处理器
        
        Args:
            top_n: 每天选取的Top-N股票数量，默认30
            verbose: 是否输出详细信息
        """
        self.top_n = top_n
        self.verbose = verbose
        self.stats = {
            'total_days': 0,
            'total_intersections': 0,
            'avg_intersection_size': 0,
            'min_intersection_size': float('inf'),
            'max_intersection_size': 0,
            'empty_days': []
        }
    
    def read_prediction_file(self, file_path: str) -> pd.DataFrame:
        """
        读取并验证预测文件
        
        Args:
            file_path: 预测文件路径
            
        Returns:
            包含datetime、instrument、score的DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"预测文件不存在: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 验证必需列
        required_columns = ['datetime', 'instrument', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件 {file_path} 缺少必需列: {missing_columns}")
        
        # 转换数据类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['instrument'] = df['instrument'].astype(str)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # 检查是否有无效数据
        if df['score'].isna().any():
            print(f"警告: 文件 {file_path} 中存在无效的score值，已删除")
            df = df.dropna(subset=['score'])
        
        if self.verbose:
            print(f"成功读取文件: {file_path}")
            print(f"  - 数据行数: {len(df)}")
            print(f"  - 日期范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
            print(f"  - 唯一股票数: {df['instrument'].nunique()}")
        
        return df
    
    def select_top_n(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按天选取Top-N股票
        
        Args:
            df: 包含datetime、instrument、score的DataFrame
            
        Returns:
            每天Top-N股票的DataFrame
        """
        top_n_list = []
        
        for date, group in df.groupby('datetime'):
            # 按score降序排序
            sorted_group = group.sort_values('score', ascending=False)
            
            # 选取前Top-N个
            top_n_stocks = sorted_group.head(self.top_n)
            top_n_list.append(top_n_stocks)
        
        result = pd.concat(top_n_list, ignore_index=True)
        
        if self.verbose:
            print(f"  - Top-{self.top_n} 选取完成，共 {len(result)} 条记录")
        
        return result
    
    def calculate_intersection(self, 
                              dfs: List[pd.DataFrame], 
                              seed_names: List[str]) -> pd.DataFrame:
        """
        计算多个预测文件的交集
        
        Args:
            dfs: 预测DataFrame列表
            seed_names: 种子名称列表
            
        Returns:
            包含交集结果和所有种子评分的DataFrame
        """
        if len(dfs) < 2:
            raise ValueError("至少需要2个预测文件才能计算交集")
        
        # 获取所有日期
        all_dates = sorted(set.union(*[set(df['datetime'].unique()) for df in dfs]))
        self.stats['total_days'] = len(all_dates)
        
        intersection_results = []
        
        for date in all_dates:
            # 获取每个文件在该日期的股票集合
            daily_stocks = []
            daily_dfs = []
            
            for df in dfs:
                date_df = df[df['datetime'] == date]
                if len(date_df) > 0:
                    daily_stocks.append(set(date_df['instrument'].values))
                    daily_dfs.append(date_df)
                else:
                    daily_stocks.append(set())
                    daily_dfs.append(pd.DataFrame())
            
            # 计算交集
            if all(len(stocks) > 0 for stocks in daily_stocks):
                intersection_stocks = set.intersection(*daily_stocks)
            else:
                intersection_stocks = set()
            
            # 统计信息
            intersection_size = len(intersection_stocks)
            self.stats['total_intersections'] += intersection_size
            self.stats['min_intersection_size'] = min(
                self.stats['min_intersection_size'], 
                intersection_size
            )
            self.stats['max_intersection_size'] = max(
                self.stats['max_intersection_size'], 
                intersection_size
            )
            
            if intersection_size == 0:
                self.stats['empty_days'].append(date)
                if self.verbose:
                    print(f"警告: {date} 交集为空")
                continue
            
            # 构建结果DataFrame
            for stock in intersection_stocks:
                row_data = {
                    'datetime': date,
                    'instrument': stock
                }
                
                # 添加每个种子的评分
                scores = []
                for i, (daily_df, seed_name) in enumerate(zip(daily_dfs, seed_names)):
                    stock_score = daily_df[daily_df['instrument'] == stock]['score'].values[0]
                    row_data[f'seed{i+1}_score'] = stock_score
                    scores.append(stock_score)
                
                # 计算平均评分
                row_data['score'] = np.mean(scores)
                
                intersection_results.append(row_data)
            
            if self.verbose:
                print(f"日期 {date}: 交集股票数 = {intersection_size}")
        
        # 计算平均交集大小
        if self.stats['total_days'] > 0:
            self.stats['avg_intersection_size'] = (
                self.stats['total_intersections'] / self.stats['total_days']
            )
        
        result_df = pd.DataFrame(intersection_results)
        
        # 按日期和评分排序
        if len(result_df) > 0:
            result_df = result_df.sort_values(['datetime', 'score'], ascending=[True, False])
        
        return result_df
    
    def parse_filename_metadata(self, file_path: str) -> Dict[str, str]:
        """
        从文件名中解析元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含元数据的字典
        """
        filename = Path(file_path).stem
        parts = filename.split('_')
        
        metadata = {}
        try:
            # 格式: master_predictions_backday_{backday}_{universe}_{seed}_{step}.csv
            if len(parts) >= 5:
                metadata['backday'] = parts[3]
                metadata['universe'] = parts[4]
                if len(parts) >= 6:
                    metadata['seed'] = parts[5]
                if len(parts) >= 7:
                    metadata['step'] = parts[6]
        except Exception as e:
            print(f"警告: 无法解析文件名 {filename}: {e}")
        
        return metadata
    
    def generate_output_filename(self, 
                                 input_files: List[str], 
                                 output_dir: str) -> str:
        """
        生成输出文件名
        
        Args:
            input_files: 输入文件路径列表
            output_dir: 输出目录
            
        Returns:
            输出文件完整路径
        """
        # 解析所有文件的元数据
        metadatas = [self.parse_filename_metadata(f) for f in input_files]
        
        # 提取种子信息
        seeds = []
        universe = None
        backday = None
        
        for metadata in metadatas:
            if 'seed' in metadata:
                seeds.append(metadata['seed'])
            if 'universe' in metadata and universe is None:
                universe = metadata['universe']
            if 'backday' in metadata and backday is None:
                backday = metadata['backday']
        
        # 组合种子标识符
        seeds_str = '_'.join(seeds) if seeds else 'multi'
        universe_str = universe if universe else 'unknown'
        
        # 生成文件名
        if backday:
            filename = f"master_predictions_intersection_backday_{backday}_{universe_str}_{seeds_str}.csv"
        else:
            filename = f"master_predictions_intersection_{universe_str}_{seeds_str}.csv"
        
        return os.path.join(output_dir, filename)
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("处理统计信息")
        print("="*60)
        print(f"总处理天数: {self.stats['total_days']}")
        print(f"总交集股票数: {self.stats['total_intersections']}")
        print(f"平均每日交集数: {self.stats['avg_intersection_size']:.2f}")
        print(f"最小交集数: {self.stats['min_intersection_size']}")
        print(f"最大交集数: {self.stats['max_intersection_size']}")
        print(f"交集为空的天数: {len(self.stats['empty_days'])}")
        if self.stats['empty_days']:
            print(f"空交集日期: {[str(d)[:10] for d in self.stats['empty_days'][:5]]}")
        print("="*60 + "\n")
    
    def process(self, 
                seed1_path: str,
                seed2_path: str,
                output_dir: str = None,
                output_file: str = None) -> str:
        """
        处理多个种子预测文件的交集
        
        Args:
            seed1_path: 第一个种子预测文件路径
            seed2_path: 第二个种子预测文件路径
            output_dir: 输出目录，默认与第一个输入文件同目录
            output_file: 输出文件名，默认自动生成
            
        Returns:
            输出文件路径
        """
        print("\n" + "="*60)
        print("多种子预测文件交集处理")
        print("="*60)
        print(f"Top-N: {self.top_n}")
        print(f"输入文件1: {seed1_path}")
        print(f"输入文件2: {seed2_path}")
        print("="*60 + "\n")
        
        # 读取预测文件
        print("步骤 1/4: 读取预测文件...")
        df1 = self.read_prediction_file(seed1_path)
        df2 = self.read_prediction_file(seed2_path)
        
        # 选取Top-N
        print(f"\n步骤 2/4: 选取每日Top-{self.top_n}股票...")
        top_n_df1 = self.select_top_n(df1)
        top_n_df2 = self.select_top_n(df2)
        
        # 计算交集
        print("\n步骤 3/4: 计算交集并融合评分...")
        seed_names = [
            Path(seed1_path).stem.split('_')[-2] if '_' in Path(seed1_path).stem else 'seed1',
            Path(seed2_path).stem.split('_')[-2] if '_' in Path(seed2_path).stem else 'seed2'
        ]
        result_df = self.calculate_intersection([top_n_df1, top_n_df2], seed_names)
        
        # 保存结果
        print("\n步骤 4/4: 保存结果文件...")
        if output_dir is None:
            output_dir = str(Path(seed1_path).parent)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if output_file is None:
            output_path = self.generate_output_filename(
                [seed1_path, seed2_path], 
                output_dir
            )
        else:
            output_path = os.path.join(output_dir, output_file)
        
        result_df.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        
        print(f"结果已保存至: {output_path}")
        print(f"输出文件行数: {len(result_df)}")
        
        # 打印统计信息
        self.print_statistics()
        
        return output_path


def main(
    seed1_path: str =  f"/home/a/notebook/zxf/data/Daily_data/Good_seed/seed7/master_predictions_backday_8_csi800_27_31.csv",
    seed2_path: str =  f"/home/a/notebook/zxf/data/Daily_data/Good_seed/seed8/master_predictions_backday_8_csi800_91_35.csv",
    output_dir: str = f"/home/a/notebook/zxf/data/Daily_data/Good_seed/seed7",
    output_file: str = "master_predictions_backday_8_csi800_multi.csv",
    top_n: int = 50,
    verbose: bool = True
):
    """
    多种子预测文件交集处理主函数
    
    Args:
        seed1_path: 第一个种子预测文件路径
        seed2_path: 第二个种子预测文件路径
        output_dir: 输出目录，默认与第一个输入文件同目录
        output_file: 输出文件名，默认自动生成
        top_n: 每天选取的Top-N股票数量，默认30
        verbose: 是否输出详细信息，默认True
    
    示例:
        python prediction_intersection_multi_seed.py \\
            --seed1_path="Data/Experiment_results/backtest_seed_mutli/master_predictions_backday_8_csi800_27_31.csv" \\
            --seed2_path="Data/Experiment_results/backtest_seed_mutli/master_predictions_backday_8_csi800_91_35.csv" \\
            --top_n=30
    """
    processor = PredictionIntersectionProcessor(top_n=top_n, verbose=verbose)
    output_path = processor.process(
        seed1_path=seed1_path,
        seed2_path=seed2_path,
        output_dir=output_dir,
        output_file=output_file
    )
    
    print(f"\n✓ 处理完成！输出文件: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)

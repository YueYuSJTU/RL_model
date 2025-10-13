import os
import shutil
import numpy as np
import logging
from typing import Dict, List
from src.evaluate_pool import evaluate_versus


# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PoolManager:
    """
    管理对手池，简化了评估和替换逻辑。

    该类负责维护一个固定大小的对手模型池。当一个新模型被提交时：
    1. 如果池未满，则直接添加。
    2. 如果池已满，新模型将逐一挑战池中所有现有模型。
    3. 只有当新模型对至少一个现有模型的对战评分大于0时，才会触发替换。
    4. 替换目标是那个被新模型以最高分击败的对手。
    """
    
    def __init__(self, pool_path: str, max_pool_size: int = 10, 
                 score_weights: Dict[str, float] = None,
                 whitelist: List[str] = ["1"]):
        """
        初始化 PoolManager.

        Args:
            pool_path (str): 对手池的根目录路径。
            max_pool_size (int): 对手池的最大容量。
            score_weights (Dict[str, float], optional): 
                用于计算模型综合评分的权重字典。
                默认为: {'win_rate': 1.0, 'loss_rate': -1.5, 'draw_rate': 0.5, 'opponent_fall_rate': -0.5}
            whitelist (List[str], optional): 不会被替换的模型编号列表 (例如: ["1", "5"])。
        """
        self.pool_path = pool_path
        self.max_pool_size = max_pool_size
        self.whitelist = set(whitelist) if whitelist is not None else set()
        if self.whitelist:
            logging.info(f"对手池白名单已激活，受保护的模型: {sorted(list(self.whitelist))}")
        
        if score_weights is None:
            # 默认评分系统,评分代表当前模型比这个对手好多少:
            # - 胜率是主要加分项
            # - 失败率是主要扣分项（权重更高，强调避免失败）
            # - 平局有少量加分
            # - 对手自主坠机证明当前模型优于对手，有少量加分
            # 请注意只有总分大于0才会触发替换，不光要看相对值，还要看绝对值
            self.score_weights = {
                'win_rate': 1.0, 
                'loss_rate': -1.0, 
                'draw_rate': 0.0,
                'opponent_fall_rate': 0.5
            }
        else:
            self.score_weights = score_weights
            
        os.makedirs(self.pool_path, exist_ok=True)
        logging.info(f"对手池管理器已初始化，路径: '{self.pool_path}', 最大容量: {self.max_pool_size}")

    def _get_pool_models(self) -> List[str]:
        """扫描并返回池中所有模型的目录名（按数字排序）。"""
        model_dirs = []
        if not os.path.exists(self.pool_path):
            return model_dirs
            
        for d in os.listdir(self.pool_path):
            if os.path.isdir(os.path.join(self.pool_path, d)):
                try:
                    int(d) # 确保目录名是数字
                    model_dirs.append(d)
                except ValueError:
                    logging.warning(f"在对手池中发现无效的目录名 '{d}'，已忽略。")
        
        return sorted(model_dirs, key=int)

    def _calculate_score(self, stats: Dict[str, float]) -> float:
        """根据评估结果和权重计算综合评分。"""
        score = 0.0
        for key, weight in self.score_weights.items():
            score += stats.get(key, 0.0) * weight
        return score

    def _add_model_to_pool(self, new_model_path: str, dest_dir_name: str):
        """
        将一个新模型文件夹完整复制到对手池中，并命名为指定的数字。

        Args:
            new_model_path (str): 待复制模型的源路径（文件夹）。
            dest_dir_name (str): 在对手池中创建的目标目录名 (例如 "1", "2"...).
        """
        destination_path = os.path.join(self.pool_path, dest_dir_name)
        
        # 如果目标已存在（替换情况），先删除
        if os.path.exists(destination_path):
            shutil.rmtree(destination_path)
            logging.info(f"已移除旧模型 '{destination_path}'。")
        
        # 复制整个文件夹
        shutil.copytree(new_model_path, destination_path)
        logging.info(f"模型从 '{new_model_path}' 完整复制到 '{destination_path}' 成功。")


    def update_pool(self, new_model_path: str, n_episodes: int = 500):
        """
        尝试将一个新训练的模型更新到对手池中。

        - 如果池未满，直接将新模型添加进去。
        - 如果池已满，新模型将挑战池中所有对手。如果它战胜了至少一个对手，
          它将替换掉被它以最高分击败的那个对手。

        Args:
            new_model_path (str): 新训练模型的路径 (文件夹)。
            n_episodes (int): 评估时每个模型对战的回合数。
        """
        current_models = self._get_pool_models()
        
        # 情况一：对手池未满
        if len(current_models) < self.max_pool_size:
            next_model_num = 1
            if current_models:
                next_model_num = int(current_models[-1]) + 1
            
            logging.info(f"对手池未满 (当前 {len(current_models)}/{self.max_pool_size})。正在添加新模型为 '{next_model_num}'...")
            self._add_model_to_pool(new_model_path, str(next_model_num))
            return

        # 情况二：对手池已满，新模型需要作为挑战者进行评估
        logging.info(f"对手池已满。新模型将挑战池中所有 {len(current_models)} 个对手...")
        
        match_results = {}
        # 新模型作为主控，与池中每个模型对战
        for opponent_name in current_models:
            opponent_num = int(opponent_name)
            logging.info(f"对战开始: 新模型 vs. 对手 '{opponent_name}'")
            
            win_rate, draw_rate, loss_rate, opponent_fall_rate, _, _ = evaluate_versus(
                model_path=new_model_path,
                pool_path=self.pool_path,
                opponent_num=opponent_num,
                n_episodes=n_episodes,
                use_tqdm=False
            )
            
            # 记录这场对战的结果
            match_results[opponent_name] = {
                "win_rate": win_rate, "draw_rate": draw_rate,
                "loss_rate": loss_rate, "opponent_fall_rate": opponent_fall_rate
            }

        # 计算新模型对战每个对手的得分
        match_scores = {
            name: self._calculate_score(stats) for name, stats in match_results.items()
        }
        
        logging.info("--- 对战评估结果 ---")
        for name, score in match_scores.items():
            stats = match_results[name]
            logging.info(f"vs. 对手 '{name}': Win={stats['win_rate']:.2%}, Loss={stats['loss_rate']:.2%}, Draw={stats['draw_rate']:.2%}, Opponent fall={stats['opponent_fall_rate']:.2%}. --> 得分: {score:.3f}")

        # 找出被新模型“战胜”的对手（得分 > 0）
        defeated_opponents = {
            name: score for name, score in match_scores.items() if score > 0
        }

        # 决策：是否进行替换
        if not defeated_opponents:
            logging.info("新模型未能战胜任何现有对手，对手池不作变更。")
            return
        
        # === 从被击败的对手中，筛选出“可被替换”的候选人 ===
        replaceable_candidates = {
            name: score for name, score in defeated_opponents.items() if name not in self.whitelist
        }
        # 如果筛选后，没有可替换的候选人了（例如，所有被击败的都在白名单里）
        if not replaceable_candidates:
            logging.info(
                "新模型战胜了一些对手，但所有被战胜的对手都在白名单中。对手池不作变更。"
            )
            return
        
        # 如果战胜了至少一个对手，找出被最“轻松”击败的那个（即新模型得分最高的对局）
        victim_name = max(replaceable_candidates, key=replaceable_candidates.get)
        victim_score = replaceable_candidates[victim_name]

        logging.info(f"新模型战胜了 {len(defeated_opponents)} 个对手。")
        logging.info(f"表现最差的对手是 '{victim_name}' (新模型得分 {victim_score:.3f})。")
        logging.info(f"正在用新模型替换对手 '{victim_name}'...")
        
        # 执行替换
        self._add_model_to_pool(new_model_path, victim_name)
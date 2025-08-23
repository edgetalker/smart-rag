"""
ContextManager(上下文管理)
从检索的答案中挑选出质量最好的N个结果
可配置的pipeline:
质量筛选 -> 去重 -> 排序 -> 数量控制 -> 内容优化

"""

from typing import List, Dict, Any
import re
import logging
from ..models import SearchResult
from ..config import config

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器"""

    def __init__(self,
                 max_results: int = None,
                 enable_quality_filter: bool = True,
                 enable_deduplication: bool = True):
        # 配置参数
        self.max_results = max_results or config.get("MAX_CONTEXT_RESULTS", 15)
        self.enable_quality_filter = enable_quality_filter
        self.enable_deduplication = enable_deduplication

        # 质量阈值
        self.min_content_length = 50
        self.max_content_length = 3000  # 单个文档最大长度
        self.similarity_threshold = 0.85  # 去重相似度阈值

    def optimize_context(
            self,
            query: str,
            search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """优化上下文 - 专注质量而非数量控制"""

        if not search_results:
            return {
                "optimized_results": [],
                "optimization_info": {
                    "original_count": 0,
                    "final_count": 0,
                    "message": "没有搜索结果"
                }
            }

        original_count = len(search_results)

        # 1. 质量筛选
        quality_filtered = self._filter_by_quality(search_results) if self.enable_quality_filter else search_results

        # 2. 去重处理
        deduplicated = self._remove_duplicates(quality_filtered) if self.enable_deduplication else quality_filtered

        # 3. 智能排序
        ranked_results = self._rank_results(query, deduplicated)

        # 4. 数量控制（保留top结果）
        final_results = ranked_results[:self.max_results]

        # 5. 内容长度优化
        optimized_results = self._optimize_content_length(final_results)

        # 生成优化报告
        optimization_info = {
            "original_count": original_count,
            "after_quality_filter": len(quality_filtered),
            "after_deduplication": len(deduplicated),
            "final_count": len(optimized_results),
            "avg_score": sum(r.score for r in optimized_results) / len(optimized_results) if optimized_results else 0,
            "total_content_length": sum(len(r.content) for r in optimized_results),
            "optimization_applied": {
                "quality_filter": self.enable_quality_filter,
                "deduplication": self.enable_deduplication,
                "content_optimization": True
            }
        }

        return {
            "optimized_results": optimized_results,
            "optimization_info": optimization_info
        }

    def _filter_by_quality(self, results: List[SearchResult]) -> List[SearchResult]:
        """基于内容质量筛选"""
        filtered = []

        for result in results:
            # 长度检查
            if len(result.content) < self.min_content_length:
                continue

            # 内容质量检查
            quality_score = self._calculate_quality_score(result.content)
            if quality_score < 0.3:  # 质量阈值
                continue

            result.metadata["quality_score"] = quality_score
            filtered.append(result)

        logger.info(f"Quality filter: {len(results)} -> {len(filtered)}")
        return filtered

    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """智能去重"""
        if not results:
            return results

        deduplicated = [results[0]]  # 保留第一个（通常分数最高）

        for result in results[1:]:
            is_duplicate = False

            for existing in deduplicated:
                similarity = self._content_similarity(result.content, existing.content)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    # 保留分数更高的
                    if result.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    break

            if not is_duplicate:
                deduplicated.append(result)

        logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)}")
        return deduplicated

    def _rank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """多维度排序 - 简化版"""

        for result in results:
            # 基础分数
            base_score = result.score

            # 查询相关性奖励
            relevance_bonus = self._calculate_relevance_bonus(query, result.content)

            # 内容完整性奖励
            completeness_bonus = self._calculate_completeness_bonus(result.content)

            # 来源可信度
            source_score = self._calculate_source_score(result.source)

            # 综合评分
            final_score = (
                    base_score * 0.5 +  # 检索相关性
                    relevance_bonus * 0.2 +  # 查询匹配度
                    completeness_bonus * 0.2 +  # 内容完整性
                    source_score * 0.1  # 来源权重
            )

            result.metadata["final_score"] = final_score
            result.metadata["ranking_components"] = {
                "base_score": base_score,
                "relevance_bonus": relevance_bonus,
                "completeness_bonus": completeness_bonus,
                "source_score": source_score
            }

        return sorted(results, key=lambda x: x.metadata["final_score"], reverse=True)

    def _optimize_content_length(self, results: List[SearchResult]) -> List[SearchResult]:
        """优化单个文档长度"""
        optimized = []

        for result in results:
            content = result.content

            # 如果内容过长，进行智能截断
            if len(content) > self.max_content_length:
                # 保留开头和结尾，中间用省略号
                keep_start = self.max_content_length // 2
                keep_end = self.max_content_length // 4

                optimized_content = (
                        content[:keep_start] +
                        "\n...[内容已优化截断]...\n" +
                        content[-keep_end:]
                )

                result.content = optimized_content
                result.metadata["content_optimized"] = True

            optimized.append(result)

        return optimized

    # 保留原有的辅助方法
    def _calculate_quality_score(self, content: str) -> float:
        """计算内容质量分数"""
        score = 0.5

        # 长度合理性
        length = len(content)
        if 200 <= length <= 1500:
            score += 0.2

        # 结构完整性
        sentences = len(re.findall(r'[。！？.!?]', content))
        if sentences >= 2:
            score += 0.2

        # 信息密度
        unique_chars = len(set(content))
        if unique_chars / len(content) > 0.3:
            score += 0.1

        return max(0, min(1, score))

    def _calculate_relevance_bonus(self, query: str, content: str) -> float:
        """计算查询相关性"""
        query_words = set(re.findall(r'\w+', query.lower()))
        content_words = set(re.findall(r'\w+', content.lower()))

        if not query_words:
            return 0

        overlap = len(query_words & content_words)
        return min(overlap / len(query_words), 0.5)

    def _calculate_completeness_bonus(self, content: str) -> float:
        """内容完整性评分"""
        # 检查是否有完整的句子结构
        complete_sentences = len(re.findall(r'[。！？.!?]$', content.strip()))
        has_structure = bool(re.search(r'[：:\n]', content))

        score = 0.5
        if complete_sentences > 0:
            score += 0.3
        if has_structure:
            score += 0.2

        return min(score, 1.0)

    def _calculate_source_score(self, source: str) -> float:
        """来源可信度评分"""
        source_lower = source.lower()

        if 'official' in source_lower or 'doc' in source_lower:
            return 0.9
        elif 'wiki' in source_lower:
            return 0.8
        elif 'blog' in source_lower:
            return 0.6
        else:
            return 0.7

    def _content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度（简化版）"""
        # 使用简单的字符级相似度
        set1 = set(content1[:200].lower())  # 只比较前200字符
        set2 = set(content2[:200].lower())

        if not set1 or not set2:
            return 0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0


# 全局实例 - 可配置开关
context_manager = ContextManager(
    max_results=config.MAX_CONTEXT_RESULTS,
    enable_quality_filter=config.ENABLE_QUALITY_FILTER,
    enable_deduplication=config.ENABLE_DEDUPLICATION
) if config.ENABLE_CONTEXT_OPTIMIZATION else None
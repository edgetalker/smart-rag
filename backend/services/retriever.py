"""
检索模块 (Retriever):
负责接受查询文本，提供多种检索方式，

1. VectRetriever 向量检索器
2. FullTextRetriever 全文检索器
3. ResultsFusion 智能结果融合
4. HybridRetriever 智能路由

设计融合异构搜索结果算法：RRF（6：4）
根据查询中的关键字来判断使用哪种检索方法
"""

import asyncio
from typing import List, Tuple
from qdrant_client import QdrantClient
from elasticsearch import AsyncElasticsearch
import logging

from ..models import SearchResult
from ..config import config
from ..utils.embeddings import embedding_service

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器"""

    def __init__(self):
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        self.collection_name = config.QDRANT_COLLECTION

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """向量相似性检索"""
        try:
            # 获取查询向量
            query_vector = await embedding_service.get_embedding(query)

            # 执行向量检索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vector=False
            )

            # 转换结果格式
            results = []
            for hit in search_result:
                result = SearchResult(
                    content=hit.payload.get("content", ""),
                    source=hit.payload.get("source", "unknown"),
                    score=float(hit.score),
                    metadata={
                        "id": str(hit.id),
                        "retrieval_method": "vector",
                        **hit.payload
                    }
                )
                results.append(result)

            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return []


class FullTextRetriever:
    """全文检索器"""

    def __init__(self):
        self.client = AsyncElasticsearch([
            f"http://{config.ES_HOST}:{config.ES_PORT}"
        ])
        self.index_name = config.ES_INDEX

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """全文检索"""
        try:
            # 构建ES查询
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "title", "metadata.tags"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k,
                "_source": ["content", "source", "metadata"]
            }

            # 执行搜索
            response = await self.client.search(
                index=self.index_name,
                body=es_query
            )

            # 转换结果格式
            results = []
            for hit in response['hits']['hits']:
                source_data = hit['_source']
                result = SearchResult(
                    content=source_data.get("content", ""),
                    source=source_data.get("source", "unknown"),
                    score=float(hit['_score']),
                    metadata={
                        "id": hit['_id'],
                        "retrieval_method": "fulltext",
                        "es_score": hit['_score'],
                        **source_data.get("metadata", {})
                    }
                )
                results.append(result)

            logger.info(f"Fulltext search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Fulltext search error: {str(e)}")
            return []

    async def close(self):
        """关闭ES连接"""
        await self.client.close()


class ResultsFusion:
    """检索结果融合器"""

    def __init__(self):
        self.k = 60  # RRF参数

    def reciprocal_rank_fusion(
            self,
            vector_results: List[SearchResult],
            fulltext_results: List[SearchResult],
            vector_weight: float = 0.6,
            fulltext_weight: float = 0.4
    ) -> List[SearchResult]:
        """互倒排序融合算法"""

        # 创建结果映射
        all_results = {}

        # 处理向量检索结果
        for rank, result in enumerate(vector_results, 1):
            doc_id = self._get_doc_identifier(result)
            rrf_score = vector_weight / (self.k + rank)

            if doc_id not in all_results:
                all_results[doc_id] = result
                all_results[doc_id].metadata["fusion_scores"] = {}

            all_results[doc_id].metadata["fusion_scores"]["vector"] = rrf_score
            all_results[doc_id].metadata["vector_rank"] = rank

        # 处理全文检索结果
        for rank, result in enumerate(fulltext_results, 1):
            doc_id = self._get_doc_identifier(result)
            rrf_score = fulltext_weight / (self.k + rank)

            if doc_id not in all_results:
                all_results[doc_id] = result
                all_results[doc_id].metadata["fusion_scores"] = {}

            all_results[doc_id].metadata["fusion_scores"]["fulltext"] = rrf_score
            all_results[doc_id].metadata["fulltext_rank"] = rank

        # 计算融合分数
        for doc_id, result in all_results.items():
            fusion_scores = result.metadata.get("fusion_scores", {})
            total_score = sum(fusion_scores.values())
            result.score = total_score
            result.metadata["fusion_method"] = "RRF"
            result.metadata["total_fusion_score"] = total_score

        # 按融合分数排序
        fused_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )

        logger.info(
            f"Fused {len(fused_results)} unique results from vector({len(vector_results)}) and fulltext({len(fulltext_results)})")
        return fused_results

    def _get_doc_identifier(self, result: SearchResult) -> str:
        """获取文档唯一标识符"""
        # 尝试多种标识方法
        if "id" in result.metadata:
            return result.metadata["id"]

        # 使用内容和来源的哈希作为标识
        import hashlib
        content_hash = hashlib.md5(
            (result.content[:100] + result.source).encode()
        ).hexdigest()
        return content_hash


class HybridRetriever:
    """混合检索管理器"""

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.fulltext_retriever = FullTextRetriever()
        self.fusion = ResultsFusion()
        self.max_individual_retrieval = 25
    async def retrieve(
            self,
            query: str,
            strategy: str = "auto",
            top_k: int = 20,
            vector_weight: float = 0.6,
            fulltext_weight: float = 0.4
    ) -> Tuple[List[SearchResult], str]:
        """执行混合检索"""

        # 确定检索策略
        actual_strategy = self._determine_strategy(query, strategy)

        if actual_strategy == "vector":
            results = await self.vector_retriever.search(query, top_k)
            return results, "vector"

        elif actual_strategy == "fulltext":
            results = await self.fulltext_retriever.search(query, top_k)
            return results, "fulltext"

        else:  # hybrid
            # 并行执行两种检索
            individual_count = min(max(top_k, 15), 25)

            vector_task = asyncio.create_task(
                self.vector_retriever.search(query, individual_count)
            )
            fulltext_task = asyncio.create_task(
                self.fulltext_retriever.search(query, individual_count)
            )

            vector_results, fulltext_results = await asyncio.gather(
                vector_task, fulltext_task
            )

            # 融合结果
            fused_results = self.fusion.reciprocal_rank_fusion(
                vector_results,
                fulltext_results,
                vector_weight,
                fulltext_weight
            )

            # 截取top_k结果
            final_results = fused_results[:top_k]
            return final_results, "hybrid"

    def _determine_strategy(self, query: str, strategy: str) -> str:
        """智能确定检索策略"""
        if strategy != "auto":
            return strategy

        # 简单的策略判断规则
        query_lower = query.lower()

        # 代码相关查询 -> 全文检索
        if any(keyword in query_lower for keyword in ['def ', 'class ', 'import ', 'function', 'method']):
            return "fulltext"

        # 概念性查询 -> 向量检索
        if any(keyword in query_lower for keyword in ['what is', 'explain', 'concept', '理解', '解释']):
            return "vector"

        # 具体事实查询 -> 全文检索
        if any(keyword in query_lower for keyword in ['when', 'where', 'who', '什么时候', '在哪里']):
            return "fulltext"

        # 默认混合检索
        return "hybrid"

    async def close(self):
        """关闭连接"""
        await self.fulltext_retriever.close()


# 全局检索器实例
retriever = HybridRetriever()
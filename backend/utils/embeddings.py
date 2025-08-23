"""
嵌入服务模块 (Embedding Service Module)

此模块用于与 Google Generative AI (Gemini) API进行交互。它的主要功能包括：

1.  将单个或批量的文本字符串转换为高维度的语义向量（Embeddings）。
2.  提供一个工具函数来计算两个向量之间的余弦相似度，以衡量它们的语义相似性。
3.  封装了 API 密钥配置、文本预处理（清理和截断）以及错误处理逻辑。

此模块旨在为应用程序的其他部分提供一个统一、可靠的文本嵌入接口。
"""

import google.generativeai as genai
import numpy as np
from typing import List
import logging
from ..config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """嵌入服务"""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured")

        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model_name = config.GEMINI_MODEL
        logger.info("✅ Embedding service initialized")

    async def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            # 清理文本
            cleaned_text = self._clean_text(text)

            # 获取嵌入
            result = genai.embed_content(
                model=self.model_name,
                content=cleaned_text,
                task_type="retrieval_query"
            )

            return result['embedding']

        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            # 返回零向量作为fallback
            return [0.0] * config.EMBEDDING_DIMENSION

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)

        return embeddings

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 移除过多的换行和空格
        cleaned = ' '.join(text.split())

        # 截断过长文本
        max_length = 8000  # Gemini embedding的大概限制
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."

        return cleaned

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception:
            return 0.0


# 全局嵌入服务实例
embedding_service = EmbeddingService()
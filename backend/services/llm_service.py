"""
LLM Service Module
构建prompt，调用Gemini模型生成结构化输出
多种生成回到的方式
1.异步生成
2.流式生成
"""
import re
import google.generativeai as genai
from typing import List, Optional, AsyncGenerator, Dict
import asyncio
import logging
from ..models import SearchResult
from ..config import config

logger = logging.getLogger(__name__)


class LLMService:
    """LLM服务管理器"""

    def __init__(self):
        self.setup_gemini()

    def setup_gemini(self):
        """初始化Gemini"""
        if not config.GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured")

        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("✅ Gemini model initialized")

    def build_prompt(
            self,
            query: str,
            context_results: List[SearchResult],
            conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """构建提示词"""

        # prompt
        system_prompt = system_prompt = """You are a professional AI assistant specialized in answering questions based on provided English documentation. 

**INSTRUCTIONS:**
1. **Source Accuracy**: Answer STRICTLY based on the provided context documents
2. **Language Handling**: 
   - Documents are in English
   - If user queries in Chinese, understand the intent and answer appropriately
   - Maintain technical accuracy regardless of query language
3. **Response Quality**:
   - Provide comprehensive, well-structured answers
   - Include specific details and examples from the documents
   - Cite sources using format [Source: document_name]
4. **Technical Content**:
   - For code questions: provide clear code examples with explanations
   - For conceptual questions: give detailed explanations with context
   - For troubleshooting: provide step-by-step solutions
5. **Limitations**: If information is not available in the context, clearly state this limitation

**CONTEXT DOCUMENTS:**
"""

        # 构建上下文
        context_section = ""
        for i, result in enumerate(context_results, 1):
            # 清理和格式化文档内容
            cleaned_content = self._clean_content(result.content)
            context_section += f"""
Document {i}:
Source: {result.source}
Relevance Score: {result.score:.3f}
Content: {cleaned_content}

---
"""

        # 对话历史部分
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_section = "\n**CONVERSATION HISTORY:**\n"
            for item in conversation_history[-3:]:  # 保留最近3轮对话
                user_query = item.get('query', '')
                assistant_answer = item.get('answer', '')
                history_section += f"User: {user_query}\nAssistant: {assistant_answer}\n\n"
            history_section += "---\n"

        # 当前问题部分
        question_section = f"""
**CURRENT QUESTION:**
{query}

**RESPONSE:**
Based on the provided documents, here is my answer:

"""

        return system_prompt + context_section + history_section + question_section

    def _clean_content(self, content: str) -> str:
        """清理和优化文档内容"""
        if not content:
            return ""

        # 移除过多的空白字符
        cleaned = re.sub(r'\n\s*\n', '\n\n', content)
        cleaned = re.sub(r' +', ' ', cleaned)

        # 确保内容不会太长（避免token溢出）
        if len(cleaned) > 2000:
            # 智能截断：保留开头和关键部分
            lines = cleaned.split('\n')
            if len(lines) > 20:
                # 保留前15行和后5行
                truncated_lines = lines[:15] + ['[... content truncated ...]'] + lines[-5:]
                cleaned = '\n'.join(truncated_lines)
            else:
                # 简单截断
                cleaned = cleaned[:2000] + '...'

        return cleaned.strip()

    async def generate_response(
            self,
            query: str,
            context_results: List[SearchResult],
            conversation_history: Optional[List[Dict]] = None,
            stream: bool = False
    ) -> str:
        """生成回答"""
        try:
            prompt = self.build_prompt(query, context_results, conversation_history)

            if stream:
                return await self._generate_stream_response(prompt)
            else:
                return await self._generate_sync_response(prompt)

        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    async def _generate_sync_response(self, prompt: str) -> str:
        """同步生成响应"""
        try:
            # 在异步环境中运行同步的Gemini调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )

            return response.text

        except Exception as e:
            logger.error(f"Sync response generation error: {str(e)}")
            raise

    async def _generate_stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        try:
            # 注意：当前版本的genai可能不支持异步流式，这里是示例实现
            response = self.model.generate_content(prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0.01)  # 小延迟以模拟流式效果

        except Exception as e:
            logger.error(f"Stream response generation error: {str(e)}")
            yield f"流式生成错误：{str(e)}"

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            test_response = await self._generate_sync_response("Hello")
            return bool(test_response)
        except:
            return False


# 全局LLM服务实例
llm_service = LLMService()
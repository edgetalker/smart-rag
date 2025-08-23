from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# 定义数据模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")
    strategy: str = Field("auto", description="检索策略: auto/vector/fulltext/hybrid")
    include_sources: bool = Field(True, description="是否包含来源信息")


class SearchResult(BaseModel):
    """单个检索结果"""
    content: str = Field(..., description="文档内容")
    source: str = Field(..., description="来源")
    score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class RAGResponse(BaseModel):
    """RAG系统响应"""
    answer: str = Field(..., description="生成的答案")
    query: str = Field(..., description="原始查询")
    sources: List[SearchResult] = Field(..., description="参考来源")
    strategy_used: str = Field(..., description="使用的检索策略")
    response_time: float = Field(..., description="响应时间(秒)")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")


class HealthStatus(BaseModel):
    """健康检查状态"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(..., description="各服务状态")
    version: str = Field("1.0.0", description="版本号")


class SearchRequest(BaseModel):
    """独立搜索请求"""
    query: str = Field(..., min_length=1, max_length=1000)
    search_type: str = Field("hybrid", description="搜索类型: vector/fulltext/hybrid")
    top_k: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult] = Field(..., description="搜索结果")
    total_found: int = Field(..., description="找到的总数")
    search_time: float = Field(..., description="搜索耗时")
    strategy: str = Field(..., description="使用的搜索策略")


class SystemMetrics(BaseModel):
    """系统指标"""
    total_queries: int = Field(0, description="总查询数")
    avg_response_time: float = Field(0.0, description="平均响应时间")
    cache_hit_rate: float = Field(0.0, description="缓存命中率")
    active_connections: int = Field(0, description="活跃连接数")
    last_updated: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    error_code: str = Field(..., description="错误代码")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="请求ID")
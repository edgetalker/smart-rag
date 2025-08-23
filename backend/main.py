# ================================
# backend/main.py - Smart RAG System 完整实现
# ================================

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any
import uuid

# 导入本地模块
from .models import (
    QueryRequest, RAGResponse, HealthStatus,
    SearchRequest, SearchResponse, SystemMetrics, ErrorResponse
)
from .config import config
from .services.retriever import retriever
from .services.context_manager import context_manager
from .services.llm_service import llm_service

# ================================
# 日志配置
# ================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


# ================================
# 系统指标和状态管理
# ================================

class SystemMetricsManager:
    """系统指标管理器"""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_response_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": 0.0,
            "uptime_start": datetime.now(),
            "last_query_time": None,
            "active_connections": 0
        }
        self.recent_queries = []  # 存储最近的查询用于分析

    def record_query(self, success: bool, response_time: float, query: str = ""):
        """记录查询指标"""
        self.metrics["total_queries"] += 1
        self.metrics["total_response_time"] += response_time
        self.metrics["last_query_time"] = datetime.now()

        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1

        # 计算平均响应时间
        if self.metrics["total_queries"] > 0:
            self.metrics["avg_response_time"] = (
                    self.metrics["total_response_time"] / self.metrics["total_queries"]
            )

        # 记录最近查询（保留最近50个）
        self.recent_queries.append({
            "timestamp": datetime.now(),
            "query": query[:100],  # 只保存前100个字符
            "success": success,
            "response_time": response_time
        })

        if len(self.recent_queries) > 50:
            self.recent_queries = self.recent_queries[-50:]

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        uptime = (datetime.now() - self.metrics["uptime_start"]).total_seconds()
        success_rate = 0.0

        if self.metrics["total_queries"] > 0:
            success_rate = (
                                   self.metrics["successful_queries"] / self.metrics["total_queries"]
                           ) * 100

        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "success_rate_percent": success_rate,
            "queries_per_minute": (
                self.metrics["total_queries"] / (uptime / 60)
                if uptime > 0 else 0
            )
        }


# 全局指标管理器
metrics_manager = SystemMetricsManager()


# ================================
# 健康检查功能
# ================================

async def health_check_services() -> Dict[str, bool]:
    """检查所有服务的健康状态"""
    services_status = {}

    try:
        # 检查LLM服务
        llm_healthy = await llm_service.health_check()
        services_status["llm_service"] = llm_healthy
        logger.info(f"LLM Service: {'✅ Healthy' if llm_healthy else '❌ Unhealthy'}")
    except Exception as e:
        services_status["llm_service"] = False
        logger.error(f"LLM Service health check failed: {e}")

    try:
        # 检查Qdrant连接
        qdrant_healthy = retriever.vector_retriever.client.get_collections()
        services_status["qdrant"] = bool(qdrant_healthy)
        logger.info("Qdrant: ✅ Healthy")
    except Exception as e:
        services_status["qdrant"] = False
        logger.error(f"Qdrant health check failed: {e}")

    try:
        # 检查Elasticsearch连接
        es_healthy = await retriever.fulltext_retriever.client.ping()
        services_status["elasticsearch"] = es_healthy
        logger.info("Elasticsearch: ✅ Healthy")
    except Exception as e:
        services_status["elasticsearch"] = False
        logger.error(f"Elasticsearch health check failed: {e}")

    try:
        # 检查Context Manager
        services_status["context_manager"] = context_manager is not None
        logger.info("Context Manager: ✅ Available")
    except Exception as e:
        services_status["context_manager"] = False
        logger.error(f"Context Manager check failed: {e}")

    return services_status


# ================================
# 应用生命周期管理
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """最简启动 - 跳过所有检查"""
    logger.info("🚀 Smart RAG System - 快速启动模式")
    metrics_manager.metrics["uptime_start"] = datetime.now()
    logger.info("✅ 启动完成，服务就绪")

    yield  # 应用运行

    logger.info("👋 系统关闭")
"""
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("🚀 Starting Smart RAG System...")

    try:
        # 启动时检查所有服务
        services_health = await health_check_services()

        critical_services = ["llm_service", "qdrant", "elasticsearch"]
        for service in critical_services:
            if not services_health.get(service, False):
                logger.error(f"❌ Critical service '{service}' is not healthy!")

        logger.info("✅ Smart RAG System started successfully")

        # 记录启动指标
        metrics_manager.metrics["uptime_start"] = datetime.now()

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # 可以选择是否在启动失败时退出
        # raise

    yield  # 应用运行期间

    # 关闭时的清理工作
    logger.info("🔄 Shutting down Smart RAG System...")
    try:
        await retriever.close()
        logger.info("✅ All services closed gracefully")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")
"""



# ================================
# 创建FastAPI应用
# ================================

app = FastAPI(
    title="Smart RAG System",
    description="智能检索增强生成系统 - An Intelligent Retrieval-Augmented Generation System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# ================================
# 中间件配置
# ================================

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制特定域名
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # 增加活跃连接数
    metrics_manager.metrics["active_connections"] += 1

    logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Completed in {process_time:.2f}s - Status: {response.status_code}"
        )

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] Request failed after {process_time:.2f}s: {e}")
        raise
    finally:
        # 减少活跃连接数
        metrics_manager.metrics["active_connections"] -= 1


# ================================
# API路由实现
# ================================

@app.post("/api/v1/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """主查询端点 - 完整RAG流程"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        logger.info(
            f"[{request_id}] RAG Query: '{request.query[:100]}...' (top_k={request.top_k}, strategy={request.strategy})")

        # 参数验证
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if request.top_k > 20:
            raise HTTPException(status_code=400, detail="top_k cannot exceed 20")

        # 1. 检索阶段 - 获取候选结果
        logger.info(f"[{request_id}] Starting retrieval with strategy: {request.strategy}")

        search_results, strategy_used = await retriever.retrieve(
            query=request.query,
            strategy=request.strategy,
            top_k=20,  # 固定获取20个候选，为context_manager提供足够选择
            enable_context_optimization=True
        )

        logger.info(f"[{request_id}] Retrieved {len(search_results)} candidates using {strategy_used}")

        # 2. 上下文优化阶段
        final_results = search_results
        optimization_info = {"optimization_applied": False}

        if context_manager and len(search_results) > request.top_k:
            logger.info(f"[{request_id}] Starting context optimization")

            context_result = context_manager.optimize_context(
                request.query,
                search_results
            )

            optimized_results = context_result["optimized_results"]
            optimization_info = context_result["optimization_info"]

            logger.info(
                f"[{request_id}] Context optimization: "
                f"{optimization_info['original_count']} → {optimization_info['final_count']} "
                f"(quality_filter: {optimization_info.get('after_quality_filter', 'N/A')}, "
                f"deduplication: {optimization_info.get('after_deduplication', 'N/A')})"
            )

            # 确保不超过用户要求的数量
            final_results = optimized_results[:request.top_k]
        else:
            # 不使用上下文优化或结果已经足够少
            final_results = search_results[:request.top_k]
            logger.info(f"[{request_id}] Skipping context optimization, using top {request.top_k} results")

        # 3. LLM生成阶段
        logger.info(f"[{request_id}] Generating response with {len(final_results)} context documents")

        answer = await llm_service.generate_response(
            query=request.query,
            context_results=final_results,
            use_smart_prompts=True  # 启用智能提示词选择
        )

        # 4. 构建响应
        response_time = time.time() - start_time

        response = RAGResponse(
            answer=answer,
            query=request.query,
            sources=final_results if request.include_sources else [],
            strategy_used=strategy_used,
            response_time=response_time,
            timestamp=datetime.now()
        )

        # 记录成功指标
        metrics_manager.record_query(
            success=True,
            response_time=response_time,
            query=request.query
        )

        logger.info(f"[{request_id}] RAG query completed successfully in {response_time:.2f}s")
        return response

    except HTTPException:
        # 重新抛出HTTP异常
        response_time = time.time() - start_time
        metrics_manager.record_query(success=False, response_time=response_time, query=request.query)
        raise

    except Exception as e:
        response_time = time.time() - start_time
        metrics_manager.record_query(success=False, response_time=response_time, query=request.query)

        logger.error(f"[{request_id}] RAG query failed after {response_time:.2f}s: {str(e)}")
        logger.error(f"[{request_id}] Error traceback: {traceback.format_exc()}")

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during query processing: {str(e)}"
        )


@app.post("/api/v1/search/vector", response_model=SearchResponse)
async def search_vector(request: SearchRequest):
    """纯向量检索端点"""
    start_time = time.time()

    try:
        results = await retriever.vector_retriever.search(
            query=request.query,
            top_k=request.top_k
        )

        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_found=len(results),
            search_time=search_time,
            strategy="vector"
        )

    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")


@app.post("/api/v1/search/fulltext", response_model=SearchResponse)
async def search_fulltext(request: SearchRequest):
    """纯全文检索端点"""
    start_time = time.time()

    try:
        results = await retriever.fulltext_retriever.search(
            query=request.query,
            top_k=request.top_k
        )

        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_found=len(results),
            search_time=search_time,
            strategy="fulltext"
        )

    except Exception as e:
        logger.error(f"Fulltext search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fulltext search error: {str(e)}")


@app.post("/api/v1/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: SearchRequest):
    """混合检索端点"""
    start_time = time.time()

    try:
        results, strategy = await retriever.retrieve(
            query=request.query,
            strategy="hybrid",
            top_k=request.top_k
        )

        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_found=len(results),
            search_time=search_time,
            strategy=strategy
        )

    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")


@app.get("/api/v1/health", response_model=HealthStatus)
async def health_check():
    """系统健康检查端点"""
    try:
        services_status = await health_check_services()

        # 判断整体状态
        all_healthy = all(services_status.values())
        status = "healthy" if all_healthy else "degraded"

        return HealthStatus(
            status=status,
            timestamp=datetime.now(),
            services=services_status,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.now(),
            services={"error": str(e)},
            version="1.0.0"
        )


@app.get("/api/v1/metrics", response_model=SystemMetrics)
async def get_metrics():
    """获取系统性能指标"""
    try:
        current_metrics = metrics_manager.get_metrics()

        return SystemMetrics(
            total_queries=current_metrics["total_queries"],
            avg_response_time=current_metrics["avg_response_time"],
            cache_hit_rate=0.0,  # TODO: 实现缓存后更新
            active_connections=current_metrics["active_connections"],
            last_updated=datetime.now()
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@app.get("/api/v1/status")
async def get_system_status():
    """获取详细系统状态（调试用）"""
    try:
        return {
            "system": "Smart RAG System",
            "version": "1.0.0",
            "timestamp": datetime.now(),
            "config": {
                "host": config.HOST,
                "port": config.PORT,
                "debug": config.DEBUG,
                "max_context_length": config.MAX_CONTEXT_LENGTH,
                "default_top_k": config.DEFAULT_TOP_K
            },
            "metrics": metrics_manager.get_metrics(),
            "recent_queries": len(metrics_manager.recent_queries)
        }

    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


# ================================
# 异常处理
# ================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "error_code": "NOT_FOUND",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """500错误处理"""
    logger.error(f"Internal server error on {request.url.path}: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


# ================================
# 根路径
# ================================

@app.get("/")
async def root():
    """根路径欢迎信息"""
    return {
        "message": "Welcome to Smart RAG System",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
        "metrics": "/api/v1/metrics"
    }


# ================================
# 应用启动信息
# ================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting Smart RAG System on {config.HOST}:{config.PORT}")
    logger.info(f"📚 API Documentation: http://{config.HOST}:{config.PORT}/api/docs")
    logger.info(f"🏥 Health Check: http://{config.HOST}:{config.PORT}/api/v1/health")

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        access_log=True,
        log_level="info"
    )
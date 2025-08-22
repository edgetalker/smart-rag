import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# 导入依赖包
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from elasticsearch import Elasticsearch

# ============================================================================
# 配置部分
# ============================================================================
load_dotenv()

# 基础配置
DATASET_PATH = os.getenv('DATASET_PATH')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# LangChain分割器配置
CHUNK_SIZE = os.getenv('CHUNK_SIZE')
CHUNK_OVERLAP = os.getenv('CHUNK_OVERLAP')

# Gemini配置
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

# Qdrant配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Elasticsearch配置
ES_HOSTS = ["http://localhost:9200"]
ES_INDEX = os.getenv("ES_INDEX")

# 处理配置
MAX_DOCUMENTS = None
SKIP_EMPTY_DOCS = True
MIN_CONTENT_LENGTH = 50

# 内容限制配置
MAX_CONTENT_LENGTH = 10000
MAX_TITLE_LENGTH = 200


@dataclass
class ProcessedChunk:
    """处理后的文档块"""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


# ============================================================================
# 工具函数
# ============================================================================

def setup_gemini_api():
    """设置Gemini API"""
    if not GEMINI_API_KEY:
        print("❌ 错误: 请设置GEMINI_API_KEY环境变量")
        print("   获取API Key: https://ai.google.dev/")
        exit(1)

    genai.configure(api_key=GEMINI_API_KEY)

    # 测试连接
    try:
        result = genai.embed_content(
            model=GEMINI_MODEL,
            content="测试连接",
            task_type="retrieval_document"
        )
        if result and 'embedding' in result:
            print("✅ Gemini API连接成功")
            return True
    except Exception as e:
        print(f"❌ Gemini API连接失败: {e}")
        return False


def load_documents():
    """加载文档数据集"""
    dataset_file = Path(DATASET_PATH)

    if not dataset_file.exists():
        print(f"❌ 数据集文件不存在: {DATASET_PATH}")
        print("   请先运行数据集构建脚本")
        exit(1)

    print(f"📚 加载数据集: {DATASET_PATH}")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # 数据过滤
    filtered_docs = []
    for doc in documents:
        # 跳过空内容
        if SKIP_EMPTY_DOCS and not doc.get('content', '').strip():
            continue

        # 检查最小长度
        if len(doc.get('content', '')) < MIN_CONTENT_LENGTH:
            continue

        filtered_docs.append(doc)

    # 限制数量（用于测试）
    if MAX_DOCUMENTS:
        filtered_docs = filtered_docs[:MAX_DOCUMENTS]

    print(f"📊 加载完成: {len(filtered_docs)} 个文档（原始: {len(documents)}）")
    return filtered_docs


def split_documents(documents):
    """使用LangChain分割文档"""
    print("📄 开始文档分割...")

    # 初始化LangChain分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )

    all_chunks = []

    for doc in tqdm(documents, desc="分割文档"):
        doc_id = doc.get('id', 'unknown')
        content = doc.get('content', '')

        if not content.strip():
            continue

        try:
            # 使用LangChain分割
            text_chunks = text_splitter.split_text(content)

            # 为每个块创建元数据
            for i, chunk_content in enumerate(text_chunks):
                chunk_id = f"{doc_id}_chunk_{i}_{hashlib.md5(chunk_content.encode()).hexdigest()[:8]}"

                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'source_id': doc_id,
                    'source': doc.get('source', 'unknown'),
                    'title': doc.get('title', ''),
                    'chunk_index': i,
                    'chunk_length': len(chunk_content),
                    'original_doc_length': len(content),
                    'has_code_blocks': len(doc.get('code_blocks', [])) > 0,
                    'created_at': datetime.now().isoformat()
                }

                chunk_data = {
                    'chunk_id': chunk_id,
                    'content': chunk_content,
                    'metadata': chunk_metadata
                }

                all_chunks.append(chunk_data)

        except Exception as e:
            print(f"⚠️ 文档 {doc_id} 分割失败: {e}")
            continue

    print(f"✅ 分割完成: {len(documents)} 文档 -> {len(all_chunks)} 块")
    return all_chunks


def generate_embeddings(chunks):
    """使用Gemini生成嵌入向量"""
    print("🧠 开始生成嵌入向量...")

    embeddings = []
    failed_count = 0

    # 批量处理
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="生成嵌入"):
        batch_chunks = chunks[i:i + BATCH_SIZE]

        for chunk in batch_chunks:
            try:
                # 调用Gemini API
                result = genai.embed_content(
                    model=GEMINI_MODEL,
                    content=chunk['content'],
                    task_type="retrieval_document"
                )

                if result and 'embedding' in result:
                    embeddings.append(result['embedding'])
                else:
                    # 失败时添加零向量
                    embeddings.append([0.0] * EMBEDDING_DIMENSION)
                    failed_count += 1

            except Exception as e:
                print(f"⚠️ 嵌入生成失败: {str(e)[:50]}...")
                embeddings.append([0.0] * EMBEDDING_DIMENSION)
                failed_count += 1

            # 避免API频率限制
            time.sleep(0.1)

        # 批次间休息
        if i > 0:
            time.sleep(1)

    success_count = len(embeddings) - failed_count
    print(f"✅ 嵌入生成完成: {success_count}/{len(chunks)} 成功")

    if failed_count > 0:
        print(f"⚠️ 失败数量: {failed_count}")

    return embeddings


def create_processed_chunks(chunks, embeddings):
    """组合文档块和嵌入向量"""
    print("🔧 组合文档块和嵌入向量...")

    processed_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        processed_chunk = ProcessedChunk(
            chunk_id=chunk['chunk_id'],
            content=chunk['content'],
            embedding=embedding,
            metadata=chunk['metadata']
        )
        processed_chunks.append(processed_chunk)

    print(f"✅ 组合完成: {len(processed_chunks)} 个处理块")
    return processed_chunks


def setup_qdrant_v17():
    """Qdrant 1.7版本专用设置"""
    print("🗄️ 设置Qdrant向量数据库...")

    try:
        # 1.7版本的简化连接（移除不兼容参数）
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # 检查连接
        try:
            collections = client.get_collections()
            print(f"✅ Qdrant连接成功，当前集合数: {len(collections.collections)}")
        except Exception as e:
            print(f"⚠️ 连接测试警告: {e}")

        # 检查集合是否存在
        collection_names = [col.name for col in collections.collections]

        if QDRANT_COLLECTION in collection_names:
            print(f"⚠️ 集合 {QDRANT_COLLECTION} 已存在，将删除重建")
            client.delete_collection(QDRANT_COLLECTION)

        # 创建新集合
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )

        print(f"✅ Qdrant集合创建成功: {QDRANT_COLLECTION}")
        return client

    except Exception as e:
        print(f"❌ Qdrant设置失败: {e}")
        print("💡 请确保Qdrant服务正在运行: docker run -p 6333:6333 qdrant/qdrant")
        return None


def generate_safe_id(chunk_id: str, use_numeric: bool = False):
    """生成安全的ID"""
    if use_numeric:
        # 数字ID方案
        return hash(chunk_id) % (2 ** 31)
    else:
        # 字符串ID方案 - 清理特殊字符
        safe_id = chunk_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
        # 限制长度
        if len(safe_id) > 100:
            safe_id = safe_id[:90] + "_" + hashlib.md5(safe_id.encode()).hexdigest()[:8]
        return safe_id


def build_qdrant_index_v17(client, processed_chunks):
    """Qdrant 1.7版本专用索引构建"""
    print("🔄 构建Qdrant向量索引（1.7版本兼容）...")

    if not client:
        print("❌ Qdrant客户端未初始化")
        return False

    # 首先尝试字符串ID方案
    success = try_string_id_upload(client, processed_chunks)

    if not success:
        print("🔧 字符串ID失败，尝试数字ID方案...")
        success = try_numeric_id_upload(client, processed_chunks)

    return success


def try_string_id_upload(client, processed_chunks):
    """尝试字符串ID上传"""
    print("🔧 尝试字符串ID方案...")

    try:
        points = []

        # 准备前10个点进行测试
        test_chunks = processed_chunks[:10]

        for chunk in test_chunks:
            safe_id = generate_safe_id(chunk.chunk_id, use_numeric=False)
            content = chunk.content[:MAX_CONTENT_LENGTH]
            title = chunk.metadata.get("title", "")[:MAX_TITLE_LENGTH]

            point = PointStruct(
                id=safe_id,
                vector=chunk.embedding,
                payload={
                    "content": content,
                    "source_id": chunk.metadata["source_id"],
                    "source": chunk.metadata["source"],
                    "title": title,
                    "chunk_index": chunk.metadata["chunk_index"],
                    "chunk_length": chunk.metadata["chunk_length"],
                    "has_code_blocks": chunk.metadata["has_code_blocks"],
                    "created_at": chunk.metadata["created_at"]
                }
            )
            points.append(point)

        # 测试上传
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

        print("✅ 字符串ID测试成功，继续全量上传...")

        # 全量上传
        return upload_all_chunks_string_id(client, processed_chunks)

    except Exception as e:
        print(f"⚠️ 字符串ID方案失败: {e}")
        return False


def upload_all_chunks_string_id(client, processed_chunks):
    """字符串ID全量上传"""
    try:
        points = []
        failed_count = 0

        for chunk in tqdm(processed_chunks, desc="准备字符串ID数据"):
            try:
                safe_id = generate_safe_id(chunk.chunk_id, use_numeric=False)
                content = chunk.content[:MAX_CONTENT_LENGTH]
                title = chunk.metadata.get("title", "")[:MAX_TITLE_LENGTH]

                point = PointStruct(
                    id=safe_id,
                    vector=chunk.embedding,
                    payload={
                        "original_chunk_id": chunk.chunk_id,
                        "content": content,
                        "source_id": chunk.metadata["source_id"],
                        "source": chunk.metadata["source"],
                        "title": title,
                        "chunk_index": chunk.metadata["chunk_index"],
                        "chunk_length": chunk.metadata["chunk_length"],
                        "has_code_blocks": chunk.metadata["has_code_blocks"],
                        "created_at": chunk.metadata["created_at"]
                    }
                )
                points.append(point)

            except Exception as e:
                failed_count += 1
                continue

        # 分批上传
        batch_size = 20
        success_count = 0

        for i in tqdm(range(0, len(points), batch_size), desc="上传字符串ID"):
            batch_points = points[i:i + batch_size]

            try:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=batch_points
                )
                success_count += len(batch_points)

            except Exception as e:
                print(f"⚠️ 批次 {i // batch_size + 1} 失败: {e}")
                continue

        success_rate = (success_count / len(points) * 100) if len(points) > 0 else 0
        print(f"✅ 字符串ID上传完成: {success_count}/{len(points)} ({success_rate:.1f}%)")

        return success_rate > 90

    except Exception as e:
        print(f"❌ 字符串ID全量上传失败: {e}")
        return False


def try_numeric_id_upload(client, processed_chunks):
    """尝试数字ID上传"""
    print("🔧 使用数字ID方案...")

    try:
        points = []
        id_mapping = {}

        for chunk in tqdm(processed_chunks, desc="准备数字ID数据"):
            try:
                numeric_id = generate_safe_id(chunk.chunk_id, use_numeric=True)
                id_mapping[str(numeric_id)] = chunk.chunk_id

                content = chunk.content[:MAX_CONTENT_LENGTH]
                title = chunk.metadata.get("title", "")[:MAX_TITLE_LENGTH]

                point = PointStruct(
                    id=numeric_id,
                    vector=chunk.embedding,
                    payload={
                        "original_chunk_id": chunk.chunk_id,
                        "content": content,
                        "source_id": chunk.metadata["source_id"],
                        "source": chunk.metadata["source"],
                        "title": title,
                        "chunk_index": chunk.metadata["chunk_index"],
                        "chunk_length": chunk.metadata["chunk_length"],
                        "has_code_blocks": chunk.metadata["has_code_blocks"],
                        "created_at": chunk.metadata["created_at"]
                    }
                )
                points.append(point)

            except Exception as e:
                continue

        # 保存ID映射
        mapping_file = Path("data/qdrant_id_mapping.json")
        mapping_file.parent.mkdir(exist_ok=True)
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(id_mapping, f, ensure_ascii=False, indent=2)
        print(f"💾 ID映射已保存: {mapping_file}")

        # 分批上传
        batch_size = 20
        success_count = 0

        for i in tqdm(range(0, len(points), batch_size), desc="上传数字ID"):
            batch_points = points[i:i + batch_size]

            try:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=batch_points
                )
                success_count += len(batch_points)

            except Exception as e:
                print(f"⚠️ 数字ID批次 {i // batch_size + 1} 失败: {e}")
                continue

        success_rate = (success_count / len(points) * 100) if len(points) > 0 else 0
        print(f"✅ 数字ID上传完成: {success_count}/{len(points)} ({success_rate:.1f}%)")

        return success_rate > 90

    except Exception as e:
        print(f"❌ 数字ID方案也失败: {e}")
        return False


def setup_elasticsearch_fixed():
    """修复版本的Elasticsearch设置"""
    print("🔍 设置Elasticsearch全文索引")

    try:
        # 连接Elasticsearch
        es = Elasticsearch(ES_HOSTS)

        # 检查连接
        if not es.ping():
            print("❌ 无法连接到Elasticsearch")
            return None

        # 删除现有索引
        if es.indices.exists(index=ES_INDEX):
            print(f"⚠️ 索引 {ES_INDEX} 已存在，将删除重建")
            es.indices.delete(index=ES_INDEX)

        # 修复的索引映射
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "index_options": "docs"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "source": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "chunk_length": {"type": "integer"},
                    "has_code_blocks": {"type": "boolean"},
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time_nanos"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "max_result_window": 10000
            }
        }

        # 创建索引
        es.indices.create(index=ES_INDEX, body=mapping)
        print(f"✅ Elasticsearch索引创建成功: {ES_INDEX}")
        return es

    except Exception as e:
        print(f"❌ Elasticsearch设置失败: {e}")
        return None


def build_elasticsearch_index_fixed(es, processed_chunks):
    """修复版本的Elasticsearch索引构建"""
    print("🔄 构建Elasticsearch全文索引（增强版）...")

    if not es:
        print("❌ Elasticsearch客户端未初始化")
        return False

    try:
        from elasticsearch.helpers import bulk

        actions = []
        failed_docs = []

        for chunk in processed_chunks:
            try:
                # 清理和验证数据
                safe_id = chunk.chunk_id.replace('/', '_').replace('\\', '_')
                content = chunk.content[:32000] if len(chunk.content) > 32000 else chunk.content
                title = chunk.metadata.get("title", "")[:500]

                # 确保created_at格式正确
                created_at = chunk.metadata["created_at"]
                if isinstance(created_at, (int, float)):
                    created_at = datetime.fromtimestamp(created_at).isoformat()

                doc = {
                    "_index": ES_INDEX,
                    "_id": safe_id,
                    "_source": {
                        "content": content,
                        "source_id": chunk.metadata["source_id"],
                        "source": chunk.metadata["source"],
                        "title": title,
                        "chunk_index": chunk.metadata["chunk_index"],
                        "chunk_length": min(chunk.metadata["chunk_length"], 32000),
                        "has_code_blocks": chunk.metadata["has_code_blocks"],
                        "created_at": created_at
                    }
                }
                actions.append(doc)

            except Exception as e:
                failed_docs.append(chunk.chunk_id)
                continue

        print(f"📊 准备索引: {len(actions)} 个文档, {len(failed_docs)} 个预处理失败")

        # 批量索引
        try:
            success, failed = bulk(es, actions, stats_only=True)
            print(f"✅ ES索引: {success} 成功, {len(failed_docs)} 预处理失败")

            # 刷新索引
            es.indices.refresh(index=ES_INDEX)
            return True

        except Exception as e:
            print(f"❌ 批量索引失败: {e}")
            return False

    except Exception as e:
        print(f"❌ Elasticsearch索引构建失败: {e}")
        return False


def save_processing_results(processed_chunks):
    """保存处理结果"""
    print("💾 保存处理结果...")

    # 创建输出目录
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # 保存处理后的数据
    output_data = []
    for chunk in processed_chunks:
        chunk_data = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata
        }
        output_data.append(chunk_data)

    output_file = output_dir / "processed_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ 处理结果已保存: {output_file} ({file_size:.2f}MB)")


def print_summary(processed_chunks, qdrant_success, es_success, total_time):
    """打印处理摘要"""
    print("\n" + "=" * 60)
    print("📊 Smart RAG 索引构建摘要")
    print("=" * 60)

    # 基础统计
    total_chunks = len(processed_chunks)
    total_chars = sum(len(chunk.content) for chunk in processed_chunks)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

    # 源分布
    source_dist = {}
    for chunk in processed_chunks:
        source = chunk.metadata["source"]
        source_dist[source] = source_dist.get(source, 0) + 1

    print(f"📄 处理文档块: {total_chunks}")
    print(f"📊 总字符数: {total_chars:,}")
    print(f"📏 平均块大小: {avg_chunk_size:.0f} 字符")
    print(f"⏱️ 总处理时间: {total_time:.2f} 秒")

    print(f"\n📋 数据源分布:")
    for source, count in source_dist.items():
        percentage = count / total_chunks * 100
        print(f"  {source}: {count} ({percentage:.1f}%)")

    print(f"\n🏗️ 索引构建状态:")
    qdrant_status = "✅ 成功" if qdrant_success else "❌ 失败"
    es_status = "✅ 成功" if es_success else "❌ 失败"
    print(f"  Qdrant向量索引: {qdrant_status}")
    print(f"  Elasticsearch全文索引: {es_status}")

    if qdrant_success and es_success:
        print(f"\n🎉 双重索引构建完成！RAG系统已准备就绪。")
    elif qdrant_success or es_success:
        print(f"\n⚠️ 部分索引构建成功，请检查失败的服务。")
    else:
        print(f"\n❌ 索引构建失败，请检查服务配置。")

    print("=" * 60)


def main():
    """主函数"""
    start_time = time.time()

    print("🚀 Smart RAG MVP - 一体化索引构建（Qdrant 1.7兼容版）")
    print("=" * 60)

    # 步骤1: 设置Gemini API
    if not setup_gemini_api():
        exit(1)

    processed_chunks_file = Path("data/processed_chunks.json")
    processed_chunks = []

    if processed_chunks_file.exists():
        print(f"✅ 发现已存在的处理文件: {processed_chunks_file}，跳过嵌入生成步骤。")
        with open(processed_chunks_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # 从加载的数据重建ProcessedChunk对象，并修正时间格式
        for chunk_data in loaded_data:
            # 修正旧文件中的时间戳格式
            ts = chunk_data["metadata"]["created_at"]
            if isinstance(ts, (int, float)):
                chunk_data["metadata"]["created_at"] = datetime.fromtimestamp(ts).isoformat()

            processed_chunks.append(ProcessedChunk(**chunk_data))
        print(f"🔧 已从文件加载并修正 {len(processed_chunks)} 个处理块。")

    else:
        print("ℹ️ 未发现已处理文件，将执行完整处理流程。")
        # 步骤2: 加载文档
        documents = load_documents()
        if not documents:
            print("❌ 没有文档可处理")
            exit(1)

        # 步骤3: 文档分割
        chunks = split_documents(documents)
        if not chunks:
            print("❌ 文档分割失败")
            exit(1)

        # 步骤4: 生成嵌入
        embeddings = generate_embeddings(chunks)
        if len(embeddings) != len(chunks):
            print("❌ 嵌入生成数量不匹配")
            exit(1)

        # 步骤5: 组合处理结果
        processed_chunks = create_processed_chunks(chunks, embeddings)

    # 步骤6: 设置数据库
    qdrant_client = setup_qdrant_v17()
    es_client = setup_elasticsearch_fixed()

    # 步骤7: 构建索引
    qdrant_success = build_qdrant_index_v17(qdrant_client, processed_chunks)
    es_success = build_elasticsearch_index_fixed(es_client, processed_chunks)

    # 步骤8: 保存结果
    save_processing_results(processed_chunks)

    # 步骤9: 显示摘要
    total_time = time.time() - start_time
    print_summary(processed_chunks, qdrant_success, es_success, total_time)


if __name__ == "__main__":
    main()
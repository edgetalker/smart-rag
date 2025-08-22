#!/usr/bin/env python3
"""
Smart RAG 数据集构建器
使用Git Clone方式获取开源项目的Markdown文档
"""

import os
import subprocess
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import time
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartRAGDatasetBuilder:
    """Smart RAG项目数据集构建器"""

    def __init__(self, config_path: str = "configs/data_sources.yaml"):
        """初始化数据集构建器"""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # 设置工作目录
        self.raw_data_dir = Path("raw_data")
        self.processed_data_dir = Path("processed_data")
        self.documents_dir = self.processed_data_dir / "documents"
        self.metadata_dir = self.processed_data_dir / "metadata"
        self.reports_dir = self.processed_data_dir / "quality_reports"

        # 创建目录结构
        self._setup_directories()

        # 统计信息
        self.stats = {
            "total_repos_cloned": 0,
            "total_files_found": 0,
            "total_files_processed": 0,
            "total_files_valid": 0,
            "processing_errors": 0,
            "start_time": None,
            "end_time": None
        }

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise

    def _setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.documents_dir,
            self.metadata_dir,
            self.reports_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 创建目录: {directory}")

    def clone_repository(self, source_name: str, source_config: Dict) -> Path:
        """克隆Git仓库"""
        repo_url = source_config["repo_url"]
        repo_dir = self.raw_data_dir / f"{source_name}_repo"

        logger.info(f"🔄 开始克隆 {source_name} 仓库...")
        logger.info(f"📍 URL: {repo_url}")

        # 如果目录已存在，先删除
        if repo_dir.exists():
            logger.warning(f"目录已存在，删除旧版本: {repo_dir}")
            shutil.rmtree(repo_dir)

        try:
            # 使用shallow clone提高速度
            cmd = [
                "git", "clone",
                "--depth", "1",  # 只克隆最新版本
                "--single-branch",  # 只克隆默认分支
                repo_url,
                str(repo_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode != 0:
                raise Exception(f"Git clone失败: {result.stderr}")

            logger.info(f"✅ {source_name} 克隆成功")
            self.stats["total_repos_cloned"] += 1
            return repo_dir

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {source_name} 克隆超时")
            raise
        except Exception as e:
            logger.error(f"❌ {source_name} 克隆失败: {e}")
            raise

    def extract_documentation(self, source_name: str, repo_dir: Path, source_config: Dict) -> List[Path]:
        """从仓库中提取文档文件"""
        docs_path = source_config["docs_path"]
        file_patterns = source_config.get("file_patterns", ["*.md"])
        exclude_patterns = source_config.get("exclude_patterns", [])

        docs_dir = repo_dir / docs_path

        if not docs_dir.exists():
            logger.warning(f"⚠️  文档目录不存在: {docs_dir}")
            return []

        logger.info(f"📂 提取 {source_name} 文档，目录: {docs_dir}")

        # 查找所有匹配的文件
        all_files = []
        for pattern in file_patterns:
            found_files = list(docs_dir.rglob(pattern))
            all_files.extend(found_files)

        # 过滤排除的文件
        filtered_files = []
        for file_path in all_files:
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern.replace("**", "*") in str(file_path):
                    should_exclude = True
                    break

            if not should_exclude:
                filtered_files.append(file_path)

        logger.info(f"📊 {source_name} 找到文档文件: {len(filtered_files)} 个")
        self.stats["total_files_found"] += len(filtered_files)

        return filtered_files

    def process_markdown_file(self, file_path: Path, source_name: str) -> Optional[Dict]:
        """处理单个Markdown文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 基础验证
            if len(content.strip()) < self.config["processing"]["min_content_length"]:
                logger.debug(f"文件内容过短，跳过: {file_path}")
                return None

            if len(content) > self.config["processing"]["max_content_length"]:
                logger.debug(f"文件内容过长，跳过: {file_path}")
                return None

            # 提取文档信息
            document_info = self._extract_document_info(content, file_path, source_name)

            self.stats["total_files_processed"] += 1
            return document_info

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            self.stats["processing_errors"] += 1
            return None

    def _extract_document_info(self, content: str, file_path: Path, source_name: str) -> Dict:
        """提取文档的详细信息"""
        import frontmatter
        import re

        # 解析frontmatter
        try:
            post = frontmatter.loads(content)
            metadata = post.metadata
            main_content = post.content
        except:
            metadata = {}
            main_content = content

        # 提取标题结构
        headings = self._extract_headings(main_content)

        # 提取代码块
        code_blocks = self._extract_code_blocks(main_content)

        # 提取链接
        links = self._extract_links(main_content)

        # 计算统计信息
        stats = self._calculate_content_stats(main_content, code_blocks)

        # 生成相对路径（用作文档ID）
        relative_path = str(file_path).replace(str(self.raw_data_dir), "")

        document_info = {
            "id": f"{source_name}_{hash(relative_path) % 1000000}",
            "source": source_name,
            "file_path": str(file_path),
            "relative_path": relative_path,
            "title": self._extract_title(headings, metadata),
            "metadata": metadata,
            "content": main_content,
            "headings": headings,
            "code_blocks": code_blocks,
            "links": links,
            "stats": stats,
            "created_at": time.time()
        }

        return document_info

    def _extract_headings(self, content: str) -> List[Dict]:
        """提取标题结构"""
        import re

        heading_pattern = r'^(#{1,6})\s+(.+)$'
        headings = []

        for i, line in enumerate(content.split('\n')):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append({
                    "level": level,
                    "text": text,
                    "line_number": i + 1
                })

        return headings

    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """提取代码块"""
        import re

        # 匹配代码块 ```language ... ```
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        code_blocks = []

        for match in re.finditer(code_pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()

            code_blocks.append({
                "language": language,
                "code": code,
                "start_pos": match.start(),
                "end_pos": match.end()
            })

        return code_blocks

    def _extract_links(self, content: str) -> List[Dict]:
        """提取链接"""
        import re

        # 匹配Markdown链接 [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = []

        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)

            links.append({
                "text": text,
                "url": url,
                "is_internal": not url.startswith(("http://", "https://"))
            })

        return links

    def _calculate_content_stats(self, content: str, code_blocks: List[Dict]) -> Dict:
        """计算内容统计信息"""
        lines = content.split('\n')

        # 计算代码行数
        code_lines = sum(len(block["code"].split('\n')) for block in code_blocks)

        # 计算文本行数（非空行）
        text_lines = len([line for line in lines if line.strip()])

        stats = {
            "total_lines": len(lines),
            "text_lines": text_lines,
            "code_lines": code_lines,
            "code_blocks_count": len(code_blocks),
            "character_count": len(content),
            "word_count": len(content.split()),
            "code_ratio": code_lines / max(text_lines, 1)
        }

        return stats

    def _safe_cleanup(self, directory: Path):
        """安全清理目录（解决Windows权限问题）"""
        import stat

        def handle_remove_readonly(func, path, exc):
            """处理只读文件的删除"""
            if os.path.exists(path):
                # 移除只读属性
                os.chmod(path, stat.S_IWRITE)
                # 重新尝试删除
                func(path)

        try:
            if directory.exists():
                # Windows特殊处理：移除只读属性
                if os.name == 'nt':  # Windows系统
                    for root, dirs, files in os.walk(directory):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            try:
                                os.chmod(dir_path, stat.S_IWRITE)
                            except:
                                pass
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            try:
                                os.chmod(file_path, stat.S_IWRITE)
                            except:
                                pass

                    # 使用shutil.rmtree的onerror回调处理剩余的权限问题
                    shutil.rmtree(directory, onerror=handle_remove_readonly)
                else:
                    # Linux/Mac使用标准删除
                    shutil.rmtree(directory)

                logger.info(f"✅ 成功清理目录: {directory}")
        except Exception as e:
            logger.warning(f"⚠️  清理目录失败 {directory}: {e}")
            logger.warning("💡 清理失败不影响数据处理结果，可手动删除raw_data目录")

    def _extract_title(self, headings: List[Dict], metadata: Dict) -> str:
        """提取文档标题"""
        # 优先使用metadata中的title
        if "title" in metadata:
            return metadata["title"]

        # 使用第一个一级标题
        for heading in headings:
            if heading["level"] == 1:
                return heading["text"]

        # 使用第一个标题
        if headings:
            return headings[0]["text"]

        return "Untitled"

    def save_processed_documents(self, documents: List[Dict], source_name: str):
        """保存处理后的文档"""

        # 保存完整文档数据
        documents_file = self.documents_dir / f"{source_name}_documents.json"
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        # 保存元数据摘要
        metadata_summary = []
        for doc in documents:
            summary = {
                "id": doc["id"],
                "title": doc["title"],
                "file_path": doc["relative_path"],
                "stats": doc["stats"],
                "headings_count": len(doc["headings"]),
                "code_blocks_count": len(doc["code_blocks"]),
                "links_count": len(doc["links"])
            }
            metadata_summary.append(summary)

        metadata_file = self.metadata_dir / f"{source_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 {source_name} 文档已保存: {len(documents)} 个")

    def process_data_source(self, source_name: str) -> List[Dict]:
        """处理单个数据源"""
        source_config = self.config["data_sources"][source_name]

        logger.info(f"🚀 开始处理数据源: {source_name}")
        logger.info(f"📝 描述: {source_config['description']}")

        try:
            # 1. 克隆仓库
            repo_dir = self.clone_repository(source_name, source_config)

            # 2. 提取文档文件
            doc_files = self.extract_documentation(source_name, repo_dir, source_config)

            if not doc_files:
                logger.warning(f"⚠️  {source_name} 未找到有效文档文件")
                self._safe_cleanup(repo_dir)
                return []

            # 3. 处理文档文件
            documents = []
            logger.info(f"📄 开始处理 {len(doc_files)} 个文档文件...")

            for file_path in tqdm(doc_files, desc=f"处理{source_name}文档"):
                doc_info = self.process_markdown_file(file_path, source_name)
                if doc_info:
                    documents.append(doc_info)
                    self.stats["total_files_valid"] += 1

            # 4. 保存处理结果
            if documents:
                self.save_processed_documents(documents, source_name)

            # 5. 安全清理临时文件
            logger.info(f"🧹 清理临时文件: {repo_dir}")
            self._safe_cleanup(repo_dir)

            logger.info(f"✅ {source_name} 处理完成，有效文档: {len(documents)} 个")
            return documents

        except Exception as e:
            logger.error(f"❌ {source_name} 处理失败: {e}")
            # 确保即使出错也要清理
            if 'repo_dir' in locals():
                self._safe_cleanup(repo_dir)
            return []

    def build_complete_dataset(self) -> Dict:
        """构建完整数据集"""
        self.stats["start_time"] = time.time()

        logger.info("🎯 开始构建Smart RAG数据集")
        logger.info("=" * 60)

        all_documents = []
        source_summary = {}

        # 按优先级排序处理数据源
        sources = sorted(
            self.config["data_sources"].items(),
            key=lambda x: x[1]["priority"]
        )

        for source_name, source_config in sources:
            documents = self.process_data_source(source_name)

            # 详细记录每个数据源的结果
            logger.info(f"📊 {source_name} 返回文档数量: {len(documents)}")

            if documents:
                all_documents.extend(documents)
                logger.info(f"✅ {source_name} 文档已添加到完整数据集")
            else:
                logger.warning(f"⚠️  {source_name} 没有返回有效文档")

            source_summary[source_name] = {
                "document_count": len(documents),
                "description": source_config["description"],
                "priority": source_config["priority"]
            }

        # 验证完整数据集
        logger.info(f"📊 完整数据集文档总数: {len(all_documents)}")

        if not all_documents:
            logger.error("❌ 完整数据集为空！请检查各数据源的处理结果")
            # 尝试从单独的文件加载数据
            all_documents = self._recover_from_individual_files()

        # 保存完整数据集
        complete_dataset_file = self.processed_data_dir / "complete_dataset.json"
        logger.info(f"💾 保存完整数据集到: {complete_dataset_file}")

        try:
            with open(complete_dataset_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)

            # 验证保存结果
            file_size = complete_dataset_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ 完整数据集保存成功，文件大小: {file_size:.2f}MB")

        except Exception as e:
            logger.error(f"❌ 保存完整数据集失败: {e}")
            return {"error": "Failed to save complete dataset"}

        # 生成最终报告
        self.stats["end_time"] = time.time()
        final_report = self._generate_final_report(source_summary)

        # 更新统计信息
        final_report["actual_documents_saved"] = len(all_documents)

        # 保存报告
        report_file = self.reports_dir / "build_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        logger.info("🎉 数据集构建完成！")
        logger.info("=" * 60)
        self._print_final_summary(final_report)

        return final_report

    def _recover_from_individual_files(self) -> List[Dict]:
        """从单独的数据源文件恢复完整数据集"""
        logger.info("🔄 尝试从单独文件恢复数据集...")

        recovered_documents = []
        documents_dir = self.documents_dir

        if not documents_dir.exists():
            logger.error(f"❌ 文档目录不存在: {documents_dir}")
            return []

        # 查找所有的*_documents.json文件
        for json_file in documents_dir.glob("*_documents.json"):
            logger.info(f"📄 尝试加载: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    documents = json.load(f)

                if isinstance(documents, list):
                    recovered_documents.extend(documents)
                    logger.info(f"✅ 从 {json_file.name} 恢复了 {len(documents)} 个文档")
                else:
                    logger.warning(f"⚠️  {json_file.name} 格式不正确")

            except Exception as e:
                logger.error(f"❌ 无法加载 {json_file}: {e}")

        logger.info(f"🎯 总共恢复了 {len(recovered_documents)} 个文档")
        return recovered_documents

    def _generate_final_report(self, source_summary: Dict) -> Dict:
        """生成最终构建报告"""
        duration = self.stats["end_time"] - self.stats["start_time"]

        report = {
            "build_info": {
                "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 2),
                "duration_minutes": round(duration / 60, 2)
            },
            "statistics": self.stats,
            "sources": source_summary,
            "quality_metrics": {
                "success_rate": self.stats["total_files_valid"] / max(self.stats["total_files_found"], 1),
                "processing_rate": self.stats["total_files_processed"] / max(self.stats["total_files_found"], 1),
                "error_rate": self.stats["processing_errors"] / max(self.stats["total_files_found"], 1)
            },
            "dataset_info": {
                "total_documents": self.stats["total_files_valid"],
                "total_sources": len(source_summary),
                "average_docs_per_source": self.stats["total_files_valid"] / max(len(source_summary), 1)
            }
        }

        return report

    def _print_final_summary(self, report: Dict):
        """打印最终摘要"""
        print("\n" + "=" * 60)
        print("📊 SMART RAG 数据集构建报告")
        print("=" * 60)

        print(f"⏱️  构建时间: {report['build_info']['duration_minutes']:.1f} 分钟")
        print(f"📁 数据源数量: {report['dataset_info']['total_sources']}")
        print(f"📄 有效文档: {report['dataset_info']['total_documents']}")
        print(f"✅ 成功率: {report['quality_metrics']['success_rate']:.1%}")

        print("\n📋 各数据源统计:")
        for source, info in report['sources'].items():
            print(f"  {source}: {info['document_count']} 文档")

        print("\n💾 输出文件:")
        print(f"  📄 完整数据集: processed_data/complete_dataset.json")
        print(f"  📊 构建报告: processed_data/quality_reports/build_report.json")
        print("=" * 60)


if __name__ == "__main__":
    # 直接运行测试
    builder = SmartRAGDatasetBuilder()
    builder.build_complete_dataset()
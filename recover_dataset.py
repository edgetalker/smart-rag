#!/usr/bin/env python3
"""
数据集恢复工具
从单独的数据源文件恢复完整数据集
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recover_complete_dataset():
    """从单独文件恢复完整数据集"""

    # 设置路径
    documents_dir = Path("processed_data/documents")
    complete_dataset_file = Path("processed_data/complete_dataset.json")

    if not documents_dir.exists():
        logger.error(f"❌ 文档目录不存在: {documents_dir}")
        return False

    logger.info("🔄 开始恢复数据集...")

    # 收集所有文档
    all_documents = []
    recovery_report = {}

    # 查找所有的*_documents.json文件
    json_files = list(documents_dir.glob("*_documents.json"))

    if not json_files:
        logger.error("❌ 未找到任何数据源文件")
        return False

    logger.info(f"📄 找到 {len(json_files)} 个数据源文件")

    for json_file in json_files:
        source_name = json_file.stem.replace("_documents", "")
        logger.info(f"📄 处理数据源: {source_name}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            if isinstance(documents, list):
                document_count = len(documents)
                all_documents.extend(documents)
                recovery_report[source_name] = {
                    "file": str(json_file),
                    "document_count": document_count,
                    "status": "success"
                }
                logger.info(f"✅ {source_name}: 恢复了 {document_count} 个文档")
            else:
                recovery_report[source_name] = {
                    "file": str(json_file),
                    "document_count": 0,
                    "status": "error",
                    "error": "Invalid format - not a list"
                }
                logger.warning(f"⚠️  {source_name}: 文件格式不正确")

        except Exception as e:
            recovery_report[source_name] = {
                "file": str(json_file),
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }
            logger.error(f"❌ {source_name}: 无法加载文件 - {e}")

    # 保存恢复的完整数据集
    if all_documents:
        try:
            with open(complete_dataset_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)

            file_size = complete_dataset_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ 完整数据集已保存: {complete_dataset_file}")
            logger.info(f"📊 文件大小: {file_size:.2f}MB")
            logger.info(f"📊 总文档数: {len(all_documents)}")

            # 保存恢复报告
            report_file = Path("processed_data/quality_reports/recovery_report.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)

            recovery_summary = {
                "recovery_timestamp": "2025-08-20",
                "total_documents_recovered": len(all_documents),
                "sources_processed": len(json_files),
                "sources_successful": len([r for r in recovery_report.values() if r["status"] == "success"]),
                "source_details": recovery_report
            }

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(recovery_summary, f, ensure_ascii=False, indent=2)

            logger.info(f"📋 恢复报告已保存: {report_file}")

            # 打印摘要
            print("\n" + "=" * 50)
            print("📊 数据集恢复摘要")
            print("=" * 50)
            print(f"总文档数: {len(all_documents)}")
            print(
                f"数据源: {len([r for r in recovery_report.values() if r['status'] == 'success'])}/{len(json_files)} 成功")

            for source, info in recovery_report.items():
                status_icon = "✅" if info["status"] == "success" else "❌"
                print(f"  {status_icon} {source}: {info['document_count']} 文档")
            print("=" * 50)

            return True

        except Exception as e:
            logger.error(f"❌ 保存完整数据集失败: {e}")
            return False
    else:
        logger.error("❌ 没有恢复到任何文档")
        return False


def verify_recovery():
    """验证恢复结果"""
    complete_dataset_file = Path("processed_data/complete_dataset.json")

    if not complete_dataset_file.exists():
        logger.error("❌ 完整数据集文件不存在")
        return False

    try:
        with open(complete_dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if isinstance(dataset, list) and len(dataset) > 0:
            logger.info(f"✅ 验证成功: 数据集包含 {len(dataset)} 个文档")

            # 检查数据结构
            sample_doc = dataset[0]
            required_fields = ["id", "source", "title", "content"]
            missing_fields = [field for field in required_fields if field not in sample_doc]

            if missing_fields:
                logger.warning(f"⚠️  样本文档缺少字段: {missing_fields}")
            else:
                logger.info("✅ 文档结构验证通过")

            return True
        else:
            logger.error("❌ 数据集为空或格式错误")
            return False

    except Exception as e:
        logger.error(f"❌ 验证过程出错: {e}")
        return False


def main():
    """主函数"""
    logger.info("🚀 开始数据集恢复过程...")

    # 恢复数据集
    if recover_complete_dataset():
        logger.info("✅ 数据集恢复成功")

        # 验证恢复结果
        if verify_recovery():
            logger.info("🎉 数据集恢复并验证完成！")
            print("\n下一步:")
            print("  python validate_dataset.py  # 运行完整质量验证")
            return 0
        else:
            logger.error("❌ 数据集验证失败")
            return 1
    else:
        logger.error("❌ 数据集恢复失败")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
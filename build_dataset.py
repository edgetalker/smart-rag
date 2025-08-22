#!/usr/bin/env python3
"""
Smart RAG 数据集构建主脚本
一键构建完整的开源文档数据集
"""

import sys
import argparse
from pathlib import Path
from data_pipeline.dataset_builder import SmartRAGDatasetBuilder
import logging


def setup_logging(verbose: bool = False):
    """设置日志级别"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_prerequisites():
    """检查前置条件"""
    import shutil

    # 检查Git是否安装
    if not shutil.which("git"):
        print("❌ 错误: 未找到Git命令，请先安装Git")
        print("   Windows: https://git-scm.com/download/win")
        return False

    # 检查网络连接（简单测试）
    try:
        import urllib.request
        urllib.request.urlopen('https://github.com', timeout=5)
    except:
        print("⚠️  警告: 网络连接可能有问题，建议检查网络后重试")

    # 检查磁盘空间（估算需要2GB）
    import shutil
    free_space = shutil.disk_usage('.').free / (1024 ** 3)  # GB
    if free_space < 2:
        print(f"⚠️  警告: 磁盘剩余空间较少 ({free_space:.1f}GB)，建议清理磁盘空间")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Smart RAG 数据集构建器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python build_dataset.py                    # 构建完整数据集
  python build_dataset.py --verbose          # 详细日志输出
  python build_dataset.py --config custom.yaml  # 使用自定义配置
  python build_dataset.py --sources langchain fastapi  # 只构建指定数据源
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="配置文件路径 (默认: configs/data_sources.yaml)"
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        help="指定要构建的数据源 (例如: langchain fastapi)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="跳过前置条件检查"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="输出目录 (默认: processed_data)"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 检查前置条件
    if not args.skip_checks:
        logger.info("🔍 检查前置条件...")
        if not check_prerequisites():
            sys.exit(1)
        logger.info("✅ 前置条件检查通过")

    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    try:
        # 创建数据集构建器
        logger.info("🚀 初始化Smart RAG数据集构建器...")
        builder = SmartRAGDatasetBuilder(config_path=args.config)

        # 如果指定了特定数据源
        if args.sources:
            logger.info(f"📌 仅构建指定数据源: {args.sources}")

            # 验证数据源名称
            available_sources = list(builder.config["data_sources"].keys())
            invalid_sources = [s for s in args.sources if s not in available_sources]
            if invalid_sources:
                logger.error(f"❌ 无效的数据源: {invalid_sources}")
                logger.error(f"   可用数据源: {available_sources}")
                sys.exit(1)

            # 只保留指定的数据源
            filtered_sources = {
                k: v for k, v in builder.config["data_sources"].items()
                if k in args.sources
            }
            builder.config["data_sources"] = filtered_sources

        # 开始构建数据集
        logger.info("🎯 开始构建数据集...")
        report = builder.build_complete_dataset()

        # 输出最终结果
        output_files = [
            "processed_data/complete_dataset.json",
            "processed_data/quality_reports/build_report.json"
        ]

        print("\n🎉 数据集构建完成！")
        print("📁 生成的文件:")
        for file_path in output_files:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                print(f"   {file_path} ({file_size:.1f}MB)")

        # 显示下一步建议
        print("\n🚀 下一步建议:")
        print("   1. 检查质量报告: processed_data/quality_reports/build_report.json")
        print("   2. 开始构建RAG系统: python build_rag_system.py")
        print("   3. 运行质量验证: python validate_dataset.py")

        return 0

    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断构建过程")
        return 1
    except Exception as e:
        logger.error(f"❌ 构建过程发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
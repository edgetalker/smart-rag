#!/usr/bin/env python3
"""
Smart RAG 数据集质量验证工具
验证构建的数据集是否符合Phase 2的需求
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetQualityValidator:
    """数据集质量验证器"""

    def __init__(self, dataset_path: str = "processed_data/complete_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()
        self.validation_results = {}

    def _load_dataset(self) -> List[Dict]:
        """加载数据集"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        logger.info(f"📊 已加载数据集: {len(dataset)} 个文档")
        return dataset

    def validate_basic_structure(self) -> Dict:
        """验证基本数据结构"""
        logger.info("🔍 验证基本数据结构...")

        required_fields = ["id", "source", "title", "content", "headings", "code_blocks", "stats"]
        validation_results = {
            "total_documents": len(self.dataset),
            "valid_documents": 0,
            "missing_fields": [],
            "field_coverage": {}
        }

        # 处理空数据集的情况
        if len(self.dataset) == 0:
            logger.warning("⚠️  数据集为空，无法进行结构验证")
            validation_results["error"] = "Empty dataset"
            validation_results["field_coverage"] = {field: {"count": 0, "coverage": 0.0} for field in required_fields}
            validation_results["missing_fields"] = required_fields
            self.validation_results["structure"] = validation_results
            return validation_results

        for field in required_fields:
            field_count = sum(1 for doc in self.dataset if field in doc and doc[field] is not None)
            coverage = field_count / len(self.dataset)  # 现在安全了，因为已经检查了len(self.dataset) > 0
            validation_results["field_coverage"][field] = {
                "count": field_count,
                "coverage": coverage
            }

            if coverage < 0.95:  # 95%覆盖率标准
                validation_results["missing_fields"].append(field)

        validation_results["valid_documents"] = sum(
            1 for doc in self.dataset
            if all(field in doc and doc[field] is not None for field in required_fields)
        )

        self.validation_results["structure"] = validation_results
        return validation_results

    def validate_content_quality(self) -> Dict:
        """验证内容质量"""
        logger.info("📝 验证内容质量...")

        quality_metrics = {
            "content_length": [],
            "heading_count": [],
            "code_block_count": [],
            "word_count": [],
            "code_ratio": []
        }

        for doc in self.dataset:
            if "stats" in doc and doc["stats"]:
                stats = doc["stats"]
                quality_metrics["content_length"].append(stats.get("character_count", 0))
                quality_metrics["word_count"].append(stats.get("word_count", 0))
                quality_metrics["code_ratio"].append(stats.get("code_ratio", 0))

            quality_metrics["heading_count"].append(len(doc.get("headings", [])))
            quality_metrics["code_block_count"].append(len(doc.get("code_blocks", [])))

        # 计算统计指标
        quality_results = {}
        for metric, values in quality_metrics.items():
            if values:
                quality_results[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "median": sorted(values)[len(values) // 2]
                }
            else:
                quality_results[metric] = {"error": "No valid data"}

        # 质量检查
        quality_issues = []

        # 检查过短文档
        short_docs = sum(1 for length in quality_metrics["content_length"] if length < 200)
        if short_docs > len(self.dataset) * 0.1:  # 超过10%的文档过短
            quality_issues.append(f"过短文档比例过高: {short_docs}/{len(self.dataset)}")

        # 检查无标题文档
        no_heading_docs = sum(1 for count in quality_metrics["heading_count"] if count == 0)
        if no_heading_docs > len(self.dataset) * 0.05:  # 超过5%的文档无标题
            quality_issues.append(f"无标题文档比例过高: {no_heading_docs}/{len(self.dataset)}")

        quality_results["issues"] = quality_issues
        quality_results["summary"] = {
            "total_characters": sum(quality_metrics["content_length"]),
            "total_words": sum(quality_metrics["word_count"]),
            "total_headings": sum(quality_metrics["heading_count"]),
            "total_code_blocks": sum(quality_metrics["code_block_count"])
        }

        self.validation_results["quality"] = quality_results
        return quality_results

    def validate_source_distribution(self) -> Dict:
        """验证数据源分布"""
        logger.info("📊 验证数据源分布...")

        source_distribution = Counter(doc.get("source", "unknown") for doc in self.dataset)

        distribution_results = {
            "sources": dict(source_distribution),
            "total_sources": len(source_distribution),
            "balance_score": self._calculate_balance_score(source_distribution) if source_distribution else 0.0
        }

        # 检查是否有源占比过高
        total_docs = len(self.dataset)
        if total_docs > 0:  # 避免除零
            max_ratio = max(count / total_docs for count in source_distribution.values()) if source_distribution else 0

            if max_ratio > 0.7:  # 单一数据源占比超过70%
                distribution_results["warning"] = f"数据源分布不均衡，最大占比: {max_ratio:.1%}"
        else:
            distribution_results["warning"] = "数据集为空，无法评估分布"

        self.validation_results["distribution"] = distribution_results
        return distribution_results

    def _calculate_balance_score(self, distribution: Counter) -> float:
        """计算分布均衡度分数 (0-1, 越接近1越均衡)"""
        if not distribution:
            return 0.0

        total = sum(distribution.values())
        expected_ratio = 1.0 / len(distribution)

        # 计算每个源与期望比例的偏差
        deviations = []
        for count in distribution.values():
            actual_ratio = count / total
            deviation = abs(actual_ratio - expected_ratio)
            deviations.append(deviation)

        # 均衡度 = 1 - 平均偏差
        avg_deviation = sum(deviations) / len(deviations)
        balance_score = max(0, 1 - avg_deviation * len(distribution))

        return balance_score

    def validate_phase2_readiness(self) -> Dict:
        """验证Phase 2定制splitting的准备度"""
        logger.info("🎯 验证Phase 2准备度...")

        phase2_metrics = {
            "structural_features": 0,
            "code_diversity": 0,
            "heading_hierarchy": 0,
            "cross_references": 0
        }

        # 处理空数据集
        if not self.dataset:
            logger.warning("⚠️  数据集为空，无法评估Phase 2准备度")
            phase2_metrics.update({
                "structural_coverage": 0.0,
                "code_language_diversity": 0,
                "heading_level_diversity": 0,
                "cross_reference_coverage": 0.0,
                "languages_found": [],
                "heading_levels_found": [],
                "readiness_score": 0,
                "readiness_factors": ["❌ 数据集为空"],
                "readiness_level": "需要重建数据集"
            })
            self.validation_results["phase2_readiness"] = phase2_metrics
            return phase2_metrics

        # 检查结构化特征
        docs_with_structure = 0
        code_languages = set()
        heading_levels = set()
        docs_with_links = 0

        for doc in self.dataset:
            # 结构化文档 (有标题和内容)
            if doc.get("headings") and len(doc["headings"]) > 0:
                docs_with_structure += 1

                # 收集标题层级
                for heading in doc["headings"]:
                    heading_levels.add(heading.get("level", 1))

            # 代码块多样性
            if doc.get("code_blocks"):
                for block in doc["code_blocks"]:
                    code_languages.add(block.get("language", "text"))

            # 交叉引用 (链接)
            if doc.get("links") and len(doc["links"]) > 0:
                docs_with_links += 1

        total_docs = len(self.dataset)

        phase2_metrics = {
            "structural_coverage": docs_with_structure / total_docs,
            "code_language_diversity": len(code_languages),
            "heading_level_diversity": len(heading_levels),
            "cross_reference_coverage": docs_with_links / total_docs,
            "languages_found": list(code_languages),
            "heading_levels_found": sorted(list(heading_levels))
        }

        # 评估准备度
        readiness_score = 0
        readiness_factors = []

        if phase2_metrics["structural_coverage"] > 0.8:
            readiness_score += 25
            readiness_factors.append("✅ 良好的文档结构覆盖")
        else:
            readiness_factors.append("⚠️ 结构化文档比例偏低")

        if phase2_metrics["code_language_diversity"] >= 5:
            readiness_score += 25
            readiness_factors.append("✅ 代码语言多样性充足")
        else:
            readiness_factors.append("⚠️ 代码语言类型较少")

        if phase2_metrics["heading_level_diversity"] >= 3:
            readiness_score += 25
            readiness_factors.append("✅ 标题层次丰富")
        else:
            readiness_factors.append("⚠️ 标题层次单一")

        if phase2_metrics["cross_reference_coverage"] > 0.3:
            readiness_score += 25
            readiness_factors.append("✅ 交叉引用覆盖良好")
        else:
            readiness_factors.append("⚠️ 交叉引用较少")

        phase2_metrics["readiness_score"] = readiness_score
        phase2_metrics["readiness_factors"] = readiness_factors
        phase2_metrics["readiness_level"] = self._get_readiness_level(readiness_score)

        self.validation_results["phase2_readiness"] = phase2_metrics
        return phase2_metrics

    def _get_readiness_level(self, score: int) -> str:
        """获取准备度等级"""
        if score >= 90:
            return "优秀 (Excellent)"
        elif score >= 75:
            return "良好 (Good)"
        elif score >= 60:
            return "中等 (Fair)"
        else:
            return "需要改进 (Needs Improvement)"

    def generate_validation_report(self) -> Dict:
        """生成完整验证报告"""
        logger.info("📋 生成验证报告...")

        # 运行所有验证
        structure_results = self.validate_basic_structure()
        quality_results = self.validate_content_quality()
        distribution_results = self.validate_source_distribution()
        phase2_results = self.validate_phase2_readiness()

        # 生成综合报告
        report = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "file_path": str(self.dataset_path),
                "total_documents": len(self.dataset)
            },
            "validation_results": {
                "structure": structure_results,
                "quality": quality_results,
                "distribution": distribution_results,
                "phase2_readiness": phase2_results
            },
            "overall_assessment": self._generate_overall_assessment()
        }

        # 保存报告
        report_path = Path("processed_data/quality_reports/validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 验证报告已保存: {report_path}")
        return report

    def _generate_overall_assessment(self) -> Dict:
        """生成总体评估"""
        assessments = []
        score = 0

        # 处理空数据集情况
        if not self.dataset:
            return {
                "overall_score": 0,
                "grade": "F (数据集为空)",
                "assessments": ["❌ 数据集为空，需要重新构建"],
                "recommendations": ["重新运行数据集构建流程", "检查数据源配置", "验证网络连接"]
            }

        # 结构完整性
        if self.validation_results["structure"]["valid_documents"] / len(self.dataset) > 0.95:
            assessments.append("✅ 数据结构完整")
            score += 25
        else:
            assessments.append("⚠️ 数据结构存在缺失")

        # 内容质量
        if not self.validation_results["quality"].get("issues"):
            assessments.append("✅ 内容质量良好")
            score += 25
        else:
            assessments.append("⚠️ 内容质量存在问题")

        # 分布均衡
        if self.validation_results["distribution"]["balance_score"] > 0.7:
            assessments.append("✅ 数据源分布均衡")
            score += 25
        else:
            assessments.append("⚠️ 数据源分布不均衡")

        # Phase 2准备度
        phase2_score = self.validation_results["phase2_readiness"]["readiness_score"]
        if phase2_score >= 75:
            assessments.append("✅ Phase 2准备度充分")
            score += 25
        else:
            assessments.append("⚠️ Phase 2准备度需要提升")

        return {
            "overall_score": score,
            "grade": self._get_grade(score),
            "assessments": assessments,
            "recommendations": self._generate_recommendations()
        }

    def _get_grade(self, score: int) -> str:
        """获取总体评级"""
        if score >= 90:
            return "A (优秀)"
        elif score >= 80:
            return "B (良好)"
        elif score >= 70:
            return "C (中等)"
        else:
            return "D (需要改进)"

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于验证结果生成建议
        if self.validation_results["structure"]["missing_fields"]:
            recommendations.append("补充缺失的数据字段")

        if self.validation_results["quality"].get("issues"):
            recommendations.append("提升内容质量，移除过短或无效文档")

        if self.validation_results["distribution"]["balance_score"] < 0.7:
            recommendations.append("平衡数据源分布，增加较少的数据源内容")

        if self.validation_results["phase2_readiness"]["readiness_score"] < 75:
            recommendations.append("增强文档结构多样性，准备Phase 2定制splitting")

        if not recommendations:
            recommendations.append("数据集质量良好，可以开始Phase 1实施")

        return recommendations

    def print_summary(self):
        """打印验证摘要"""
        report = self.generate_validation_report()

        print("\n" + "=" * 60)
        print("📊 SMART RAG 数据集质量验证报告")
        print("=" * 60)

        # 基本信息
        total_docs = report['dataset_info']['total_documents']
        print(f"📄 总文档数: {total_docs}")

        if total_docs == 0:
            print("❌ 数据集为空！")
            print("\n💡 建议:")
            print("  1. 检查是否存在单独的数据源文件:")
            print("     dir processed_data\\documents")
            print("  2. 如果存在，运行数据恢复:")
            print("     python recover_dataset.py")
            print("  3. 如果不存在，重新构建数据集:")
            print("     python build_dataset.py")
        else:
            print(f"📊 总体评级: {report['overall_assessment']['grade']}")

            # 只有在数据不为空时才显示Phase 2准备度
            if 'phase2_readiness' in self.validation_results and 'readiness_level' in self.validation_results[
                'phase2_readiness']:
                print(f"🎯 Phase 2准备度: {self.validation_results['phase2_readiness']['readiness_level']}")

            # 数据源分布
            if report['validation_results']['distribution']['sources']:
                print(f"\n📋 数据源分布:")
                for source, count in report['validation_results']['distribution']['sources'].items():
                    percentage = count / total_docs * 100
                    print(f"  {source}: {count} ({percentage:.1f}%)")

            # 评估结果
            print(f"\n✅ 评估结果:")
            for assessment in report['overall_assessment']['assessments']:
                print(f"  {assessment}")

        # 改进建议
        print(f"\n💡 改进建议:")
        for recommendation in report['overall_assessment']['recommendations']:
            print(f"  • {recommendation}")

        print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Smart RAG 数据集质量验证")
    parser.add_argument(
        "--dataset",
        type=str,
        default="processed_data/complete_dataset.json",
        help="数据集文件路径"
    )

    args = parser.parse_args()

    try:
        validator = DatasetQualityValidator(args.dataset)
        validator.print_summary()

    except Exception as e:
        logger.error(f"❌ 验证过程发生错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
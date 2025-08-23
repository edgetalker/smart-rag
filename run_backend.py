import sys
import subprocess
from pathlib import Path


def main():
    """启动后端服务"""

    # 基本检查
    if not (Path.cwd() / "backend" / "main.py").exists():
        print("❌ 错误: 找不到 backend/main.py")
        print("请确保在项目根目录下运行此脚本")
        sys.exit(1)

    # 显示启动信息
    print("🚀 启动 Smart RAG 后端服务...")
    print(f"📂 项目目录: {Path.cwd().name}")
    print("-" * 50)

    # 启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload"
    ]

    try:
        # 运行服务器
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except FileNotFoundError:
        print("❌ 错误: 找不到 uvicorn")
        print("请安装: pip install uvicorn")


if __name__ == "__main__":
    main()
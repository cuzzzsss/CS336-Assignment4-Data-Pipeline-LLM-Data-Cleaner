# CS336-Assignment4: Large-Scale LLM Data Pipeline

本项目是 Stanford CS336 (Spring 2025) 课程 Assignment 4 的完整实现。构建了一个从原始 HTML 到高质量预训练语料的工业级端到端处理流水线。

---

## 🌟 核心功能实现

本项目在 `./cs336_data` 模块中实现了以下核心组件，并已全部通过 `pytest` 验证：

### 1. 文本提取与启发式过滤 (Heuristic Cleaning)
- **HTML Extraction**：基于 `resiliparse` 从复杂的网页字节流中精准提取正文内容。
- **Gopher 质量过滤**：完整实现 Gopher 论文中的启发式规则，通过词长、符号占比、项目符号比例及重复行占比等指标自动剔除低质量噪声。

### 2. 安全与隐私保障 (Safety & Privacy)
- **PII 屏蔽**：利用正则表达式自动检测并屏蔽 Email、电话号码及 IP 地址等个人敏感隐私信息。
- **毒性检测**：接入 FastText 预训练模型，实现对有害言论（Toxic Speech）与色情内容（NSFW）自动化分类与过滤。

### 3. 大规模去重 (Deduplication)
- **精确行去重**：通过全局行频率统计，高效识别并抹除网页模板带来的重复噪声。
- **MinHash LSH（Fuzzy Deduplication）**：项目核心技术点。利用局部敏感哈希（LSH）算法，在接近 $O(N)$ 的时间复杂度内识别内容高度相似的文档，有效降低大规模语料库中的重复带来的过拟合风险。

---

## 📂 项目结构 (Project Structure)

```text
.
├── cs336_basics       # 课程提供的优化训练框架实现
├── cs336_data         # 我的核心实现：包含数据预处理与去重逻辑
│   ├── preprocessing.py
│   └── deduplication.py
├── tests              # 验证脚本：包含质量过滤、去重等所有单元测试
├── README.md
└── pyproject.toml
🚀 快速开始 (Quick Start)
本项目使用 uv 进行依赖管理。

1. 环境配置 (Setup)
# 安装依赖
uv pip install -e .
2. 运行测试 (Run Tests)
# 运行去重模块测试
pytest tests/test_deduplication.py

# 运行质量过滤测试
pytest tests/test_quality.py
3. 生成提交 (Make Submission)
./test_and_make_submission.sh
📝 课程背景 (Course Context)
本项目基于 Stanford CS336: Spring 2025 课程。详细作业说明请参考仓库内的 cs336_spring2025_assignment4_data.pdf。

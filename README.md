# SearchRefine-LLDs: Enhancing GRPO Stability with AutoRefine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL/Coder-orange.svg)](https://github.com/QwenLM/Qwen2.5)

**SearchRefine-LLDs** 是一个旨在提升大语言模型（LLM）推理能力的开源项目。本项目基于 [AutoRefine](https://github.com/syr-cn/AutoRefine) 框架，并针对论文 [arXiv:2512.04220](https://arxiv.org/abs/2512.04220) 中提出的 **GRPO 训练坍塌 (Training Collapse)** 问题进行了深度优化与复现。

---

## 🌟 核心特性 (Key Features)

- **Iterative Search & Refine**: 集成 AutoRefine 的自动化搜索流，通过多次 Rollout 和自我修正（Self-Correction）生成高质量推理路径。
- **GRPO Anti-Collapse Mechanisms**: 
    - **Reward Rescaling**: 实现论文建议的奖励重缩放，有效缓解组内相对奖励差异过大导致的梯度爆炸。
    - **Stabilized KL-Divergence**: 优化了 $D_{KL}(\pi_\theta || \pi_{ref})$ 的惩罚项，确保在大规模 Post-training 过程中策略更新的平滑性。
- **Efficient Scalability**: 适配分布式训练环境（Megatron/DeepSpeed），支持 vLLM 快速推理采样。

---

## 📈 算法原理 (Algorithm & Stability)

在标准的 GRPO 训练中，由于缺乏 Critic 网络，组内奖励的方差极易导致训练坍塌。本项目参考 [arXiv:2512.04220](https://arxiv.org/abs/2512.04220) 引入了稳定性修正项：

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \mathcal{L}_{clip}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$

针对 **Training Collapse**，我们特别实现了：
1. **Group-wise Advantage Whitening**: 在采样组内执行严格的归一化，确保 $\hat{A}_i$ 的分布稳定性。
2. **Adaptive Reward Clipping**: 动态调整奖励裁剪阈值，防止单一异常样本破坏整个 Batch 的梯度方向。

---

## 🛠 安装与快速开始 (Quick Start)

### 1. 安装环境
```bash
git clone [https://github.com/tcn-create/SearchRefine-LLDs.git](https://github.com/tcn-create/SearchRefine-LLDs.git)
cd SearchRefine-LLDs
pip install -r requirements.txt
```
## 致谢 (Acknowledgements)
特别感谢以下开源项目和研究工作：

[AutoRefine](https://github.com/syr-cn/AutoRefine): 提供了优秀的迭代搜索框架。

[arXiv:2512.04220](https://arxiv.org/abs/2512.04220): 为解决 GRPO 训练稳定性提供了核心理论支撑。

## 📄 开源协议 (License)
本项目采用 Apache 2.0 协议开源。
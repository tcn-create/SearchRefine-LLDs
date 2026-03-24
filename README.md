# SearchRefine-LLDs: Enhancing GRPO Stability with AutoRefine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL/Coder-orange.svg)](https://github.com/QwenLM/Qwen2.5)

**SearchRefine-LLDs** 是一个旨在提升大语言模型（LLM）推理能力的开源项目。本项目基于 [AutoRefine](https://github.com/syr-cn/AutoRefine) 框架，并针对论文 [arXiv:2512.04220](https://arxiv.org/abs/2512.04220) 中提出的 **GRPO 训练坍塌 (Training Collapse)** 问题进行了深度优化与复现。

---

## 🌟 核心特性 (Key Features)

- **Iterative Search & Refine**: 集成 AutoRefine 的自动化搜索流，通过多次 Rollout 和自我修正（Self-Correction）生成高质量推理路径。
- **GRPO Anti-Collapse Mechanisms**: 
    - **Group-wise Reward Whitening**: 引入严格的组内奖励标准化逻辑，加入稳定性常数 $\eta$ 防止梯度爆炸。
    - **Dynamic KL-Divergence Control**: 实现论文建议的动态 $\beta$ 调节机制，根据实时策略偏差自动调整约束强度。
- **Efficient Scalability**: 适配分布式训练环境（Megatron/DeepSpeed），支持 vLLM 快速推理采样。

---

## 📈 算法原理 (Algorithm & Stability)

在标准的 GRPO 训练中，由于缺乏 Critic 网络，组内奖励的方差极易导致训练坍塌。本项目参考 [arXiv:2512.04220](https://arxiv.org/abs/2512.04220) 引入了以下稳定性修正：

### 1. 目标函数优化
我们在损失函数中强化了对策略偏移的动态约束：
$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) - \beta_t D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$

### 2. 抑制坍塌的核心改进
针对 **Training Collapse**，本项目重点实现了：

* **Advantage Whitening (优势函数白化)**:
    通过以下公式确保每个采样组内的优势函数分布稳定：
    $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \eta}$$
    *其中 \eta 为稳定性常数，确保在奖励分布极度集中时仍能提供可靠梯度。*

* **Adaptive KL Control (动态 KL 控制)**:
    实现了动态更新系数 \beta_t 的逻辑，防止模型在推理路径过长时出现策略突变：
    $$\beta_{t+1} = \beta_t + \alpha (D_{KL} - \text{Target}_{KL})$$

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
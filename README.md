<div align="center">

# Awesome RL-based Reasoning MLLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>

Recent advancements in leveraging reinforcement learning to enhance LLM reasoning capabilities have yielded remarkably promising results, exemplified by [DeepSeek-R1](https://arxiv.org/pdf/2501.12948), [Kimi k1.5](https://arxiv.org/pdf/2501.12599), [OpenAI o3-mini](https://openai.com/index/o3-mini-system-card/), [Grok 3](https://x.ai/blog/grok-3). These exhilarating achievements herald ascendance of Large Reasoning Models, making us advance further along the thorny path towards Artificial General Intelligence (AGI). Study of LLM reasoning has garnered significant attention within the community, and researchers have concurrently summarized [awesome RL-based LLM reasoning](https://github.com/bruno686/Awesome-RL-based-LLM-Reasoning). Meanwhile, we have observed that remarkably awesome work has already been done in the domain of Multimodal Large Language Models (MLLMs), encompassing both **multimodal understanding** and **autoregressive text-to-image generation**.
<div align="center">
    "The senses are the organs by which man perceives the world, and the soul acts through them as through tools."
</div>
<div align="right">
— Leonardo da Vinci
</div>
This repository provides valuable reference for researchers in the field of multimodality, please start your exploratory travel in RL-based Reasoning MLLMs!

## Papers📄
* [2503] [Seg-Zero] [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/pdf/2503.06520) (CUHK) [Models 🤗](https://huggingface.co/Ricky06662/Seg-Zero-7B) [Dataset 🤗](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840) [Code 💻](https://github.com/dvlab-research/Seg-Zero)

* [2503] [MM-Eureka] [MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf) (Shanghai AI Laboratory) [Models 🤗](https://huggingface.co/FanqingM) [Dataset 🤗](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset) [Code 💻](https://github.com/ModalMinds/MM-EUREKA)

* [2503] [Visual-RFT] [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/pdf/2503.01785) (SJTU) [Project 🌐](https://github.com/Liuziyu77/Visual-RFT) [Datasets 🤗](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df) [Code 💻](https://github.com/Liuziyu77/Visual-RFT)

* [2502] [MedVLM-R1] [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning](https://arxiv.org/pdf/2502.19634) (TUM)
  
* [2501] [Kimi k1.5] [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599) (MoonshotAI) [Project 🌐](https://github.com/MoonshotAI/Kimi-k1.5)
  
* [2501] [Mulberry] [Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search](https://arxiv.org/pdf/2412.18319) (THU) [Model 🤗](https://huggingface.co/HuanjinYao/Mulberry_llava_8b) [Code 💻](https://github.com/HJYao00/Mulberry)

* [2501] [Virgo] [Virgo: A Preliminary Exploration on Reproducing o1-like MLLM](https://arxiv.org/abs/2501.01904v2) (RUC) [Model 🤗](https://huggingface.co/RUC-AIBOX/Virgo-72B) [Code 💻](https://github.com/RUCAIBox/Virgo)
  
* [2501] [Text-to-image COT] [Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step](https://arxiv.org/pdf/2501.13926) (CUHK) [Project 🌐](https://github.com/ZiyuGuo99/Image-Generation-CoT) [Model 🤗](https://huggingface.co/ZiyuG/Image-Generation-CoT)  [Code 💻](https://github.com/ZiyuGuo99/Image-Generation-CoT)
  
* [2501] [LlamaV-o1] [LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs](https://arxiv.org/pdf/2501.06186) (MBZUAI) [Project 🌐](https://mbzuai-oryx.github.io/LlamaV-o1/) [Model 🤗](https://huggingface.co/omkarthawakar/LlamaV-o1)  [Code 💻](https://github.com/mbzuai-oryx/LlamaV-o1)

* [2411] [InternVL2.5-MPO] [InternVL2.5-MPO: Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization
](https://arxiv.org/abs/2411.10442) (Shanghai AI Laboratory) [Project 🌐](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/) [Model 🤗](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO) [Code 💻](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo)

* [2411] [LLaVA-CoT] [LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/abs/2411.10440v4) (PKU) [Project 🌐](https://github.com/PKU-YuanGroup/LLaVA-CoT) [Model 🤗](https://huggingface.co/Xkev/Llama-3.2V-11B-cot) [Demo🤗](https://huggingface.co/spaces/Xkev/Llama-3.2V-11B-cot) [Code 💻](https://github.com/PKU-YuanGroup/LLaVA-CoT)


## Benchmarks📊

* [2502] [MME-CoT] [MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency](https://arxiv.org/abs/2502.09621) (CUHK) [Project 🌐](https://mmecot.github.io/) [Dataset 🤗](https://huggingface.co/datasets/CaraJ/MME-CoT) [Code 💻](https://github.com/CaraJ7/MME-CoT)

* [2502] [ZeroBench] [ZeroBench: An Impossible* Visual Benchmark for Contemporary Large Multimodal Models](https://arxiv.org/pdf/2502.09696) (Cambridge) [Project 🌐](https://zerobench.github.io/) [Dataset 🤗](https://huggingface.co/datasets/jonathan-roberts1/zerobench) [Code 💻](https://github.com/jonathan-roberts1/zerobench/)


## Open-Source Projects🌐

* [EasyR1 💻](https://github.com/hiyouga/EasyR1)  ![EasyR1](https://img.shields.io/github/stars/hiyouga/EasyR1) (An Efficient, Scalable, Multi-Modality RL Training Framework)

* [Multimodal Open R1 💻](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)  ![Multimodal Open R1](https://img.shields.io/github/stars/EvolvingLMMs-Lab/open-r1-multimodal) [Model 🤗](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) [Dataset 🤗](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified)
  
* [LMM-R1 💻](https://github.com/TideDra/lmm-r1) ![LMM-R1](https://img.shields.io/github/stars/TideDra/lmm-r1) [Code 💻](https://github.com/TideDra/lmm-r1)

* [R1-V 💻](https://github.com/Deep-Agent/R1-V)  ![R1-V](https://img.shields.io/github/stars/Deep-Agent/R1-V) [Blog 🎯](https://deepagent.notion.site/rlvr-in-vlms) [Datasets 🤗](https://huggingface.co/collections/MMInstruction/r1-v-67aae24fa56af9d2e2755f82)
  
* [VLM-R1 💻](https://github.com/om-ai-lab/VLM-R1)  ![VLM-R1](https://img.shields.io/github/stars/om-ai-lab/VLM-R1) [Model 🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)  [Dataset 🤗](https://huggingface.co/datasets/omlab/VLM-R1) [Demo 🤗](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression)

* [R1-Vision 💻](https://github.com/yuyq96/R1-Vision) ![R1-Vision](https://img.shields.io/github/stars/yuyq96/R1-Vision) [Cold-Start Datasets 🤗](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83)

* [R1-Onevision 💻](https://github.com/Fancy-MLLM/R1-Onevision)  ![R1-Onevision](https://img.shields.io/github/stars/Fancy-MLLM/R1-Onevision) [Model 🤗](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)  [Dataset 🤗](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) [Demo 🤗](https://huggingface.co/spaces/Fancy-MLLM/R1-Onevision) [Report 📝](https://yangyi-vai.notion.site/r1-onevision)

* [VisualThinker-R1-Zero 💻](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)  ![R1-Onevision](https://img.shields.io/github/stars/turningpoint-ai/VisualThinker-R1-Zero) [Report 📝](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model) (Aha Moment on a 2B non-SFT Model)

* [Open R1 Video 💻](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![Open R1 Video](https://img.shields.io/github/stars/Wang-Xiaodong1899/Open-R1-Video) [Models 🤗](https://huggingface.co/Xiaodong/Open-R1-Video-7B)  [Datasets 🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k) [Datasets 🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)

* [Video-R1 💻](https://github.com/tulerfeng/Video-R1) ![Video-R1](https://img.shields.io/github/stars/tulerfeng/Video-R1) [Code 💻](https://github.com/tulerfeng/Video-R1)
 [Dataset 🤗](https://huggingface.co/datasets/Video-R1/DVD-counting)

* [Seg-Zero 💻](https://github.com/dvlab-research/Seg-Zero) ![Seg-Zero](https://img.shields.io/github/stars/dvlab-research/Seg-Zero) [Code 💻](https://github.com/dvlab-research/Seg-Zero) (Based on veRL and EasyR1, more efficient and memory friendly.)

##  Star Chart⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs)](https://star-history.com/#Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs&Date)

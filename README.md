<div align="center">

# Awesome RL-based Reasoning MLLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>

Recent advancements in leveraging reinforcement learning to enhance LLM reasoning capabilities have yielded remarkably promising results, exemplified by [DeepSeek-R1](https://arxiv.org/pdf/2501.12948), [Kimi k1.5](https://arxiv.org/pdf/2501.12599), [OpenAI o3-mini](https://openai.com/index/o3-mini-system-card/), [Grok 3](https://x.ai/blog/grok-3). These exhilarating achievements herald ascendance of Large Reasoning Models, making us advance further along the thorny path towards Artificial General Intelligence (AGI). Study of LLM reasoning has garnered significant attention within the community, and researchers have concurrently summarized [Awesome RL-based LLM Reasoning](https://github.com/bruno686/Awesome-RL-based-LLM-Reasoning). Recently, researchers have also compiled a collection of some projects with detailed configurations about Large Reasoning Models in [Awesome RL Reasoning Recipes ("Triple R")](https://github.com/TsinghuaC3I/Awesome-RL-Reasoning-Recipes). Meanwhile, we have observed that remarkably awesome work has already been done in the domain of **RL-based Reasoning Multimodal Large Language Models (MLLMs)**. We aim to provide the community with a comprehensive and timely synthesis of this fascinating and promising field, as well as some insights into it.

<div align="center">
    "The senses are the organs by which man perceives the world, and the soul acts through them as through tools."
</div>
<div align="right">
— Leonardo da Vinci
</div>

This repository provides valuable reference for researchers in the field of multimodality, please start your exploratory travel in RL-based Reasoning MLLMs!

## News 

🔥🔥🔥[2025-5-24] We write the position paper [Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models](https://huggingface.co/papers/2505.18536)  that summarizes recent advancements on the topic of RFT for MLLMs. We focus on answering the following three questions: ***1. What background should researchers interested in this field know?***  ***2. What has the community done?***  ***3. What could the community do next?***  We hope that this position paper will provide valuable insights to the community at this pivotal stage in the advancement toward AGI.

📧📧📧[2025-4-10] Based on existing work in the community, we provide some insights into this field, which you can find in the [PowerPoint presentation file](https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs/blob/main/Report_on_2025-4-10.pptx).

![image](Multimodal.jpg)

**Figure 1: An overview of the works done on reinforcement fine-tuning (RFT) for multimodal large language models (MLLMs). Works are sorted by release time and are collected up to May 15, 2025.** 

## Papers (Sort by Time of Release)📄

### Vision (Image)👀 

* [2505] [URSA] [URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics](https://arxiv.org/abs/2501.04686) [[Model 🤗](https://huggingface.co/URSA-MATH/URSA-8B-PS-GRPO)]  [[Datasets 🤗](https://huggingface.co/URSA-MATH)]  [[Code 💻](https://github.com/URSA-MATH)]

* [2505] [V-Triune] [One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/abs/2505.18129) [[Models 🤗](https://huggingface.co/collections/One-RL-to-See-Them-All/one-rl-to-see-them-all-6833d27abce23898b2f9815a)]  [[Dataset 🤗](https://huggingface.co/datasets/One-RL-to-See-Them-All/Orsta-Data-47k)]  [[Code 💻](https://github.com/MiniMax-AI/One-RL-to-See-Them-All)]

* [2505] [GoT-R1 (generation)] [GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning](https://arxiv.org/abs/2505.17022) [[Models 🤗](https://huggingface.co/gogoduan)] [[Code 💻](https://github.com/gogoduan/GoT-R1)]

* [2505] [SophiaVL-R1] [SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](https://arxiv.org/pdf/2505.17018) [[Models 🤗](https://huggingface.co/bunny127)]  [[Datasets 🤗](https://huggingface.co/bunny127)]  [[Code 💻](https://github.com/kxfan2002/SophiaVL-R1)]

* [2505] [R1-ShareVL] [R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO](https://arxiv.org/pdf/2505.16673) [[Code 💻](https://github.com/HJYao00/R1-ShareVL)]

* [2505] [VLM-R^3] [VLM-R^3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192) 

* [2505] [TON] [Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models](https://arxiv.org/pdf/2505.16854) [[Models 🤗](https://huggingface.co/collections/kolerk/ton-682ad9038395c21e228a645b)]  [[Datasets 🤗](https://huggingface.co/collections/kolerk/ton-682ad9038395c21e228a645b)]  [[Code 💻](https://github.com/kokolerk/TON)]

* [2505] [Pixel Reasoner] [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966) [[Project 🌐](https://tiger-ai-lab.github.io/Pixel-Reasoner/)] [[Models 🤗](https://huggingface.co/collections/TIGER-Lab/pixel-reasoner-682fe96ea946d10dda60d24e)]  [[Datasets 🤗](https://huggingface.co/collections/TIGER-Lab/pixel-reasoner-682fe96ea946d10dda60d24e)] [[Demo 🤗](https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner)] [[Code 💻](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)]

* [2505] [GRIT] [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879) [[Project 🌐](https://grounded-reasoning.github.io/)]  [[Demo 🤗](https://b86dd615e41b242e22.gradio.live/)] [[Code 💻](https://github.com/eric-ai-lab/GRIT)]

* [2505] [STAR-R1] [STAR-R1: Spacial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](https://arxiv.org/abs/2505.15804) [[Code 💻](https://github.com/zongzhao23/STAR-R1)]

* [2505] [VARD (generation)] [VARD: Efficient and Dense Fine-Tuning for Diffusion Models with Value-based RL](https://arxiv.org/abs/2505.15791)

* [2505] [Chain-of-Focus] [Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](https://arxiv.org/abs/2505.15436) [[Project 🌐](https://cof-reasoning.github.io/)]

* [2505] [Visionary-R1] [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.14677) [[Code 💻](https://github.com/maifoundations/Visionary-R1)]

* [2505] [VisualQuality-R1] [VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank](https://arxiv.org/abs/2505.14460) [[Models 🤗](https://huggingface.co/TianheWu/VisualQuality-R1-7B)] [[Code 💻](https://github.com/TianheWu/VisualQuality-R1)]

* [2505] [DeepEyes] [Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362) [[Project 🌐](https://visual-agent.github.io/)] [[Model 🤗](https://huggingface.co/ChenShawn/DeepEyes-7B)]  [[Dataset 🤗](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)] [[Code 💻](https://github.com/Visual-Agent/DeepEyes)]

* [2505] [Visual-ARFT] [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246) [[Models 🤗](https://huggingface.co/collections/laolao77/visual-arft-682c601d0e35ac6470adfe9f)] [[Datasets 🤗](https://huggingface.co/collections/laolao77/visual-arft-682c601d0e35ac6470adfe9f)] [[Code 💻](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT)]

* [2505] [UniVG-R1] [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](https://arxiv.org/pdf/2505.14231) [[Project 🌐](https://amap-ml.github.io/UniVG-R1-page/)] [[Model 🤗](https://huggingface.co/GD-ML/UniVG-R1)]  [[Dataset 🤗](https://huggingface.co/datasets/GD-ML/UniVG-R1-data)]  [[Code 💻](https://github.com/AMAP-ML/UniVG-R1)]

* [2505] [G1] [G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning](https://arxiv.org/abs/2505.13426)  [[Code 💻](https://github.com/chenllliang/G1)]

* [2505] [VisionReasoner] [VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.12081) [[Models 🤗](https://huggingface.co/Ricky06662)]  [[Datasets 🤗](https://huggingface.co/Ricky06662)]  [[Code 💻](https://github.com/dvlab-research/VisionReasoner)]

* [2505] [VPRL] [Visual Planning: Let’s Think Only with Images](https://arxiv.org/abs/2505.11409) [[Code 💻](https://github.com/yix8/VisualPlanning)]

* [2505] [GuardReasoner-VL] [GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning](https://arxiv.org/pdf/2505.11049) [[Code 💻](https://github.com/yueliu1999/GuardReasoner-VL)]

* [2505] [OpenThinkIMG] [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617) [[Model 🤗](https://huggingface.co/Warrieryes/OpenThinkIMG-Chart-Qwen2-2B-VL)]  [[Datasets 🤗](https://huggingface.co/collections/Warrieryes/openthinkimg-68244a63e97a24d9b7ffcde9)] [[Code 💻](https://github.com/zhaochen0110/OpenThinkIMG)]

* [2505] [DanceGRPO (generation)] [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) [[Project 🌐](https://dancegrpo.github.io/)] 
 [[Code 💻](https://github.com/XueZeyue/DanceGRPO)]

* [2505] [Flow-GRPO (generation)] [Flow-GRPO: Training Flow Matching Models via Online RL](https://www.arxiv.org/abs/2505.05470) [[Models 🤗](https://huggingface.co/jieliu)]  [[Code 💻](https://github.com/yifan123/flow_grpo)]

* [2505] [X-Reasoner] [X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains](https://arxiv.org/abs/2505.03981) [[Code 💻](https://github.com/microsoft/x-reasoner)]

* [2505] [T2I-R1 (generation)] [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/abs/2505.00703) [[Code 💻](https://github.com/CaraJ7/T2I-R1)]

* [2504] [FAST] [Fast-Slow Thinking for Large Vision-Language Model Reasoning](https://arxiv.org/abs/2504.18458) [[Code 💻](https://github.com/Mr-Loevan/FAST)]

* [2504] [Skywork R1V2] [Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning](https://arxiv.org/abs/2504.16656) [[Models 🤗](https://huggingface.co/collections/Skywork/skywork-r1v2-68075a3d947a5ae160272671)]  [[Code 💻](https://github.com/SkyworkAI/Skywork-R1V)]

* [2504] [Relation-R1] [Relation-R1: Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relational Comprehension](https://arxiv.org/abs/2504.14642)  [[Code 💻](https://github.com/HKUST-LongGroup/Relation-R1)]

* [2504] [R1-SGG] [Compile Scene Graphs with Reinforcement Learning](https://www.arxiv.org/abs/2504.13617) [[Code 💻](https://github.com/gpt4vision/R1-SGG)]

* [2504] [NoisyRollout] [Reinforcing Visual Reasoning with Data Augmentation](https://arxiv.org/abs/2504.13055) [[Models 🤗](https://huggingface.co/collections/xyliu6/noisyrollout-67ff992d1cf251087fe021a2)]  [[Datasets 🤗](https://huggingface.co/collections/xyliu6/noisyrollout-67ff992d1cf251087fe021a2)] [[Code 💻](https://github.com/John-AI-Lab/NoisyRollout)]

* [2504] [Qwen-AD] [Look Before You Decide: Prompting Active Deduction of MLLMs for Assumptive Reasoning](https://arxiv.org/abs/2404.12966) [[Code 💻](https://github.com/LeeeeTX/Qwen-AD)]

* [2504] [SimpleAR (generation)] [SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL](https://arxiv.org/abs/2504.11455) [[Models 🤗](https://huggingface.co/collections/Daniel0724/simplear-6805053f5b4b9961ac025136)]  [[Code 💻](https://github.com/wdrink/SimpleAR)]

* [2504] [VL-Rethinker] [Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837) [[Project 🌐](https://tiger-ai-lab.github.io/VL-Rethinker/)] [[Models 🤗](https://huggingface.co/collections/TIGER-Lab/vl-rethinker-67fdc54de07c90e9c6c69d09)] [[Dataset 🤗](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) [Code 💻](https://github.com/TIGER-AI-Lab/VL-Rethinker)]

* [2504] [Kimi-VL] [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491) [[Project 🌐](https://github.com/MoonshotAI/Kimi-VL)] [[Models 🤗](https://huggingface.co/collections/moonshotai/kimi-vl-a3b-67f67b6ac91d3b03d382dd85)] [[Demo 🤗](https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B-Thinking)] [[Code 💻](https://github.com/MoonshotAI/Kimi-VL)]

* [2504] [VLAA-Thinking] [SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models](https://github.com/UCSC-VLAA/VLAA-Thinking/blob/main/assets/VLAA-Thinker.pdf) [[Models 🤗](https://huggingface.co/collections/UCSC-VLAA/vlaa-thinker-67eda033419273423d77249e)]  [[Dataset 🤗](https://huggingface.co/datasets/UCSC-VLAA/VLAA-Thinking)]  [[Code 💻](https://github.com/UCSC-VLAA/VLAA-Thinking)]

* [2504] [Perception-R1] [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](https://arxiv.org/abs/2504.07954) [[Model 🤗](https://huggingface.co/collections/Kangheng/perception-r1-67f6b14f89d307a0ece985af)]  [[Datasets 🤗](https://huggingface.co/collections/Kangheng/perception-r1-67f6b14f89d307a0ece985af)]  [[Code 💻](https://github.com/linkangheng/PR1)]

* [2504] [SoTA with Less] [SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement](https://arxiv.org/abs/2504.07934) [[Model 🤗](https://huggingface.co/russwang/ThinkLite-VL-7B)]  [[Datasets 🤗](https://huggingface.co/collections/russwang/thinklite-vl-67f88c6493f8a7601e73fe5a)]  [[Code 💻](https://github.com/si0wang/ThinkLite-VL)]

* [2504] [VLM-R1] [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/abs/2504.07615) [[Model 🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)]  [[Dataset 🤗](https://huggingface.co/datasets/omlab/VLM-R1)] [[Demo 🤗](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression)] [[Code 💻](https://github.com/om-ai-lab/VLM-R1)]

* [2504] [CrowdVLM-R1] [CrowdVLM-R1: Expanding R1 Ability to Vision Language Model for Crowd Counting using Fuzzy Group Relative Policy Reward](https://arxiv.org/abs/2504.03724) [[Dataset 🤗](https://huggingface.co/datasets/yeyimilk/CrowdVLM-R1-data)] [[Code 💻](https://github.com/yeyimilk/CrowdVLM-R1)]

* [2504] [MAYE] [Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme](https://www.arxiv.org/abs/2504.02587) [[Dataset 🤗](https://huggingface.co/datasets/ManTle/MAYE)]  [[Code 💻](https://github.com/GAIR-NLP/MAYE)]

* [2503] [Q-Insight] [Q-Insight: Understanding Image Quality via Visual Reinforcement Learning](https://arxiv.org/abs/2503.22679) [[Code 💻](https://github.com/bytedance/Q-Insight)] [[Model 🤗](https://huggingface.co/ByteDance/Q-Insight)]

* [2503] [Reason-RFT] [Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning](https://arxiv.org/abs/2503.20752) [[Project 🌐](https://tanhuajie.github.io/ReasonRFT)] [[Dataset 🤗](https://huggingface.co/datasets/tanhuajie2001/Reason-RFT-CoT-Dataset)] [[Code 💻](https://github.com/tanhuajie/Reason-RFT)]

* [2503] [OpenVLThinker] [OpenVLThinker: An Early Exploration to Vision-Language Reasoning via Iterative Self-Improvement](https://arxiv.org/abs/2503.17352) [[Model 🤗](https://huggingface.co/ydeng9/OpenVLThinker-7B)] [[Code 💻](https://github.com/yihedeng9/OpenVLThinker)]

* [2503] [Think or Not Think] [Think or Not Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.16188)  [[Models 🤗](https://huggingface.co/afdsafas)] [[Datasets 🤗](https://huggingface.co/afdsafas)] [[Code 💻](https://github.com/minglllli/CLS-RL)]

* [2503] [OThink-MR1] [OThink-MR1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning](https://arxiv.org/abs/2503.16081) 

* [2503] [R1-VL] [R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](https://arxiv.org/abs/2503.12937)  [[Model 🤗](https://huggingface.co/jingyiZ00)] [[Code 💻](https://github.com/jingyi0000/R1-VL)]

* [2503] [Skywork R1V] [Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](https://github.com/SkyworkAI/Skywork-R1V/blob/main/Skywork_R1V.pdf) [[Model 🤗](https://huggingface.co/Skywork/Skywork-R1V-38B)] [[Code 💻](https://github.com/SkyworkAI/Skywork-R1V)]

* [2503] [R1-Onevision] [R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](https://arxiv.org/abs/2503.10615)  [[Model 🤗](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)]  [[Dataset 🤗](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision)]  [[Demo 🤗](https://huggingface.co/spaces/Fancy-MLLM/R1-Onevision)]  [[Code 💻](https://github.com/Fancy-MLLM/R1-Onevision)]

* [2503] [VisualPRM] [VisualPRM: An Effective Process Reward Model for Multimodal Reasoning](https://arxiv.org/abs/2503.10291v1)  [[Project 🌐](https://internvl.github.io/blog/2025-03-13-VisualPRM/)]  [[Model 🤗](https://huggingface.co/OpenGVLab/VisualPRM-8B)]  [[Dataset 🤗](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K)]  [[Benchmark 🤗](https://huggingface.co/datasets/OpenGVLab/VisualProcessBench)]

* [2503] [LMM-R1] [LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL](https://arxiv.org/abs/2503.07536) [[Code 💻](https://github.com/TideDra/lmm-r1)]

* [2503] [Curr-ReFT] [Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning](https://arxiv.org/abs/2503.07065)  [[Models 🤗](https://huggingface.co/ZTE-AIM)] [[Dataset 🤗](https://huggingface.co/datasets/ZTE-AIM/Curr-ReFT-data)] [[Code 💻](https://github.com/ding523/Curr_REFT)]

* [2503] [VisualThinker-R1-Zero] [R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model](https://arxiv.org/abs/2503.05132)  [[Code 💻](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)]

* [2503] [Vision-R1] [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](https://arxiv.org/abs/2503.06749) [[Code 💻](https://github.com/Osilly/Vision-R1)]

* [2503] [Seg-Zero] [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520) [[Model 🤗](https://huggingface.co/Ricky06662/Seg-Zero-7B)] [[Dataset 🤗](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840)] [[Code 💻](https://github.com/dvlab-research/Seg-Zero)]

* [2503] [MM-Eureka] [MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf) [[Models 🤗](https://huggingface.co/FanqingM)] [[Dataset 🤗](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)] [[Code 💻](https://github.com/ModalMinds/MM-EUREKA)]

* [2503] [Visual-RFT] [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) [[Project 🌐](https://github.com/Liuziyu77/Visual-RFT)] [[Datasets 🤗](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df)] [[Code 💻](https://github.com/Liuziyu77/Visual-RFT)]
  
* [2501] [Kimi k1.5] [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) [[Project 🌐](https://github.com/MoonshotAI/Kimi-k1.5)]
  
* [2501] [Mulberry] [Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search](https://arxiv.org/abs/2412.18319) [[Model 🤗](https://huggingface.co/HuanjinYao/Mulberry_llava_8b)] [[Code 💻](https://github.com/HJYao00/Mulberry)]

* [2501] [Virgo] [Virgo: A Preliminary Exploration on Reproducing o1-like MLLM](https://arxiv.org/abs/2501.01904v2) [[Model 🤗](https://huggingface.co/RUC-AIBOX/Virgo-72B)] [[Code 💻](https://github.com/RUCAIBox/Virgo)]
  
* [2501] [Text-to-image COT] [Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step](https://arxiv.org/abs/2501.13926) [[Project 🌐](https://github.com/ZiyuGuo99/Image-Generation-CoT)] [[Model 🤗](https://huggingface.co/ZiyuG/Image-Generation-CoT)]  [[Code 💻](https://github.com/ZiyuGuo99/Image-Generation-CoT)]

* [2411] [InternVL2-MPO] [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://arxiv.org/abs/2411.10442) [[Project 🌐](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/)] [[Model 🤗](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO)] [[Code 💻](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo)]

* [2411] [Insight-V] [Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](https://arxiv.org/abs/2411.14432) [[Model 🤗](https://huggingface.co/collections/THUdyh/insight-v-673f5e1dd8ab5f2d8d332035)] [[Code 💻](https://github.com/dongyh20/Insight-V)]

### Vision (Video)📹 

* [2505] [SpaceR] [SpaceR: Reinforcing MLLMs in Video Spatial Reasoning](https://arxiv.org/pdf/2504.01805v2) [[Model 🤗](https://huggingface.co/RUBBISHLIKE/SpaceR)] [[Dataset 🤗](https://huggingface.co/datasets/RUBBISHLIKE/SpaceR-151k)] [[Code 💻]( https://github.com/OuyangKun10/SpaceR)]

* [2504] [TinyLLaVA-Video-R1] [TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning](https://arxiv.org/abs/2504.09641)  [[Model 🤗](https://huggingface.co/Zhang199/TinyLLaVA-Video-R1)] [[Code 💻](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)]

* [2504] [VideoChat-R1] [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.06958)  [[Model 🤗](https://huggingface.co/collections/OpenGVLab/videochat-r1-67fbe26e4eb08c83aa24643e)] [[Code 💻](https://github.com/OpenGVLab/VideoChat-R1)]

* [2504] [Spatial-R1] [Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning](https://arxiv.org/abs/2504.01805) [[Code 💻](https://github.com/OuyangKun10/Spatial-R1)]

* [2504] [R1-Zero-VSI] [Improved Visual-Spatial Reasoning via R1-Zero-Like Training](https://arxiv.org/abs/2504.00883) [[Code 💻](https://github.com/zhijie-group/R1-Zero-VSI)]

* [2503] [SEED-Bench-R1] [Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1](https://arxiv.org/abs/2503.24376) [[Dataset 🤗](https://huggingface.co/datasets/TencentARC/SEED-Bench-R1)]  [[Code 💻](https://github.com/TencentARC/SEED-Bench-R1)]

* [2503] [Video-R1] [Video-R1: Reinforcing Video Reasoning in MLLMs](https://arxiv.org/abs/2503.21776) [[Model 🤗](https://huggingface.co/Video-R1/Video-R1-7B)] [[Dataset 🤗](https://huggingface.co/datasets/Video-R1/Video-R1-data)] [[Code 💻](https://github.com/tulerfeng/Video-R1)]

* [2503] [TimeZero] [TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM](https://arxiv.org/abs/2503.13377) [[Model 🤗](https://huggingface.co/wwwyyy/TimeZero-Charades-7B)] [[Code 💻](https://github.com/www-Ye/TimeZero)]

### Medical Vision🏥 

* [2505] [Patho-R1] [Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner](https://arxiv.org/abs/2505.11404) [[Code 💻](https://github.com/Wenchuan-Zhang/Patho-R1)]

* [2504] [ChestX-Reasoner] [ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification](https://arxiv.org/pdf/2504.20930) 

* [2503] [Med-R1] [Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models](https://arxiv.org/abs/2503.13939v3) [[Model 🤗](https://huggingface.co/yuxianglai117/Med-R1)]  [[Code 💻](https://github.com/Yuxiang-Lai117/Med-R1)]

* [2502] [MedVLM-R1] [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning](https://arxiv.org/abs/2502.19634) [[Model 🤗](https://huggingface.co/JZPeterPan/MedVLM-R1)]

### Embodied Vision🤖 

* [2504] [Embodied-R] [Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning](https://arxiv.org/abs/2504.12680) [[Code 💻](https://github.com/EmbodiedCity/Embodied-R.code)]

* [2503] [Embodied-Reasoner] [Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks](https://arxiv.org/abs/2503.21696v1) [[Project 🌐](https://embodied-reasoner.github.io/)] [[Dataset 🤗](https://huggingface.co/datasets/zwq2018/embodied_reasoner)] [[Code 💻](https://github.com/zwq2018/embodied_reasoner)]

### Multimodal Reward Model 💯

* [2505] [Skywork-VL Reward] [Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding and Reasoning](https://arxiv.org/abs/2505.07263) [[Models 🤗](https://huggingface.co/Skywork/Skywork-VL-Reward-7B)] [[Code 💻](https://github.com/SkyworkAI/Skywork-R1V)]

* [2505] [UnifiedReward-Think] [Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.03318)  [[Project 🌐](https://codegoat24.github.io/UnifiedReward/think)] [[Models 🤗](https://huggingface.co/collections/CodeGoat24/unifiedreward-models-67c3008148c3a380d15ac63a)] [[Datasets 🤗](https://huggingface.co/collections/CodeGoat24/unifiedreward-training-data-67c300d4fd5eff00fa7f1ede)] [[Code 💻](https://github.com/CodeGoat24/UnifiedReward)]

* [2505] [R1-Reward] [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835) [[Model 🤗](https://huggingface.co/yifanzhang114/R1-Reward)]  [[Dataset 🤗](https://huggingface.co/datasets/yifanzhang114/R1-Reward-RL)]  [[Code 💻](https://github.com/yfzhang114/r1_reward)]

### Audio👂

* [2504] [SARI] [SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning](https://arxiv.org/pdf/2504.15900)  

* [2503] [R1-AQA] [Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering](https://arxiv.org/abs/2503.11197v2) [[Model 🤗](https://huggingface.co/mispeech/r1-aqa)]  [[Code 💻](https://github.com/xiaomi-research/r1-aqa)]

* [2503] [Audio-Reasoner] [Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models](https://arxiv.org/abs/2503.02318) [[Project 🌐](https://xzf-thu.github.io/Audio-Reasoner/)] [[Model 🤗](https://huggingface.co/zhifeixie/Audio-Reasoner)]  [[Code 💻](https://github.com/xzf-thu/Audio-Reasoner)]

### Omni☺️

* [2505] [Omni-R1 (ZJU)] [Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration](https://arxiv.org/abs/2505.20256)  [Project 🌐](https://aim-uofa.github.io/OmniR1/) [Model 🤗](https://huggingface.co/Haoz0206/Omni-R1) [Code 💻](https://github.com/aim-uofa/Omni-R1)

* [2505] [Omni-R1 (MIT)] [Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?](https://arxiv.org/pdf/2505.09439)  

* [2505] [EchoInk-R1] [EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.04623) [[Model 🤗](https://huggingface.co/harryhsing/EchoInk-R1-7B)] [[Dataset 🤗](https://huggingface.co/datasets/harryhsing/OmniInstruct_V1_AVQA_R1)] [[Code 💻](https://github.com/HarryHsing/EchoInk)]

* [2503] [R1-Omni] [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379)  [[Model 🤗](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)] [[Code 💻](https://github.com/HumanMLLM/R1-Omni)]

### GUI Agent📲

* [2505] [ARPO] [ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay](https://www.arxiv.org/pdf/2505.16282)  [[Model 🤗](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B)]  [[Code 💻](https://github.com/dvlab-research/ARPO)]

* [2505] [GUI-G1] [GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents](https://arxiv.org/abs/2505.15810) [[Code 💻](https://github.com/Yuqi-Zhou/GUI-G1)]

* [2505] [UIShift] [UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning](https://arxiv.org/pdf/2505.12493) 

* [2505] [MobileIPL] [Enhance Mobile Agents Thinking Process Via Iterative Preference Learning](https://arxiv.org/pdf/2505.12299) 

* [2504] [InfiGUI-R1] [InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/pdf/2504.14239) [[Model 🤗](https://huggingface.co/Reallm-Labs/InfiGUI-R1-3B)]  [[Code 💻](https://github.com/Reallm-Labs/InfiGUI-R1)]

* [2504] [GUI-R1] [GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/abs/2504.10458) [[Model 🤗](https://huggingface.co/ritzzai/GUI-R1)]  [[Dataset 🤗](https://huggingface.co/datasets/ritzzai/GUI-R1)]  [[Code 💻](https://github.com/ritzz-ai/GUI-R1)]

* [2503] [UI-R1] [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/abs/2503.21620)

### Web Agent🌏

* [2505] [Web-Shepherd] [Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](https://arxiv.org/abs/2505.15277) [[Models 🤗](https://huggingface.co/collections/LangAGI-Lab/web-shepherd-advancing-prms-for-reinforcing-web-agents-682b4f4ad607fc27c4dc49e8)] [[Datasets 🤗](https://huggingface.co/collections/LangAGI-Lab/web-shepherd-advancing-prms-for-reinforcing-web-agents-682b4f4ad607fc27c4dc49e8)] [[Code 💻](https://github.com/kyle8581/Web-Shepherd)]

### Autonomous Driving🚙

* [2505] [AgentThink] [AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2505.15298)

### Metaverse🌠

* [2503] [MetaSpatial] [MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](https://arxiv.org/abs/2503.18470) [[Dataset 🤗](https://huggingface.co/datasets/zhenyupan/3d_layout_reasoning)] [[Code 💻](https://github.com/PzySeere/MetaSpatial)]


## Benchmarks📊

* [2505] [ReasonMap] [Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps](https://arxiv.org/abs/2505.18675) [[Project 🌐](https://fscdc.github.io/Reason-Map/)] [[🤗 Dataset](https://huggingface.co/datasets/FSCCS/ReasonMap)] [[💻 Code](https://github.com/fscdc/ReasonMap)] 

* [2505] [PhyX] [PhyX: Does Your Model Have the "Wits" for Physical Reasoning?](https://arxiv.org/abs/2505.15929) [[Project 🌐](https://phyx-bench.github.io/)] [[🤗 Dataset](https://huggingface.co/datasets/Cloudriver/PhyX)] [[💻 Code](https://github.com/NastyMarcus/PhyX)] 

* [2505] [GDI-Bench] [GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling](https://www.arxiv.org/abs/2505.00063)

* [2504] [VisuLogic] [VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models](http://arxiv.org/abs/2504.15279)  [[Project 🌐](https://visulogic-benchmark.github.io/VisuLogic)] [[🤗 Dataset](https://huggingface.co/datasets/VisuLogic/VisuLogic)] [[💻 Code](https://github.com/VisuLogic-Benchmark)] 

* [2504] [Video-MMLU] [Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark](https://arxiv.org/abs/2504.14693) [[Project 🌐](https://enxinsong.com/Video-MMLU-web/)] [[🤗 Dataset](https://huggingface.co/datasets/Enxin/Video-MMLU)] [[💻 Code](https://github.com/Espere-1119-Song/Video-MMLU)] 

* [2504] [GeoSense] [GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning](https://arxiv.org/abs/2504.12597)

* [2504] [VCR-Bench] [VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning](https://arxiv.org/abs/2504.07956) 
 [[Project 🌐](https://vlm-reasoning.github.io/VCR-Bench/)] [[Dataset 🤗](https://huggingface.co/datasets/VLM-Reasoning/VCR-Bench)] [[Code 💻](https://github.com/zhishuifeiqian/VCR-Bench)]

* [2504] [MDK12-Bench] [MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2504.05782) [[Code 💻](https://github.com/LanceZPF/MDK12)]

* [2503] [V1-33K] [V1: Toward Multimodal Reasoning by Designing Auxiliary Tasks] [[Project 🌐](https://github.com/haonan3/V1)] [[Dataset 🤗](https://huggingface.co/datasets/haonan3/V1-33K)] [[Code 💻](https://github.com/haonan3/V1)]

* [2502] [MM-IQ] [MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models](https://arxiv.org/abs/2502.00698)  [[Project 🌐](https://acechq.github.io/MMIQ-benchmark/)] [[Dataset 🤗](https://huggingface.co/datasets/huanqia/MM-IQ)] [[Code 💻](https://github.com/AceCHQ/MMIQ)] 

* [2502] [MME-CoT] [MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency](https://arxiv.org/abs/2502.09621)  [[Project 🌐](https://mmecot.github.io/)] [[Dataset 🤗](https://huggingface.co/datasets/CaraJ/MME-CoT)] [[Code 💻](https://github.com/CaraJ7/MME-CoT)]

* [2502] [ZeroBench] [ZeroBench: An Impossible* Visual Benchmark for Contemporary Large Multimodal Models](https://arxiv.org/abs/2502.09696)  [[Project 🌐](https://zerobench.github.io/)] [[Dataset 🤗](https://huggingface.co/datasets/jonathan-roberts1/zerobench)] [[Code 💻](https://github.com/jonathan-roberts1/zerobench/)]

* [2502] [HumanEval-V] [HumanEval-V: Benchmarking High-Level Visual Reasoning with Complex Diagrams in Coding Tasks](https://arxiv.org/abs/2410.12381) [[Project 🌐](https://humaneval-v.github.io/)] [[Dataset 🤗](https://huggingface.co/datasets/HumanEval-V/HumanEval-V-Benchmark)] [[Code 💻](https://github.com/HumanEval-V/HumanEval-V-Benchmark)]

## Open-Source Projects (Repos without Paper)🌐

### Training Framework 🗼

* [EasyR1 💻](https://github.com/hiyouga/EasyR1)  ![EasyR1](https://img.shields.io/github/stars/hiyouga/EasyR1) (An Efficient, Scalable, Multi-Modality RL Training Framework)

### Vision (Image) 👀

* [R1-V 💻](https://github.com/Deep-Agent/R1-V)  ![R1-V](https://img.shields.io/github/stars/Deep-Agent/R1-V) [Blog 🎯](https://deepagent.notion.site/rlvr-in-vlms) [Datasets 🤗](https://huggingface.co/collections/MMInstruction/r1-v-67aae24fa56af9d2e2755f82)

* [Multimodal Open R1 💻](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)  ![Multimodal Open R1](https://img.shields.io/github/stars/EvolvingLMMs-Lab/open-r1-multimodal) [Model 🤗](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) [Dataset 🤗](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified)

* [MMR1 💻](https://github.com/LengSicong/MMR1) ![LengSicong/MMR1](https://img.shields.io/github/stars/LengSicong/MMR1) [Code 💻](https://github.com/LengSicong/MMR1) [Model 🤗](https://huggingface.co/MMR1/MMR1-Math-v0-7B) [Dataset 🤗](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) 

* [R1-Multimodal-Journey 💻](https://github.com/FanqingM/R1-Multimodal-Journey) ![R1-Multimodal-Journey](https://img.shields.io/github/stars/FanqingM/R1-Multimodal-Journey) (Latest progress at [MM-Eureka](https://github.com/ModalMinds/MM-EUREKA))

* [R1-Vision 💻](https://github.com/yuyq96/R1-Vision) ![R1-Vision](https://img.shields.io/github/stars/yuyq96/R1-Vision) [Cold-Start Datasets 🤗](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83)

* [Ocean-R1 💻](https://github.com/VLM-RL/Ocean-R1)  ![Ocean-R1](https://img.shields.io/github/stars/VLM-RL/Ocean-R1) [Models 🤗](https://huggingface.co/minglingfeng) [Datasets 🤗](https://huggingface.co/minglingfeng)

* [R1V-Free 💻](https://github.com/Exgc/R1V-Free)  ![Exgc/R1V-Free](https://img.shields.io/github/stars/Exgc/R1V-Free) [Models 🤗](https://huggingface.co/collections/Exgc/r1v-free-67f769feedffab8761b8f053) [Dataset 🤗](https://huggingface.co/datasets/Exgc/R1V-Free_RLHFV)

* [SeekWorld 💻](https://github.com/TheEighthDay/SeekWorld)  ![TheEighthDay/SeekWorld](https://img.shields.io/github/stars/TheEighthDay/SeekWorld) [Model 🤗](https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS) [Dataset 🤗](https://huggingface.co/datasets/TheEighthDay/SeekWorld) [Demo 🤗](https://huggingface.co/spaces/TheEighthDay/SeekWorld_APP)

* [R1-Track 💻](https://github.com/Wangbiao2/R1-Track)  ![Wangbiao2/R1-Track](https://img.shields.io/github/stars/Wangbiao2/R1-Track) [Models 🤗](https://huggingface.co/WangBiao) [Datasets 🤗](https://huggingface.co/WangBiao)

### Vision (Video)📹 

* [Open R1 Video 💻](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![Open R1 Video](https://img.shields.io/github/stars/Wang-Xiaodong1899/Open-R1-Video) [Models 🤗](https://huggingface.co/Xiaodong/Open-R1-Video-7B)  [Datasets 🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)

* [Temporal-R1 💻](https://github.com/appletea233/Temporal-R1)  ![Temporal-R1](https://img.shields.io/github/stars/appletea233/Temporal-R1) [Code 💻](https://github.com/appletea233/Temporal-R1) [Models 🤗](https://huggingface.co/appletea2333)

* [Open-LLaVA-Video-R1 💻](https://github.com/Hui-design/Open-LLaVA-Video-R1) ![Open-LLaVA-Video-R1](https://img.shields.io/github/stars/Hui-design/Open-LLaVA-Video-R1) [Code 💻](https://github.com/Hui-design/Open-LLaVA-Video-R1)

### Agent 👥

* [VAGEN 💻](https://github.com/RAGEN-AI/VAGEN) ![VAGEN](https://img.shields.io/github/stars/RAGEN-AI/VAGEN) [Code 💻](https://github.com/RAGEN-AI/VAGEN)

## Contribution and Acknowledgment❤️

This is an active repository and your contributions are always welcome! If you have any question about this opinionated list, do not hesitate to contact me sun-hy23@mails.tsinghua.edu.cn. 

I extend my sincere gratitude to all community members who provided valuable supplementary support.

## Citation📑

If you find this repository useful for your research and applications, please star us ⭐ and consider citing:

```tex
@misc{sun2025reinforcementfinetuningpowersreasoning,
      title={Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models}, 
      author={Haoyuan Sun and Jiaqi Wu and Bo Xia and Yifu Luo and Yifei Zhao and Kai Qin and Xufei Lv and Tiantian Zhang and Yongzhe Chang and Xueqian Wang},
      year={2025},
      eprint={2505.18536},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18536}, 
}
```
and
```tex
@misc{sun2025RL-Reasoning-MLLMs,
  title={Awesome RL-based Reasoning MLLMs},
  author={Haoyuan Sun, Xueqian Wang},
  year={2025},
  howpublished={\url{https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs}},
  note={Github Repository},
}
```

##  Star Chart⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs)](https://star-history.com/#Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs&Date)

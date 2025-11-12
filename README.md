# CastX: Cohort-Level Causal Inference Meets Statistical Testing for Faithful and Reliable GNN Explanations


## 🧩 Overview

**CastX** is a unified framework that integrates **cohort-level causal inference** with **rigorous statistical causality testing** to generate **faithful** and **statistically reliable explanations** for Graph Neural Networks (GNNs).

Unlike existing GNN explainers that rely on individual-level interventions, CastX introduces:
- **Cohort-level Conditional Average Treatment Effect (CATE)** estimation to guide explanatory subgraph discovery.
- A **reinforcement learning–based dynamic edge pruning** process to iteratively remove spurious edges.
- A **non-parametric permutation test** to assess the statistical significance of each explanatory edge.

<p align="center">
  <img src="docs/castx_framework.png" width="650"/>
</p>

---

## 🚀 Key Features

- **Cohort-level causal estimation:** Enhances the *faithfulness* of explanations by mitigating individual-level noise.  
- **RL-based edge pruning:** Sequentially identifies causally informative substructures.  
- **Statistical validation:** Non-parametric permutation testing ensures *reliability* and *statistical rigor*.  
- **Superior performance:** Outperforms existing explainers on Mutagenicity, REDDIT-MULTI-5K, and Visual Genome datasets.

---

## Plan of Attack

### Project Structure
All experiments will follow a consistent and reproducible folder layout:

```

notebooks/
└── ml_experiments/
├── baseline/
├── traditional_ml/
│   ├── linear_regression/
│   ├── svr/
│   └── tree_based/
├── deep_learning/
│   ├── mlp/
│   └── cnn_or_task_specific/
└── llm_finetuning/
└── openai_4o_mini/

```

Each folder contains:
- A dedicated notebook for experiments  
- Configuration and hyperparameters  
- Evaluation outputs and comparison notes  

---

1. **Establish a Strong Baseline**  
   Begin by defining clear evaluation metrics and implementing a simple baseline model (`baseline/`).  
   This serves as the reference point for all subsequent comparisons.

2. **Explore Traditional Machine Learning Approaches**  
   Implement classical ML models such as linear regression, SVR, and tree-based methods (`traditional_ml/[ALGO_NAME]/`).  
   Evaluate their performance and learning behavior relative to the baseline.

3. **Move to Deep Learning Models**  
   Experiment with neural network–based architectures suited to the task (`deep_learning/[MODEL_NAME]/`).  
   Compare results against both baseline and traditional ML approaches to assess gains from model capacity.

4. **Fine-tune a Frontier LLM**  
   Fine-tune a state-of-the-art Large Language Model, specifically **OpenAI 4o Mini** (`llm_finetuning/openai_4o_mini/`).  
   Maintain consistent data splits, metrics, and evaluation protocols for fair comparison.

5. **Comprehensive Comparison & Analysis**  
   Compare all approaches—baseline, traditional ML, deep learning, and LLM fine-tuning.  
   Evaluate not only performance metrics but also robustness, generalization, compute cost, and scalability.

6. **Iterate and Refine**  
   Use insights from the analysis to refine the most promising approach and perform targeted optimizations where necessary.


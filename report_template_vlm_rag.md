# Does VLM-RAG Reduce Hallucination in Document Visual Question Answering?

## 1. Introduction
Hallucination remains a major challenge for vision-language models (VLMs), especially in document question answering tasks where models may produce plausible but unsupported claims. This project studies whether retrieval-augmented generation (RAG) can reduce hallucination by grounding answers in retrieved visual-textual evidence from source documents.

**Research Question**: Does VLM-RAG reduce hallucination compared with a non-retrieval VLM baseline in document visual question answering?

## 2. Method

### 2.1 Experimental Setup
- **Task**: Document visual question answering
- **Baseline**: VLM answers directly from prompt (no retrieval context)
- **VLM-RAG**: Retrieve relevant page/region/text chunks, then answer using retrieved evidence
- **Dataset size**: 20-30 questions

### 2.2 Model and Parameters
- **Model(s)**: [Fill in]
- **Temperature**: [Fill in]
- **Max tokens**: [Fill in]
- **Retriever / embedding method**: [Fill in]
- **Knowledge source**: [Fill in: PDF pages, screenshots, reports, etc.]

### 2.3 Evaluation Criteria
Each answer is annotated with one of the following labels:
- **Correct**: Answer is supported by source evidence and factually correct
- **Partially Correct**: Some correct content, but incomplete or slightly inaccurate
- **Hallucinated / Incorrect**: Unsupported or wrong claim

We report:
- **Accuracy** = Correct / Total
- **Hallucination Rate** = Hallucinated / Total

## 3. Results

### 3.1 Quantitative Results
| Method | Correct | Partial | Hallucinated | Accuracy | Hallucination Rate |
|---|---:|---:|---:|---:|---:|
| Baseline VLM | [ ] | [ ] | [ ] | [ ] | [ ] |
| VLM-RAG | [ ] | [ ] | [ ] | [ ] | [ ] |

### 3.2 Key Observations
1. [Observation about hallucination difference]
2. [Observation about answer completeness]
3. [Observation about retrieval failure cases]

## 4. Case Analysis

### Case A (RAG helps)
- **Question**: [Fill in]
- **Baseline output**: [Fill in]
- **RAG output**: [Fill in]
- **Why RAG helped**: [Fill in]

### Case B (Both fail)
- **Question**: [Fill in]
- **Baseline output**: [Fill in]
- **RAG output**: [Fill in]
- **Failure reason**: [Fill in: retrieval miss/OCR error/ambiguous question]

### Case C (RAG introduces noise)
- **Question**: [Fill in]
- **Baseline output**: [Fill in]
- **RAG output**: [Fill in]
- **Failure reason**: [Fill in: irrelevant context/conflicting evidence]

## 5. Discussion
The results suggest that VLM-RAG [improves / does not significantly improve] factual grounding in document QA. When retrieval quality is high, hallucinations decrease because the model is constrained by evidence. However, retrieval errors and noisy OCR can still propagate incorrect signals into generation.

**Main limitations**:
- Small sample size (20-30 questions)
- Manual annotation subjectivity
- Single model setting (limited generalization)

## 6. Conclusion
This project evaluates VLM-RAG for hallucination mitigation in document visual question answering. In our setting, VLM-RAG [reduced / did not reduce] hallucination rate from [X]% to [Y]%. Future work includes larger datasets, stronger retrievers, and automatic factuality verification.

## 7. Reproducibility Checklist
- [ ] Prompts saved
- [ ] Model versions recorded
- [ ] Parameter settings recorded
- [ ] Retrieved evidence snapshots saved
- [ ] Annotation sheet completed


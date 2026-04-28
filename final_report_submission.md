# Final Assessment Report  
## Evaluating and Reducing Hallucination in Vision-Language Image Captioning with Progressive RAG Guardrails

## 1. Introduction

Large vision-language models (VLMs) can produce fluent, plausible captions for images, but they still frequently hallucinate: they mention objects or attributes that are not present in the image. This limits reliability in practical deployments such as assistive systems, surveillance summaries, accessibility tools, and autonomous pipelines where factual correctness is critical.  

In my midterm report, I identified hallucination mitigation as a practical and feasible computer vision problem for the course timeline. The final assessment extends that direction into a full implementation-and-iteration study. Instead of only proposing methods conceptually, I implemented multiple pipelines, evaluated each one with the same metric protocol, analyzed failure modes, and iteratively improved the system design.

The core question in this final assessment is:

**Can retrieval-augmented generation (RAG) reduce object-level hallucination in image captioning compared with a non-retrieval baseline?**

To answer this question, I designed and ran a sequence of experiments:

1. A non-RAG baseline and prompt-constrained baseline.
2. A naive image-RAG pipeline (CLIP retrieval + BLIP generation).
3. Guarded and safety-gated RAG variants.
4. Split-corpus RAG (train retrieval, val evaluation) to reduce retrieval leakage.
5. A v6 minimal pipeline inspired by recent literature: coarse retrieval + rerank, uncertainty-triggered RAG, object whitelist decoding, and verifier fallback.

The key contribution of this report is not a single best score, but a **well-documented iterative process** that demonstrates:

- which RAG choices fail and why,
- which safety mechanisms stabilize performance,
- and what remains necessary to exceed baseline performance consistently.

This is aligned with real-world model development: robust systems are built through controlled experimentation, failure analysis, and staged safeguards rather than one-shot improvements.

---

## 2. Literature Review

### 2.1 Hallucination in VLM Captioning

Image captioning models have improved from CNN-RNN architectures to transformer-based VLMs (e.g., BLIP family), but hallucination remains a persistent issue. Prior work and benchmarks show that generated captions can contain objects not grounded in the image. This becomes severe when models rely too heavily on language priors (e.g., likely co-occurrences of objects) instead of visual evidence.

Traditional mitigation directions include:

- constrained decoding,
- additional fine-tuning,
- external grounding modules,
- and retrieval augmentation.

### 2.2 Why RAG for Vision?

Text RAG has become a standard hallucination mitigation strategy in language models by injecting external evidence at generation time. The analogous hypothesis in vision-language settings is that retrieving similar visual examples (or associated textual descriptions) can improve grounding and reduce unsupported claims.

However, multimodal RAG introduces extra failure modes:

- retrieval mismatch (visually similar but semantically wrong evidence),
- noisy evidence conversion,
- and incorrect evidence fusion in generation.

Recent literature in multimodal RAG and LVLM hallucination mitigation supports this: naive retrieval is often insufficient, and strong reranking/verification/gating are required.

### 2.3 Relevant Recent Methods and Their Implications

From recent papers and technical sources reviewed during implementation:

- **Active Retrieval Augmentation (ARA)** highlights that retrieval should be selective and uncertainty-aware, not always-on.
- **Reasoning-Retrieval-Reranking frameworks (e.g., R3G)** emphasize two-stage retrieval and reranking rather than direct top-k use.
- **CLIP-guided or calibration-aware decoding** indicates that external consistency signals can improve grounding without full retraining.
- **Evidence-guided multimodal reasoning frameworks** stress that poor evidence integration can worsen hallucination.

These ideas directly motivated my v5 and v6 changes:

- retrieval/evaluation split,
- small reranking stage,
- uncertainty-triggered retrieval use,
- fallback-to-baseline gate,
- and object-level whitelist constraints.

---

## 3. Discussion on Data and Chosen Framework

### 3.1 Dataset Selection

I used the **MS COCO 2017** dataset because it is a standard benchmark for image captioning and includes annotations suitable for object-level hallucination analysis.

Used files:

- `val2017` images (evaluation images),
- `captions_val2017.json` (caption references),
- `instances_val2017.json` (ground-truth object categories per image),
- `train2017` images + `captions_train2017.json` (retrieval corpus in split-RAG experiments).

### 3.2 Why COCO Is Suitable

COCO is appropriate for this project because:

- it contains realistic scenes with multiple objects,
- object categories are standardized,
- and annotations support direct grounding checks.

This allows operational hallucination measurement at object-level granularity:

- If a generated caption mentions a category not present in `instances` labels for that image, that mention is counted as hallucinated.

### 3.3 Potential Biases and Limitations

- COCO categories are finite and coarse; some semantically valid mentions may not map cleanly.
- Lexical matching can miss synonyms/paraphrases.
- Caption style bias in training data can affect model priors.
- The metric focuses on object mentions; it does not fully evaluate relational or attribute truthfulness.

Despite these limitations, COCO remains a strong fit for controlled, reproducible hallucination experiments.

---

## 4. System Design

### 4.1 Base Architecture

I used pretrained models only (no training/fine-tuning):

- **Generator**: `Salesforce/blip-image-captioning-base`
- **Retriever Embedding Model**: `openai/clip-vit-base-patch32`

Framework and libraries:

- PyTorch
- Hugging Face Transformers
- pandas / tqdm / PIL

### 4.2 Evaluation Metrics

For each method, I report:

- **Hallucination Rate**: fraction of examples with at least one hallucinated object mention
- **Average Hallucinated Mentions**
- **Object Precision**: grounded mentioned objects / total mentioned objects

### 4.3 Pipeline Variants

Implemented methods:

1. `baseline`: BLIP caption from image only  
2. `grounded` prompt baseline (early experiment)
3. `rag`: naive retrieval-augmented captioning
4. `rag_guarded`: consensus-constrained prompting
5. `rag_safe_gate`: candidate selection with CLIP-based safety gate + fallback
6. `rag_v6_pipeline`: v6 minimal method with:
   - coarse retrieval + reranking,
   - uncertainty-triggered retrieval use,
   - object whitelist decoding,
   - final verifier + fallback

### 4.4 Reproducibility and Evidence Logging

To support reproducibility:

- each run stores `run_config_rag.json`,
- results stored in `results_rag.csv`,
- summaries in `summary_rag.csv`,
- case labeling via `annotate_bad_cases.py`:
  - `good_case`, `bad_case`, `neutral_case`
  - unique `case_id = experiment_name__method__image_id`

This produced direct evidence files used in analysis and report examples.

---

## 5. Details and Discussion of the Implementation Process

This section documents the actual iterative process from initial implementation to final v6.

### 5.1 Stage A: Prompt-based Baseline (Non-RAG)

First, I built a simple captioning benchmark (`run_caption_experiment.py`) to ensure metric correctness and pipeline stability.

Initial comparison (`sample_size=30`):

- `baseline`: hallucination 0.0333, precision 0.9583
- `grounded prompt`: hallucination 0.0000, precision 1.0000

This stage validated:

- data loading,
- mention extraction,
- object-level comparison metrics,
- and CSV summarization scripts.

### 5.2 Stage B: True RAG v1 (Naive Retrieval Augmentation)

I implemented a true RAG variant:

- CLIP image embeddings for retrieval,
- top-k similar images,
- retrieved reference captions injected as evidence prompt,
- BLIP generation over image + evidence prompt.

Result (`outputs_rag`, sample 30):

- `rag` hallucination jumped to 0.9000

This was a major regression and immediately exposed a key failure mode:

**naive retrieval evidence was highly noisy and directly polluted generation.**

### 5.3 Stage C: Guarding and Gating (v2-v4)

To address failure, I added:

- consensus-based constraints (`rag_guarded`),
- safety-gate candidate selection with fallback (`rag_safe_gate`),
- margin threshold (`safe_margin`) to avoid unnecessary switching.

Observed behavior:

- `rag_guarded` alone did not recover performance in early versions.
- `rag_safe_gate` consistently reduced catastrophic degradation and returned near baseline.

Interpretation:

- guardrails are necessary to prevent retrieval-induced hallucinations.
- retrieval quality remained the core bottleneck.

### 5.4 Stage D: Split Retrieval Corpus (v5)

I then changed a key design assumption:

- retrieval corpus from `train2017`,
- evaluation on `val2017`.

This better matches realistic deployment and avoids same-pool retrieval effects.

I also added `.pt` retrieval embedding caching for speed and reproducibility:

- first run builds cache,
- subsequent runs load cache directly.

v5 results (`sample_size=50`):

- baseline: 0.08 hallucination, 0.8857 precision
- rag: 0.56, 0.5196
- rag_guarded: 0.58, 0.5146
- rag_safe_gate: 0.14, 0.84

Findings:

- split retrieval did not make naive RAG competitive;
- safety-gated RAG remained much better than naive RAG but still below baseline.

### 5.5 Stage E: v6 Minimal Literature-Inspired Pipeline

Following reviewed literature patterns, I implemented v6 with:

1. **Top-50 coarse retrieval + rerank to top-5**
2. **Uncertainty-triggered RAG** (only activate retrieval rewrite under low baseline confidence)
3. **Object whitelist constraints**
4. **Verifier + fallback**

v6 results (`outputs_rag_v6`, `sample_size=50`):

- baseline: 0.08, precision 0.8857
- rag: 0.48, precision 0.5882
- rag_guarded: 0.48, precision 0.5882
- rag_safe_gate: 0.18, precision 0.7885
- rag_v6_pipeline: 0.14, precision 0.8043

Compared to v5:

- `rag` improved (0.56 -> 0.48),
- `rag_v6_pipeline` improved over `rag_safe_gate` in hallucination rate (0.14 vs 0.18),
- but still not better than baseline.

### 5.6 Practical Engineering Improvements Completed

During implementation I also resolved practical issues:

- prompt echo handling in generation output,
- robust output directory creation,
- better runtime errors for invalid retrieval paths,
- embedding cache metadata validation,
- case annotation exports for report evidence.

These changes improved reproducibility and made experimentation feasible within time and compute limits.

---

## 6. Discussion of System Performance, Identified Improvements, and Effects

### 6.1 Initial Evaluation and Motivation for Improvements

The earliest true-RAG results showed dramatic degradation. This initial analysis motivated all later changes:

- evidence quality needed reranking/filtering,
- retrieval should not be always-on,
- final generation must include fallback safeguards.

This directly informed the v4-v6 roadmap.

### 6.2 Final Evaluation and Comparative Analysis

On the final v6 run:

- `baseline` remains best on both primary metrics.
- `rag_v6_pipeline` is substantially better than naive RAG but still below baseline.

Performance trend across versions indicates:

- naive RAG is fragile in caption grounding tasks,
- incremental guardrails reduce harm,
- but current retrieval+fusion quality is insufficient for consistent baseline outperformance.

### 6.3 Strengths of Finalized System

1. **Strong reproducibility**  
   - clear scripts, run configs, CSV outputs, case tags.

2. **Robust failure analysis workflow**  
   - bad-case extraction and method-wise diagnosis.

3. **Progressive engineering safeguards**  
   - gating, fallback, uncertainty trigger, cache.

4. **Evidence-driven iteration**  
   - each improvement was motivated by observed failure patterns.

### 6.4 Weaknesses of Finalized System

1. **Retriever semantic precision remains limited**  
   CLIP image similarity alone often retrieves contextually wrong evidence.

2. **Caption-level evidence is noisy**  
   Retrieved captions can inject unrelated objects.

3. **Lexical object matching is simplistic**  
   It may undercount semantic correctness or over-penalize wording variations.

4. **No train-time adaptation**  
   The system relies entirely on inference-time controls.

### 6.5 Why Improvements Helped but Did Not Surpass Baseline

The guardrail components reduced catastrophic failure by refusing low-confidence RAG replacements. That improves stability, but also means the system often falls back to baseline. Therefore, gains are mostly in risk control rather than absolute accuracy gains.

To exceed baseline, evidence quality must become substantially better before generation, not only filtered after generation.

### 6.6 Future Improvements if Revisited

If revisiting this system, the highest-priority upgrades would be:

1. **Stronger reranking**
   - CLIP recall + cross-modal reranker/judge for top-k evidence.

2. **Region-level retrieval**
   - object/ROI-aware retrieval instead of whole-image global similarity.

3. **Hybrid retrieval signals**
   - combine dense retrieval with lexical/object constraints.

4. **Edit-based caption correction**
   - revise baseline by controlled object edits only, disallow free-form additions.

5. **Calibration and abstention policy**
   - explicit confidence calibration and “I don’t know” mode for low support.

6. **Broader evaluation**
   - add semantic metrics and human evaluation for qualitative grounding quality.

---

## 7. Conclusion and Future Directions

This final project implemented and evaluated multiple strategies to reduce hallucination in VLM image captioning, moving from a simple baseline to increasingly sophisticated RAG-based pipelines with guardrails.

The most important empirical outcome is:

- **naive RAG strongly worsened hallucination**,  
- **guarded/gated RAG reduced that harm**,  
- **but baseline remained strongest in this setup**.

This is a meaningful and realistic finding. It demonstrates that retrieval augmentation is not automatically beneficial in multimodal captioning. Performance depends critically on retrieval precision, evidence quality, and controlled fusion.

From an engineering standpoint, the final system is stronger than the initial version: it is reproducible, better instrumented, and supported by explicit failure evidence. From a research standpoint, it provides a clear next-step roadmap centered on better reranking, region-aware retrieval, and stricter evidence-grounded generation.

Overall, this project achieved the intended course objective: building, evaluating, and critically iterating on a computer vision system while documenting both successful and unsuccessful approaches with evidence-backed analysis.

---

## 7.1 Extended Lessons Learned from Implementation

To fully reflect the implementation process, I summarize practical lessons that were not obvious at design time but became clear during iterative execution.

### Lesson 1: “Retrieval added” is not equivalent to “grounding improved”

At the beginning, I assumed that adding external evidence would mechanically improve factuality. In practice, the opposite happened in naive settings. Retrieval introduced semantically plausible but visually incorrect context. This produced captions that were coherent in language but less faithful to the input image. The strongest practical takeaway is that retrieval quality and evidence integration quality must both be validated; otherwise, RAG can increase hallucination rather than reduce it.

### Lesson 2: Fallback is not optional in production-style systems

The experiments repeatedly showed that unsafe switching from baseline to RAG candidates causes large regressions. The safety gate with fallback was the most reliable component for damage control. Even when it did not improve over baseline, it prevented catastrophic degradation. This mirrors industry systems where reliability constraints are often more important than occasional gains.

### Lesson 3: Evaluation design drives interpretation

Using object-level hallucination metrics with COCO instance labels was useful because it made failures explicit and measurable. However, it also highlighted that metric selection shapes conclusions. For example, very short captions can reduce hallucination counts but may reduce informativeness. Therefore, future evaluations should combine factuality with informativeness/coverage metrics to avoid over-optimizing a single dimension.

### Lesson 4: Engineering details strongly affect research velocity

Several non-model changes had outsized impact:

- embedding cache in `.pt` files,
- stable output directory handling,
- robust path and input checks,
- and automatic case annotation files.

Without these improvements, repeated experiments would have been too slow or too brittle to complete in the available time. In other words, reproducibility tooling is part of the scientific method, not just a convenience.

### Lesson 5: Negative results can still be high-value outcomes

A key outcome of this project is a well-supported negative result: under this setup, naive and lightly constrained RAG do not outperform baseline for object-level hallucination in captioning. This is still a strong final assessment result because:

- it is empirically validated across multiple iterations,
- the failure modes are diagnosed and evidenced,
- and concrete mitigation and future directions are proposed.

In applied ML, understanding when a method fails is often as valuable as showing when it succeeds.

---

## 7.2 Project Planning and Execution Timeline

This project followed a staged execution plan with explicit checkpoints:

### Phase 1: Metric and baseline validation

- Implemented baseline caption generation and object-matching metric.
- Verified CSV outputs and summarization pipeline.
- Confirmed that evaluation scripts run end-to-end with small samples.

### Phase 2: First true-RAG implementation

- Added CLIP retrieval and BLIP evidence-conditioned generation.
- Ran controlled small-sample experiments.
- Identified severe hallucination regression and documented bad cases.

### Phase 3: Safety-focused iteration

- Implemented guarded prompting and safety-gate selection.
- Added margin-based fallback policy.
- Generated annotated evidence files (`bad_case` exports).

### Phase 4: Data split realism upgrade

- Switched to train retrieval / val evaluation setup.
- Added retrieval embedding caching to reduce rerun cost.
- Reran larger sample and compared against previous versions.

### Phase 5: Literature-informed v6 enhancement

- Added coarse retrieval + reranking.
- Added uncertainty-triggered RAG activation.
- Added whitelist-constrained generation and verifier fallback.
- Ran final comparison and produced final evidence files.

This timeline demonstrates clear milestone-based execution, iterative decision-making, and completion of all core deliverables: implementation, evaluation, analysis, and presentation preparation.

---

## 8. Source Code and Reproducibility Instructions

### 8.1 Project Files

- `run_caption_experiment.py`  
- `run_caption_experiment_with_rag.py`  
- `summarize_results.py`  
- `annotate_bad_cases.py`  
- `requirements.txt`  
- `README.md`

### 8.2 Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 8.3 Core Runs

Non-RAG baseline:

```bash
python run_caption_experiment.py \
  --image-dir data/coco/val2017 \
  --captions-json data/coco/annotations/captions_val2017.json \
  --instances-json data/coco/annotations/instances_val2017.json \
  --sample-size 30 \
  --output-dir outputs
```

RAG v5 split retrieval:

```bash
python run_caption_experiment_with_rag.py \
  --image-dir data/coco/val2017 \
  --captions-json data/coco/annotations/captions_val2017.json \
  --instances-json data/coco/annotations/instances_val2017.json \
  --retrieval-image-dir data/coco/train2017 \
  --retrieval-captions-json data/coco/annotations/captions_train2017.json \
  --retrieval-sample-size 20000 \
  --retrieval-cache-pt outputs_rag_v5/retrieval_cache.pt \
  --sample-size 50 \
  --top-k 5 \
  --min-consensus 2 \
  --safe-margin 0.02 \
  --output-dir outputs_rag_v5
```

RAG v6:

```bash
python run_caption_experiment_with_rag.py \
  --image-dir data/coco/val2017 \
  --captions-json data/coco/annotations/captions_val2017.json \
  --instances-json data/coco/annotations/instances_val2017.json \
  --retrieval-image-dir data/coco/train2017 \
  --retrieval-captions-json data/coco/annotations/captions_train2017.json \
  --retrieval-sample-size 20000 \
  --retrieval-cache-pt outputs_rag_v5/retrieval_cache.pt \
  --sample-size 50 \
  --retrieval-top-m 50 \
  --rerank-k 5 \
  --top-k 5 \
  --min-consensus 2 \
  --safe-margin 0.02 \
  --uncertainty-threshold 0.30 \
  --v6-object-cap 6 \
  --output-dir outputs_rag_v6
```

Summaries + annotations:

```bash
python summarize_results.py --results-csv outputs_rag_v6/results_rag.csv --output-csv outputs_rag_v6/summary_rag.csv
python annotate_bad_cases.py --results-csv outputs_rag_v6/results_rag.csv --output-csv outputs_rag_v6/results_rag_annotated.csv
```

### 8.4 Key Output Files for Submission Evidence

- `outputs/summary.csv`
- `outputs_rag/summary_rag.csv`
- `outputs_rag_v4/summary_rag.csv`
- `outputs_rag_v5/summary_rag.csv`
- `outputs_rag_v6/summary_rag.csv`
- `outputs_rag_v6/results_rag_annotated_bad_only.csv`


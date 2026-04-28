# 5-Minute Presentation Script

Hi everyone, my name is Jimmy, and this is my final project on reducing hallucination in vision-language image captioning.

My project question is simple:  
**Can RAG reduce hallucination in image captioning compared with a non-retrieval baseline?**

---

For context, hallucination means the model mentions objects that are not actually in the image.  
This matters because VLM outputs can look fluent and convincing even when they are wrong.

I used the MS COCO 2017 dataset and pretrained models, so this is an inference-time study, not a training-heavy one.

The main models are:

- BLIP base for caption generation
- CLIP ViT-B/32 for image embedding retrieval

I implemented everything in Python with PyTorch and Hugging Face Transformers.

---

I structured the project as iterative system development:

### Step 1: Baseline and metric validation

I first built a non-RAG captioning baseline and a prompt-constrained baseline to verify the evaluation pipeline.

On a 30-image sample:

- baseline hallucination rate: 0.0333
- constrained prompt hallucination rate: 0.0000

This confirmed the metric code and data pipeline were working.

---

### Step 2: True RAG baseline

Then I implemented true image-RAG:

- retrieve similar images with CLIP
- take their reference captions as external evidence
- feed evidence to BLIP for generation

This naive RAG performed very poorly.  
Hallucination jumped to 0.9 on 30 samples.

That was an important turning point: it showed retrieval can hurt if evidence is noisy.

---

### Step 3: Guardrails and safety gating

I added:

- consensus-based guarded RAG
- CLIP-based safety gate with fallback to baseline

This stabilized the system and removed catastrophic failures, but still did not consistently beat baseline.

---

### Step 4: Split retrieval and evaluation corpus

To make the setup more realistic, I used:

- train2017 as retrieval corpus
- val2017 as evaluation set

This is the v5 setup with 50 evaluation images and 20,000 retrieval items.

v5 summary:

- baseline: hallucination 0.08, precision 0.8857
- naive RAG: 0.56, precision 0.5196
- guarded RAG: 0.58, precision 0.5146
- safe-gated RAG: 0.14, precision 0.84

So gating helps a lot, but baseline still wins.

---

### Step 5: v6 minimal literature-inspired pipeline

Based on current papers, I implemented a v6 minimal pipeline:

1. top-50 retrieval then rerank to top-5
2. uncertainty-triggered RAG
3. object whitelist constraints
4. verifier plus fallback

v6 results:

- baseline: hallucination 0.08, precision 0.8857
- rag_v6_pipeline: hallucination 0.14, precision 0.8043

v6 improved over earlier RAG variants, but still did not exceed baseline.

---

### Key conclusion

My final conclusion is:

1. **Naive RAG is not automatically helpful in VLM captioning.**
2. **Retrieval noise can increase hallucination significantly.**
3. **Guardrails, gating, and fallback are essential for stable behavior.**
4. **In this implementation, RAG became safer but not better than baseline.**

This is still a strong result because it is evidence-driven and reproducible, and it shows exactly what failed and why.

---

### Challenges I faced

Main challenges were:

- retrieval-quality mismatch,
- evidence fusion errors,
- and runtime cost of embedding retrieval corpus.

I solved practical issues by:

- adding `.pt` embedding cache,
- adding case tagging (`good_case`, `bad_case`, `neutral_case`),
- and exporting bad-case-only evidence files for analysis.

---

### Future work

If I continued this project, I would focus on:

- stronger reranking models,
- region-level retrieval instead of global image similarity,
- edit-based caption correction rather than free-form rewrite,
- and better calibration for abstain/fallback decisions.

Thank you. I’m happy to answer questions.


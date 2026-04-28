# Final Report: Evaluating and Reducing Hallucination in Vision-Language Models

## 1. Introduction
Hallucination in vision-language models (VLMs) occurs when generated captions describe objects or attributes that are not present in the image. This problem impacts trust and reliability in real-world AI systems. In this project, we evaluate hallucination in image captioning on MS COCO and test whether constrained prompting can reduce hallucinated object mentions.

**Research question**: Can simple prompt constraints reduce hallucination rate in pretrained VLM caption generation?

## 2. Related Work
Early image captioning systems based on CNN-RNN pipelines often produced object-level hallucinations due to weak grounding. Transformer-based models (e.g., BLIP family) improved generation quality but still hallucinate in complex scenes. Prior mitigation methods include constrained decoding, grounding modules, and prompt engineering. This project focuses on a lightweight, reproducible mitigation strategy using prompt constraints.

## 3. Data and Experimental Setup

### 3.1 Dataset
- MS COCO val2017 images
- `captions_val2017.json` and `instances_val2017.json` annotations
- Sample size: **30** images

### 3.2 Model
- Base model: **Salesforce/blip-image-captioning-base**
- Framework: PyTorch + HuggingFace Transformers

### 3.3 Methods
1. **Baseline prompt**: `""` (no text prompt; image-only captioning)
2. **Grounded prompt**: `"a photo of"` (short grounding prefix)

### 3.4 Hallucination Detection Rule
For each generated caption:
1. Extract object category mentions from caption text
2. Compare them against COCO ground-truth object categories for that image
3. Count mentions not in ground truth as hallucinated mentions

This yields an operational hallucination metric suitable for fast comparison.

## 4. Metrics
- **Hallucination Rate**: percentage of examples with at least one hallucinated object mention
- **Average Hallucinated Mentions**: average hallucinated object count per example
- **Object Precision**: proportion of mentioned objects that appear in image ground truth

## 5. Results

### 5.1 Quantitative Comparison
| Method | #Examples | Hallucination Rate | Avg Hallucinated Mentions | Object Precision |
|---|---:|---:|---:|---:|
| Baseline | 30 | 0.0333 | 0.0333 | 0.9583 |
| Grounded Prompt | 30 | 0.0000 | 0.0000 | 1.0000 |

### 5.2 Analysis
- The grounded prompt reduced hallucination rate from **3.33%** to **0.00%** on this sample.
- Average hallucinated mentions also dropped from **0.0333** to **0.0000** per image.
- Object precision improved from **0.9583** to **1.0000**, indicating better grounding of mentioned objects.

## 6. Case Studies

### Case 1: Grounded prompt improves factuality
- Image ID: `56344`
- Baseline caption: "a desk with a computer, a phone, and a laptop"
- Grounded caption: "a desk with a computer and a phone"
- Explanation: COCO ground truth for this image does not include `laptop`; the grounded prompt removes this unsupported object mention.

### Case 2: Both methods fail
- Image ID: N/A in this 30-image run
- Baseline caption: N/A
- Grounded caption: N/A
- Explanation: No clear case where both methods produced object-level hallucination under the current lexical metric and sample size.

### Case 3: Grounded prompt is over-conservative
- Image ID: `80274`
- Baseline caption: "two elephants in a zoo"
- Grounded caption: "two elephants"
- Explanation: The grounded prompt generates a shorter and safer caption, which may reduce detail richness while improving factual reliability.

## 7. Limitations
- Keyword/object matching does not capture all semantic paraphrases
- Category mention extraction is lexical and may miss synonyms
- Only one base model is tested in this report
- No fine-tuning or grounding module is included

## 8. Conclusion
This project provides a reproducible benchmark for measuring object-level hallucination in caption generation and tests a low-cost mitigation strategy. Results show that constrained prompting **reduces** hallucination under the current setup, with hallucination rate decreasing from **3.33%** (baseline) to **0.00%** (grounded prompt) on 30 sampled images. Future work will extend to larger sample sizes, stronger grounding methods, richer semantic evaluation, and multi-model comparison.

## 9. Reproducibility
- Code: `run_caption_experiment.py`, `summarize_results.py`
- Prompt config: `prompt_templates.json`
- Environment: `requirements.txt`
- Outputs:
  - `outputs/results.csv`
  - `outputs/summary.csv`
  - `outputs/run_config.json`

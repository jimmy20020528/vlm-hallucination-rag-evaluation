# Hallucination Reduction in Vision-Language Models

This project evaluates object-level hallucination in image captioning and implements iterative mitigation strategies, including baseline prompting, true image-RAG, safety-gated RAG, and a literature-inspired v6 pipeline.

## Goal
Measure how often a pretrained captioning model hallucinates objects not present in the image, then compare:

1. Baseline caption generation
2. Constrained prompting for grounded captions
3. Naive image-RAG (CLIP retrieval + BLIP generation)
4. Guarded/safety-gated RAG
5. v6 minimal pipeline (top-M retrieval + rerank + uncertainty trigger + whitelist + fallback)

## Dataset
- MS COCO val2017 images
- COCO annotations (`captions_val2017.json`, `instances_val2017.json`)

## Quick Start
1. Create environment and install packages:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run experiments:
   - `python run_caption_experiment.py --image-dir /path/to/val2017 --captions-json /path/to/captions_val2017.json --instances-json /path/to/instances_val2017.json --sample-size 200 --output-dir outputs`
3. Generate summary metrics:
   - `python summarize_results.py --results-csv outputs/results.csv`

## True RAG Version (Image Retrieval + Caption Generation)
This repo also includes a retrieval-augmented pipeline:
1. Use CLIP image embeddings to retrieve top-k similar images
2. Use retrieved captions as external evidence
3. Generate caption with BLIP conditioned on retrieved evidence

Run:
- `python run_caption_experiment_with_rag.py --image-dir /path/to/val2017 --captions-json /path/to/captions_val2017.json --instances-json /path/to/instances_val2017.json --sample-size 30 --top-k 3 --output-dir outputs_rag`
- `python summarize_results.py --results-csv outputs_rag/results_rag.csv --output-csv outputs_rag/summary_rag.csv`

Recommended stronger setup (retrieval split):
- Use `train2017 + captions_train2017.json` as retrieval corpus
- Use `val2017 + captions_val2017.json` for evaluation
- Example:
  - `python run_caption_experiment_with_rag.py --image-dir /path/to/val2017 --captions-json /path/to/captions_val2017.json --instances-json /path/to/instances_val2017.json --retrieval-image-dir /path/to/train2017 --retrieval-captions-json /path/to/captions_train2017.json --sample-size 50 --top-k 5 --safe-margin 0.02 --output-dir outputs_rag_split`

## Evidence Annotation (good/bad case tags)
To tag each sample with `good_case`, `bad_case`, or `neutral_case`:
- `python annotate_bad_cases.py --results-csv outputs/results.csv --output-csv outputs/results_annotated.csv`
- `python annotate_bad_cases.py --results-csv outputs_rag/results_rag.csv --output-csv outputs_rag/results_rag_annotated.csv`

Generated files:
- `*_annotated.csv`: all samples with `case_tag` and `case_id`
- `*_annotated_bad_only.csv`: only bad cases for report evidence

## Project Files
- `run_caption_experiment.py`: caption generation + object-level hallucination check
- `run_caption_experiment_with_rag.py`: CLIP retrieval + BLIP generation (true RAG)
- `summarize_results.py`: aggregate metrics and compare methods
- `annotate_bad_cases.py`: add case tags and export bad-case evidence
- `prompt_templates.json`: prompts tested in the project
- `final_report_draft.md`: report template aligned to your midterm direction

## Hallucination Definition (Operational)
For each image, if a generated caption mentions object categories that are not in COCO ground-truth labels for that image, those mentions are counted as hallucinated mentions.

## Metrics
- `hallucination_rate`: fraction of examples with at least one hallucinated mention
- `avg_hallucinated_mentions`: average number of hallucinated object mentions per example
- `object_precision`: mentioned_objects_in_gt / total_mentioned_objects

## Notes
- This is a lightweight reproducible baseline using keyword/object matching against COCO categories.
- It is intentionally simple and can be extended with stronger factuality checks later.

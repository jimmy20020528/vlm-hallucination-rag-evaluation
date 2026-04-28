import argparse
import json
import os
import random
import re
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_to_gt_categories(instances_json):
    categories = {c["id"]: c["name"].lower() for c in instances_json["categories"]}
    image_to_cats = defaultdict(set)
    for ann in instances_json["annotations"]:
        image_to_cats[ann["image_id"]].add(categories[ann["category_id"]])
    return image_to_cats, set(categories.values())


def build_image_id_to_filename(captions_json):
    return {img["id"]: img["file_name"] for img in captions_json["images"]}


def build_image_id_to_ref_captions(captions_json):
    image_to_caps = defaultdict(list)
    for ann in captions_json["annotations"]:
        image_to_caps[ann["image_id"]].append(ann["caption"].strip())
    return image_to_caps


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def extract_mentioned_categories(caption, category_names):
    text = normalize_text(caption)
    words = set(text.split())
    mentioned = set()

    for cat in category_names:
        if " " not in cat and cat in words:
            mentioned.add(cat)
        elif " " in cat and cat in text:
            mentioned.add(cat)

    return mentioned


def clean_caption_text(caption):
    text = normalize_text(caption)
    return text


def category_consensus_from_captions(captions, category_names):
    counts = defaultdict(int)
    for cap in captions:
        mentioned = extract_mentioned_categories(cap, category_names)
        for cat in mentioned:
            counts[cat] += 1
    return counts


def generate_caption(processor, model, image, prompt, device):
    prompt = (prompt or "").strip()
    if prompt:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    # Remove prompt echo if present.
    lower_caption = caption.lower()
    lower_prompt = prompt.lower()
    if lower_prompt and lower_caption.startswith(lower_prompt):
        caption = caption[len(prompt) :].strip(" .,:;!-")

    return caption


def encode_image_clip(clip_processor, clip_model, image, device):
    clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**clip_inputs)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats[0]


def encode_text_clip(clip_processor, clip_model, text, device):
    clip_inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**clip_inputs)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats[0]


def clip_score(clip_processor, clip_model, image_emb, caption, device):
    text_emb = encode_text_clip(clip_processor, clip_model, caption, device)
    return float(torch.dot(image_emb, text_emb).item())


def build_rag_prompt(retrieved_caps):
    bullet_text = "\n".join([f"- {c}" for c in retrieved_caps if c])
    prompt = (
        "Use the retrieved captions as external evidence. "
        "Describe only clearly visible objects and actions.\n"
        f"Retrieved evidence:\n{bullet_text}\n"
        "Caption:"
    )
    return prompt


def build_edit_rag_prompt(draft_caption, retrieved_caps):
    bullet_text = "\n".join([f"- {c}" for c in retrieved_caps if c])
    prompt = (
        "You are given a draft caption and retrieved evidence from similar images.\n"
        "Revise the draft to be concise and factual for the current image.\n"
        "Important: do not introduce new object categories that are not already in the draft caption.\n"
        "If evidence conflicts, trust the current image and keep conservative wording.\n"
        f"Draft caption: {draft_caption}\n"
        f"Retrieved evidence:\n{bullet_text}\n"
        "Final caption:"
    )
    return prompt


def build_guarded_rag_prompt(retrieved_caps, allowed_categories):
    evidence = "\n".join([f"- {c}" for c in retrieved_caps if c])
    allowed = ", ".join(sorted(allowed_categories)) if allowed_categories else "none"
    prompt = (
        "You are captioning a real image. Use visual evidence and retrieved hints conservatively.\n"
        f"Retrieved evidence:\n{evidence}\n"
        f"Likely object candidates: {allowed}\n"
        "Write one short factual caption. If uncertain, keep it generic."
    )
    return prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--retrieval-image-dir", default=None, help="Optional retrieval image directory (e.g., train2017)")
    parser.add_argument("--retrieval-captions-json", default=None, help="Optional retrieval captions json (e.g., captions_train2017.json)")
    parser.add_argument(
        "--retrieval-sample-size",
        type=int,
        default=0,
        help="If > 0, randomly sample this many retrieval items from the retrieval corpus",
    )
    parser.add_argument(
        "--retrieval-cache-pt",
        default="",
        help="Optional .pt cache path for retrieval embeddings and ids",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3, help="Retrieved neighbors for RAG")
    parser.add_argument("--retrieval-top-m", type=int, default=50, help="Coarse retrieval pool size before reranking")
    parser.add_argument("--rerank-k", type=int, default=5, help="Final retrieved evidence count after reranking")
    parser.add_argument("--min-consensus", type=int, default=2, help="Minimum mentions across retrieved captions to trust an object")
    parser.add_argument(
        "--safe-margin",
        type=float,
        default=0.02,
        help="Required CLIP score improvement over baseline before switching to RAG candidate",
    )
    parser.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", default="outputs_rag")
    parser.add_argument("--experiment-name", default="clip_retrieval_rag_v1")
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.30,
        help="If baseline CLIP score is below this value, trigger v6 retrieval rewrite",
    )
    parser.add_argument(
        "--v6-object-cap",
        type=int,
        default=6,
        help="Maximum allowed object candidates in v6 whitelist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions = load_json(args.captions_json)
    retrieval_captions = load_json(args.retrieval_captions_json) if args.retrieval_captions_json else captions
    instances = load_json(args.instances_json)
    image_to_cats, all_category_names = build_image_to_gt_categories(instances)
    image_id_to_filename = build_image_id_to_filename(captions)
    retrieval_image_id_to_filename = build_image_id_to_filename(retrieval_captions)
    image_id_to_ref_caps = build_image_id_to_ref_captions(retrieval_captions)
    all_image_ids = list(image_id_to_filename.keys())
    retrieval_image_ids = list(retrieval_image_id_to_filename.keys())
    retrieval_image_dir = args.retrieval_image_dir or args.image_dir
    if not os.path.isdir(retrieval_image_dir):
        raise FileNotFoundError(f"retrieval_image_dir not found: {retrieval_image_dir}")
    if args.retrieval_sample_size and args.retrieval_sample_size > 0:
        if args.retrieval_sample_size < len(retrieval_image_ids):
            retrieval_image_ids = random.sample(retrieval_image_ids, args.retrieval_sample_size)

    if args.sample_size > len(all_image_ids):
        raise ValueError(f"sample_size={args.sample_size} > available images={len(all_image_ids)}")
    sampled_ids = random.sample(all_image_ids, args.sample_size)

    blip_processor = BlipProcessor.from_pretrained(args.blip_model)
    blip_model = BlipForConditionalGeneration.from_pretrained(args.blip_model).to(device)
    blip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()

    # Precompute or load CLIP embeddings for retrieval pool.
    retrieval_emb_matrix = None
    valid_retrieval_ids = []
    cache_loaded = False
    if args.retrieval_cache_pt and os.path.exists(args.retrieval_cache_pt):
        cache_obj = torch.load(args.retrieval_cache_pt, map_location="cpu")
        cache_ids = cache_obj.get("valid_retrieval_ids", [])
        cache_emb = cache_obj.get("retrieval_emb_matrix", None)
        cache_meta = cache_obj.get("meta", {})
        expected_meta = {
            "clip_model": args.clip_model,
            "retrieval_image_dir": retrieval_image_dir,
            "retrieval_captions_json": args.retrieval_captions_json if args.retrieval_captions_json else args.captions_json,
            "retrieval_sample_size": args.retrieval_sample_size,
        }
        if cache_emb is not None and cache_ids and cache_meta == expected_meta:
            retrieval_emb_matrix = cache_emb.to(device)
            valid_retrieval_ids = cache_ids
            cache_loaded = True
            print(f"Loaded retrieval embedding cache from: {args.retrieval_cache_pt}")
        else:
            print("Embedding cache exists but meta mismatch; rebuilding cache.")

    if not cache_loaded:
        retrieval_emb_list = []
        missing_retrieval_files = 0
        for image_id in tqdm(retrieval_image_ids, desc="Encoding retrieval CLIP features"):
            img_path = os.path.join(retrieval_image_dir, retrieval_image_id_to_filename[image_id])
            if not os.path.exists(img_path):
                missing_retrieval_files += 1
                continue
            image = Image.open(img_path).convert("RGB")
            emb = encode_image_clip(clip_processor, clip_model, image, device)
            retrieval_emb_list.append(emb)
            valid_retrieval_ids.append(image_id)

        if len(retrieval_emb_list) == 0:
            raise RuntimeError(
                "No retrieval images were loaded. "
                f"Check --retrieval-image-dir ({retrieval_image_dir}) and caption json alignment. "
                f"Missing files counted: {missing_retrieval_files} / {len(retrieval_image_ids)}"
            )

        retrieval_emb_matrix = torch.stack(retrieval_emb_list, dim=0)  # [N, D]

        if args.retrieval_cache_pt:
            cache_parent = os.path.dirname(args.retrieval_cache_pt)
            if cache_parent:
                os.makedirs(cache_parent, exist_ok=True)
            cache_obj = {
                "valid_retrieval_ids": valid_retrieval_ids,
                "retrieval_emb_matrix": retrieval_emb_matrix.detach().cpu(),
                "meta": {
                    "clip_model": args.clip_model,
                    "retrieval_image_dir": retrieval_image_dir,
                    "retrieval_captions_json": args.retrieval_captions_json if args.retrieval_captions_json else args.captions_json,
                    "retrieval_sample_size": args.retrieval_sample_size,
                },
            }
            torch.save(cache_obj, args.retrieval_cache_pt)
            print(f"Saved retrieval embedding cache to: {args.retrieval_cache_pt}")

    rows = []
    for image_id in tqdm(sampled_ids, desc="Running baseline and RAG"):
        file_name = image_id_to_filename[image_id]
        img_path = os.path.join(args.image_dir, file_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        gt_categories = image_to_cats[image_id]
        query_emb = encode_image_clip(clip_processor, clip_model, image, device)

        # Baseline (no text prompt)
        baseline_caption = generate_caption(blip_processor, blip_model, image, "", device)
        baseline_mentioned = extract_mentioned_categories(baseline_caption, all_category_names)
        baseline_hall = baseline_mentioned - gt_categories
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "image_id": image_id,
                "file_name": file_name,
                "method": "baseline",
                "caption": baseline_caption,
                "retrieved_image_ids": "",
                "retrieved_captions": "",
                "gt_categories": "|".join(sorted(gt_categories)),
                "mentioned_categories": "|".join(sorted(baseline_mentioned)),
                "hallucinated_categories": "|".join(sorted(baseline_hall)),
                "mentioned_count": len(baseline_mentioned),
                "hallucinated_count": len(baseline_hall),
                "has_hallucination": int(len(baseline_hall) > 0),
            }
        )

        # RAG retrieval by CLIP image similarity.
        query = query_emb.unsqueeze(0)  # [1, D]
        sims = torch.matmul(query, retrieval_emb_matrix.t())[0]  # [N]
        coarse_k = min(args.retrieval_top_m, len(valid_retrieval_ids))
        coarse_vals, coarse_pos = torch.topk(sims, k=coarse_k)
        coarse_scores = coarse_vals.tolist()
        coarse_indices = coarse_pos.tolist()

        # Small rerank stage: CLIP caption-image score from one ref caption.
        rerank_items = []
        for idx_in_pool, retrieval_pos in enumerate(coarse_indices):
            rid = valid_retrieval_ids[retrieval_pos]
            cap_list = image_id_to_ref_caps[rid]
            if not cap_list:
                continue
            ref_cap = cap_list[0]
            txt_score = clip_score(clip_processor, clip_model, query_emb, ref_cap, device)
            combined = 0.5 * float(coarse_scores[idx_in_pool]) + 0.5 * float(txt_score)
            rerank_items.append((rid, ref_cap, combined))
        rerank_items.sort(key=lambda x: x[2], reverse=True)
        rerank_items = rerank_items[: min(args.rerank_k, len(rerank_items))]

        retrieved_ids = [item[0] for item in rerank_items]
        retrieved_caps = []
        for rid, ref_cap, _ in rerank_items:
            _ = rid
            retrieved_caps.append(ref_cap)

        rag_prompt = build_rag_prompt(retrieved_caps)
        rag_caption = generate_caption(blip_processor, blip_model, image, rag_prompt, device)
        rag_mentioned = extract_mentioned_categories(rag_caption, all_category_names)
        rag_hall = rag_mentioned - gt_categories
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "image_id": image_id,
                "file_name": file_name,
                "method": "rag",
                "caption": rag_caption,
                "retrieved_image_ids": "|".join([str(x) for x in retrieved_ids]),
                "retrieved_captions": " || ".join(retrieved_caps),
                "gt_categories": "|".join(sorted(gt_categories)),
                "mentioned_categories": "|".join(sorted(rag_mentioned)),
                "hallucinated_categories": "|".join(sorted(rag_hall)),
                "mentioned_count": len(rag_mentioned),
                "hallucinated_count": len(rag_hall),
                "has_hallucination": int(len(rag_hall) > 0),
            }
        )

        # Guarded RAG: use retrieval consensus + baseline overlap to reduce noisy evidence injection.
        consensus_counts = category_consensus_from_captions(retrieved_caps, all_category_names)
        baseline_words = extract_mentioned_categories(baseline_caption, all_category_names)
        consensus_set = {k for k, v in consensus_counts.items() if v >= args.min_consensus}
        allowed_candidates = consensus_set.intersection(baseline_words)

        # Fallback: if intersection too small, keep strongest consensus categories.
        if len(allowed_candidates) == 0 and len(consensus_counts) > 0:
            sorted_consensus = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)
            allowed_candidates = {k for k, _ in sorted_consensus[:2]}

        guarded_prompt = build_guarded_rag_prompt(retrieved_caps, allowed_candidates)
        guarded_caption = generate_caption(blip_processor, blip_model, image, guarded_prompt, device)
        guarded_mentioned = extract_mentioned_categories(guarded_caption, all_category_names)
        guarded_hall = guarded_mentioned - gt_categories
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "image_id": image_id,
                "file_name": file_name,
                "method": "rag_guarded",
                "caption": guarded_caption,
                "retrieved_image_ids": "|".join([str(x) for x in retrieved_ids]),
                "retrieved_captions": " || ".join(retrieved_caps),
                "gt_categories": "|".join(sorted(gt_categories)),
                "mentioned_categories": "|".join(sorted(guarded_mentioned)),
                "hallucinated_categories": "|".join(sorted(guarded_hall)),
                "mentioned_count": len(guarded_mentioned),
                "hallucinated_count": len(guarded_hall),
                "has_hallucination": int(len(guarded_hall) > 0),
            }
        )

        # Safety gate: choose best caption by image-text consistency score.
        image_emb = query_emb
        candidates = [
            ("baseline", baseline_caption),
            ("rag", rag_caption),
            ("rag_guarded", guarded_caption),
        ]
        scored = [(m, c, clip_score(clip_processor, clip_model, image_emb, c, device)) for m, c in candidates]
        score_map = {m: (c, s) for m, c, s in scored}
        best_source, best_caption, best_score = sorted(scored, key=lambda x: x[2], reverse=True)[0]
        baseline_score = score_map["baseline"][1]
        if best_source != "baseline" and (best_score - baseline_score) < args.safe_margin:
            best_source = "baseline"
            best_caption = score_map["baseline"][0]
            best_score = baseline_score

        safe_mentioned = extract_mentioned_categories(best_caption, all_category_names)
        safe_hall = safe_mentioned - gt_categories
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "image_id": image_id,
                "file_name": file_name,
                "method": "rag_safe_gate",
                "caption": best_caption,
                "retrieved_image_ids": "|".join([str(x) for x in retrieved_ids]),
                "retrieved_captions": " || ".join(retrieved_caps),
                "gt_categories": "|".join(sorted(gt_categories)),
                "mentioned_categories": "|".join(sorted(safe_mentioned)),
                "hallucinated_categories": "|".join(sorted(safe_hall)),
                "mentioned_count": len(safe_mentioned),
                "hallucinated_count": len(safe_hall),
                "has_hallucination": int(len(safe_hall) > 0),
                "selected_from": best_source,
                "selected_score": round(best_score, 6),
            }
        )

        # v6 pipeline: uncertainty-triggered rewrite + object whitelist + verifier fallback.
        baseline_clip = clip_score(clip_processor, clip_model, image_emb, baseline_caption, device)
        trigger_v6 = baseline_clip < args.uncertainty_threshold

        v6_selected_from = "baseline_no_trigger"
        v6_caption = baseline_caption
        v6_score = baseline_clip
        v6_prompt = ""
        allowed_for_v6 = []
        if trigger_v6 and retrieved_caps:
            consensus_counts_v6 = category_consensus_from_captions(retrieved_caps, all_category_names)
            ranked_consensus = sorted(consensus_counts_v6.items(), key=lambda x: x[1], reverse=True)
            allowed_consensus = [k for k, v in ranked_consensus if v >= args.min_consensus]
            if not allowed_consensus:
                allowed_consensus = [k for k, _ in ranked_consensus[: args.v6_object_cap]]
            # Keep whitelist compact.
            allowed_for_v6 = allowed_consensus[: args.v6_object_cap]
            v6_prompt = build_guarded_rag_prompt(retrieved_caps, set(allowed_for_v6))
            candidate_v6 = generate_caption(blip_processor, blip_model, image, v6_prompt, device)
            candidate_v6_score = clip_score(clip_processor, clip_model, image_emb, candidate_v6, device)
            candidate_v6_objs = extract_mentioned_categories(candidate_v6, all_category_names)
            # Penalty for introducing objects outside whitelist.
            extra_objs = candidate_v6_objs - set(allowed_for_v6)
            candidate_v6_adj_score = candidate_v6_score - 0.05 * len(extra_objs)

            if candidate_v6_adj_score > (baseline_clip + args.safe_margin):
                v6_caption = candidate_v6
                v6_score = candidate_v6_adj_score
                v6_selected_from = "v6_rag_candidate"

        v6_mentioned = extract_mentioned_categories(v6_caption, all_category_names)
        v6_hall = v6_mentioned - gt_categories
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "image_id": image_id,
                "file_name": file_name,
                "method": "rag_v6_pipeline",
                "caption": v6_caption,
                "retrieved_image_ids": "|".join([str(x) for x in retrieved_ids]),
                "retrieved_captions": " || ".join(retrieved_caps),
                "gt_categories": "|".join(sorted(gt_categories)),
                "mentioned_categories": "|".join(sorted(v6_mentioned)),
                "hallucinated_categories": "|".join(sorted(v6_hall)),
                "mentioned_count": len(v6_mentioned),
                "hallucinated_count": len(v6_hall),
                "has_hallucination": int(len(v6_hall) > 0),
                "selected_from": v6_selected_from,
                "selected_score": round(v6_score, 6),
                "v6_triggered": int(trigger_v6),
                "v6_allowed_objects": "|".join(sorted(allowed_for_v6)),
                "v6_prompt": v6_prompt,
            }
        )

    df = pd.DataFrame(rows)
    results_csv = os.path.join(args.output_dir, "results_rag.csv")
    df.to_csv(results_csv, index=False)

    config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "retrieval_image_dir": retrieval_image_dir,
        "retrieval_captions_json": args.retrieval_captions_json if args.retrieval_captions_json else args.captions_json,
        "num_retrieval_items": len(valid_retrieval_ids),
        "retrieval_sample_size": args.retrieval_sample_size,
        "retrieval_cache_pt": args.retrieval_cache_pt if args.retrieval_cache_pt else None,
        "top_k": args.top_k,
        "retrieval_top_m": args.retrieval_top_m,
        "rerank_k": args.rerank_k,
        "min_consensus": args.min_consensus,
        "safe_margin": args.safe_margin,
        "uncertainty_threshold": args.uncertainty_threshold,
        "v6_object_cap": args.v6_object_cap,
        "blip_model": args.blip_model,
        "clip_model": args.clip_model,
        "device": device,
    }
    with open(os.path.join(args.output_dir, "run_config_rag.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved detailed results to: {results_csv}")
    print("Next: python summarize_results.py --results-csv outputs_rag/results_rag.csv --output-csv outputs_rag/summary_rag.csv")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import random
import re
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


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


def normalize_token(token):
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9 ]+", "", token)
    return token


def extract_mentioned_categories(caption, category_names):
    text = normalize_token(caption)
    words = set(text.split())
    mentioned = set()

    # Single-word category match
    for cat in category_names:
        if " " not in cat and cat in words:
            mentioned.add(cat)

    # Multi-word category match
    for cat in category_names:
        if " " in cat and cat in text:
            mentioned.add(cat)

    return mentioned


def generate_caption(processor, model, image, prompt, device):
    normalized_prompt = (prompt or "").strip()
    if normalized_prompt:
        inputs = processor(images=image, text=normalized_prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    # BLIP may sometimes echo prompt text when conditioned with free-form instructions.
    # Remove echoed prefix to keep only generated caption content.
    lower_caption = caption.lower().strip()
    lower_prompt = normalized_prompt.lower()
    if lower_prompt and lower_caption.startswith(lower_prompt):
        caption = caption[len(normalized_prompt) :].strip(" .,:;!-")

    return caption.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Path to COCO val2017 image directory")
    parser.add_argument("--captions-json", required=True, help="Path to captions_val2017.json")
    parser.add_argument("--instances-json", required=True, help="Path to instances_val2017.json")
    parser.add_argument("--prompt-json", default="prompt_templates.json", help="Prompt template JSON path")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of sampled images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", default="prompt_baseline_vs_grounded")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_json(args.prompt_json)
    captions = load_json(args.captions_json)
    instances = load_json(args.instances_json)

    image_to_cats, all_category_names = build_image_to_gt_categories(instances)
    image_id_to_filename = build_image_id_to_filename(captions)
    image_ids = list(image_id_to_filename.keys())

    if args.sample_size > len(image_ids):
        raise ValueError(f"sample_size={args.sample_size} > available images={len(image_ids)}")
    sampled_image_ids = random.sample(image_ids, args.sample_size)

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name).to(device)
    model.eval()

    rows = []
    for image_id in tqdm(sampled_image_ids, desc="Running caption experiment"):
        file_name = image_id_to_filename[image_id]
        image_path = os.path.join(args.image_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        gt_categories = image_to_cats[image_id]

        for method_name, prompt in prompts.items():
            caption = generate_caption(processor, model, image, prompt, device)
            mentioned = extract_mentioned_categories(caption, all_category_names)
            hallucinated = mentioned - gt_categories

            rows.append(
                {
                    "experiment_name": args.experiment_name,
                    "image_id": image_id,
                    "file_name": file_name,
                    "method": method_name,
                    "prompt": prompt,
                    "caption": caption,
                    "gt_categories": "|".join(sorted(gt_categories)),
                    "mentioned_categories": "|".join(sorted(mentioned)),
                    "hallucinated_categories": "|".join(sorted(hallucinated)),
                    "mentioned_count": len(mentioned),
                    "hallucinated_count": len(hallucinated),
                    "has_hallucination": int(len(hallucinated) > 0),
                }
            )

    df = pd.DataFrame(rows)
    results_csv = os.path.join(args.output_dir, "results.csv")
    df.to_csv(results_csv, index=False)

    config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "model_name": args.model_name,
        "device": device,
        "prompts": prompts,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved detailed results to: {results_csv}")
    print("Next step: python summarize_results.py --results-csv outputs/results.csv")


if __name__ == "__main__":
    main()

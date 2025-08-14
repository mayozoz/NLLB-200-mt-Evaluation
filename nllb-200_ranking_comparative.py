import json
import os
import sys
import re
import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 

mod = "1.3B"
model_path = f"facebook/nllb-200-{mod}"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="bod_Tibt", tgt_lang="zho_Hans")
print("Loading reversed tokenizer...")
reversed_tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="zho_Hans", tgt_lang="bod_Tibt")
print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
print("Loaded successfully!")
pipe = pipeline(
    "translation",
    model=model,
    tokenizer=reversed_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=8
)
print("Pipeline loaded successfully!")

def quality_estimation_score(source, translations):
    """Generate quality estimation score using cross-lingual representation"""
    scores = []
    src_inputs = tokenizer(source, return_tensors="pt", padding=True)

    with torch.no_grad():
        # get encoder representations 
        src_outputs = model.get_encoder()(**src_inputs)
        src_embeddings = src_outputs.last_hidden_state.mean(dim=1) # avg pooling
        
        for tgt in translations:
            tgt_inputs = tokenizer(tgt, return_tensors="pt", padding=True)
            tgt_outputs = model.get_encoder()(**tgt_inputs)
            tgt_embeddings = tgt_outputs.last_hidden_state.mean(dim=1)

            similarity = torch.cosine_similarity(src_embeddings, tgt_embeddings, dim=1)
            scores.append(similarity.item())
    return scores


def conditional_probability_score(source, translations):
    """Calculate P(target|source) using the model, then normalized."""
    scores = []
    # Prepare inputs
    inputs = tokenizer([source]*len(translations), return_tensors="pt", padding=True, truncation=True).to(model.device)
    target_ids = tokenizer(translations, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=target_ids)
        # Get per-token log probabilities
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Calculate average log probability
        target_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        # avg_log_prob = target_log_probs.mean().item()
        
        scores = (target_log_probs.sum(dim=1) / (target_ids != tokenizer.pad_token_id).sum(dim=1))
    
        max_score = scores.max()
        return (1 - (scores/max_score)).tolist()
    # return scores


def roundtrip_consistency_score(source, translations):
    """Measure consistency via back-translation."""
    
    scores = []

    src_inputs = reversed_tokenizer(source, return_tensors="pt", padding=True)

    # run quality_estimation_score basically 
    with torch.no_grad():
        # get encoder representations 
        src_outputs = model.get_encoder()(**src_inputs)
        src_embeddings = src_outputs.last_hidden_state.mean(dim=1) # avg pooling
        
        for tgt in translations:
            # print(f"back translating {tgt}")
            back_translation = pipe(tgt, src_lang="zho_Hans", tgt_lang="bod_Tibt")[0]['translation_text']
            # print("back_translation")
            tgt_inputs = reversed_tokenizer(back_translation, return_tensors="pt", padding=True)
            # print("tgt_inputs")
            tgt_outputs = model.get_encoder()(**tgt_inputs)
            # print("tgt_outputs")
            tgt_embeddings = tgt_outputs.last_hidden_state.mean(dim=1)
            # print("tgt_embeddings")

            similarity = torch.cosine_similarity(src_embeddings, tgt_embeddings, dim=1)
            scores.append(similarity.item())
    return scores


def score(source, translations):
    """Score translations using:
        1. backtrack scoring
        2. perplexity scoring 
    """
    scores = []
    for trans in translations:
        # 1. BACKTRACK
        bt_score = backtranslate_score(source, trans)

        # 2. PERPLEXITY
        inputs = tokenizer(trans).to(model.device)
        with torch.no_grad():
            loss = model(**inputs).loss
        perplexity = torch.exp(loss).item()
        perplexity_score = max(1, min(5, 5-(perplexity/1)))

        # weighted avg 
        combined_score = round(0.7 * bt_score + 0.3 * perplexity_score)
        scores.append(combined_score)


def normalize_global_minmax(all_scores):
    """Normalize using global min/max across all scores."""
    flat_scores = [score for sentence_scores in all_scores for score in sentence_scores]
    min_s, max_s = min(flat_scores), max(flat_scores)

    if max_s == min_s:
        return [[3] * len(sentence_scores) for sentence_scores in all_scores]
    normalized = []
    for sentence_scores in all_scores:
        sentence_normalized = [round(1 + 4 * (s - min_s) / (max_s - min_s)) for s in sentence_scores]
        normalized.append(sentence_normalized)
    return normalized


def normalize_zscore_to_scale(all_scores, target_mean=3, target_std=1):
    """Z-score normalization, then map to 1-5 scale with specific mean and std."""
    flat_scores = [score for sentence_scores in all_scores for score in sentence_scores]
    mean_s = np.mean(flat_scores)
    std_s = np.std(flat_scores)

    if std_s == 0:
        return [[3] * len(sentence_scores) for sentence_scores in all_scores]

    normalized = []
    for sentence_scores in all_scores:
        z_scores = [(s - mean_s) / std_s for s in sentence_scores]
        scaled = [target_mean + z * target_std for z in z_scores]
        clipped = [max(1, min(5, round(s))) for s in scaled]
        normalized.append(clipped)
    return normalized


def normalize_percentile_based(all_scores):
    """
    map scores based on percentiles:
    - bottom 20% --> 1
    - next 20% --> 2
    ... etc.
    - top 20% --> 5
    """
    flat_scores = [score for sentence_scores in all_scores for score in sentence_scores]
    percentiles = np.percentile(flat_scores, [20, 40, 60, 80])

    def score_to_rating(score):
        if score <= percentiles[0]:
            return 1
        elif score <= percentiles[1]:
            return 2
        elif score <= percentiles[2]:
            return 3
        elif score <= percentiles[3]:
            return 4
        else:
            return 5
    
    normalized = []
    for sentence_scores in all_scores:
        sentence_normalized = [score_to_rating(s) for s in sentence_scores]
        normalized.append(sentence_normalized)

    return normalized


def normalize_sigmoid_based(all_scores, midpoint=None):
    """
    Use sigmoid function to map scores to 1-5.
    Midpoint defualts to median of all scores.
    """
    flat_scores = [score for sentence_scores in all_scores for score in sentence_scores]
    if midpoint is None:
        midpoint = np.median(flat_scores)
    
    scale = 2.0 # adjust to control steepness

    normalized = []
    for sentence_scores in all_scores:
        sentence_normalized = []
        for s in sentence_scores:
            sigmoid_val = 1 / (1 + np.exp(-scale * (s - midpoint))) # map (0,1)
            rating = round(1 + 4*sigmoid_val) # map [1,5]
            sentence_normalized.append(rating)
        normalized.append(sentence_normalized)
    
    return normalized 



def main(src_file, *tgt_files):
    """
python3 NLLB-200/nllb-200_ranking_comparative.py tbt-cn-200/test-mt-hyps/src.txt tbt-cn-200/test-mt-hyps/modelA.txt tbt-cn-200/test-mt-hyps/modelB.txt tbt-cn-200/test-mt-hyps/modelC.txt
python3 NLLB-200/nllb-200_ranking_comparative.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b
    """
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    tgt_lines = []
    for f in tgt_files:
        with open(f, 'r', encoding='utf-8') as fin:
            tgt_lines.append([line.strip() for line in fin])
    
    results = []
    all_raw_scores = []


    for i in tqdm(range(len(src_lines))):
        source = src_lines[i]
        translations = [tgt_lines[j][i] for j in range(len(tgt_files))]
        # scores = quality_estimation_score(source, translations)
        # scores = conditional_probability_score(source, translations)
        scores = roundtrip_consistency_score(source, translations)

        all_raw_scores.append(scores)
        results.append({
            "id": i,
            "source": source,
            "translations": translations,
            "scores": scores
        })

    # Method 1: Global min-max (recommended for most cases)
    normalized_scores = normalize_global_minmax(all_raw_scores)

    # Method 2: Z-score normalization (good for normally distributed scores)
    # normalized_scores = normalize_zscore_to_scale(all_raw_scores)

    # Method 3: Percentile-based (ensures even distribution across 1-5)
    # normalized_scores = normalize_percentile_based(all_raw_scores)

    # Method 4: Sigmoid-based (good for handling outliers)
    # normalized_scores = normalize_sigmoid_based(all_raw_scores)


    mode = "normalize_global_minmax"
    normalization_method = "normalize_sigmoid_based"

    # output_dir = f"tbt-cn-200/NLLB-200-{mod}_mev_scores/{mode}/{normalization_method}"
    output_dir = f"cn-tbt-200/NLLB-200-{mod}_mev_scores/{mode}/{normalization_method}"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(f"{output_dir}/nllb-200-{mod}.json", "w", encoding="utf-8") as f:
    # with open(f"{output_dir}/nllb-200_roundtrip_consistency_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/nllb-200-{mod}_deepseek-v3", "w", encoding="utf-8") as f1, \
        open(f"{output_dir}/nllb-200-{mod}_google-translate", "w", encoding="utf-8") as f2, \
        open(f"{output_dir}/nllb-200-{mod}_qwen2.5_72b", "w", encoding="utf-8") as f3:
        for group in normalized_scores:
            f1.write(f"{group[0]}\n")
            f2.write(f"{group[1]}\n")
            f3.write(f"{group[2]}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python m2m100_scoring.py <source_file> <translation1> [translation2...]")
        sys.exit(1)
    main(*sys.argv[1:])

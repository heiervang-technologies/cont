import json
import logging
import copy
import torch
import torch.nn.functional as F

from unsloth import FastLanguageModel
from lile.state import ModelState
from lile.engine.train import TrainEngine
from lile.objectives.unlike import unlike_loss, _prefix_ids, _pad_prefixes
from lile.objectives.safety import safety_monitor_loss

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    calibration_corpus = load_jsonl("lile/teach/rlaif/calibration_corpus.jsonl")
    neutral_corpus = load_jsonl("lile/teach/rlaif/neutral_corpus.jsonl")

    model_name = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
    max_seq_length = 512
    log.info(f"Loading {model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_id

    # The calibration corpus contains raw text completions.
    # If we wrap them in a chat template, the first assistant token will be a sentence starter
    # rather than the target bad_token, causing all ranks to be ~100k and probabilities ~0.
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None

    # For safety monitor, we need a watchlist
    # Let's get the token ids for bad_tokens and good_tokens
    def get_token_id(t_str):
        # We need the first token of the word with a leading space
        # since it follows the prefix which ends in a space or word.
        return tokenizer(" " + t_str.strip(), add_special_tokens=False).input_ids[0]

    # Convert corpus strings to ids
    for d in calibration_corpus + neutral_corpus:
        d["bad_token_id"] = get_token_id(d["bad_token"])
        d["good_token_id"] = get_token_id(d["good_token"])

    etas = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    rank_belows = [1, 3, 5, 10, 20, None]
    prob_aboves = [0.01, 0.05, 0.1, 0.3, None]

    results = []

    # Prepare base state
    state = ModelState(model, tokenizer, base_model_name=model_name, lora_rank=16, lora_alpha=16)
    state.frozen_ref = model # simple mock for KL
    
    for eta in etas:
        for rb in rank_belows:
            for pa in prob_aboves:
                if rb is None and pa is None:
                    continue # skipped
                
                log.info(f"Evaluating eta={eta}, rank_below={rb}, prob_above={pa}")

                engine = TrainEngine(state, lr=eta, per_objective=True)

                # Reset optimizer
                engine.reset_optimizer()

                # Calculate metrics for calibration corpus
                calib_samples = copy.deepcopy(calibration_corpus)
                for s in calib_samples:
                    s["rank_below"] = rb
                    s["prob_above"] = pa
                
                with torch.no_grad():
                    ul_res = unlike_loss(model, tokenizer, calib_samples, allow_unanchored=True)
                    trigger_rate_calib = ul_res["components"]["unlike_triggered"] / max(ul_res["components"]["unlike_n"], 1)

                neutral_samples = copy.deepcopy(neutral_corpus)
                for s in neutral_samples:
                    s["rank_below"] = rb
                    s["prob_above"] = pa
                
                with torch.no_grad():
                    ul_res_neutral = unlike_loss(model, tokenizer, neutral_samples, allow_unanchored=True)
                    false_fire_rate = ul_res_neutral["components"]["unlike_triggered"] / max(ul_res_neutral["components"]["unlike_n"], 1)

                # Measure single-shot correction success and p_bad delta
                # To do this safely, we will save the LoRA weights, do a step, measure, and restore.
                
                # Get pre-step p_bad and argmax
                pre_p_bads = []
                pre_argmax = []
                for s in calib_samples:
                    input_ids = torch.tensor([_prefix_ids(tokenizer, s["prefix"])]).to(model.device)
                    with torch.no_grad():
                        out = model(input_ids=input_ids, use_cache=False)
                        logits = out.logits[0, -1, :]
                        p_bad = F.softmax(logits.float(), dim=-1)[s["bad_token_id"]].item()
                        pre_p_bads.append(p_bad)
                        pre_argmax.append(logits.argmax().item() == s["bad_token_id"])

                # Do 1 step on calib_samples
                spec = {
                    "objective": "unlike",
                    "samples": calib_samples,
                    "batch_objectives": [
                        {"name": "kl_anchor", "scope": "target_position"}
                    ],
                    "kwargs": {
                        "allow_unanchored": True,
                        "effective_lr": eta,
                    }
                }
                
                # Save peft weights
                peft_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                
                engine.step(spec)

                # Get post-step p_bad and argmax
                post_p_bads = []
                post_argmax = []
                for s in calib_samples:
                    input_ids = torch.tensor([_prefix_ids(tokenizer, s["prefix"])]).to(model.device)
                    with torch.no_grad():
                        out = model(input_ids=input_ids, use_cache=False)
                        logits = out.logits[0, -1, :]
                        p_bad = F.softmax(logits.float(), dim=-1)[s["bad_token_id"]].item()
                        post_p_bads.append(p_bad)
                        post_argmax.append(logits.argmax().item() == s["bad_token_id"])

                # Measure grower set cardinality and watchlist hit rate via safety_monitor
                # (Need to run safety_monitor manually on post-step to get metrics)
                watchlist = [s["bad_token_id"] for s in calib_samples] + [s["good_token_id"] for s in calib_samples]
                
                sm_samples = [{"prefix": s["prefix"]} for s in calib_samples]
                
                padded = _pad_prefixes([_prefix_ids(tokenizer, s["prefix"]) for s in calib_samples], pad_id)
                input_ids = padded["input_ids"]
                attention_mask = padded["attention_mask"]

                sm_kwargs = {
                    "target_positions": [[len(_prefix_ids(tokenizer, s["prefix"])) - 1] for s in calib_samples],
                    "target_token_ids": [[s["good_token_id"]] for s in calib_samples],
                    "default_watchlist": watchlist,
                    "effective_lr": eta,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                
                with torch.no_grad():
                    # safety monitor doesn't need to be backwarded
                    try:
                        sm_res = safety_monitor_loss(model, tokenizer, sm_samples, **sm_kwargs)
                        grower_count = sm_res["components"].get("safety_monitor_grower_tokens", 0)
                        watchlist_hits = sm_res["components"].get("safety_monitor_watchlist_hits", 0)
                    except Exception as e:
                        log.error(f"SM failed: {e}")
                        grower_count = 0
                        watchlist_hits = 0

                # Restore peft weights
                model.load_state_dict(peft_state, strict=False)

                # Compute aggregated
                delta_p_bad = [post - pre for pre, post in zip(pre_p_bads, post_p_bads)]
                mean_delta = sum(delta_p_bad) / len(delta_p_bad)
                correction_success = sum(1 for pre, post in zip(pre_argmax, post_argmax) if pre and not post)
                total_pre_argmax = sum(1 for pre in pre_argmax if pre)
                success_rate = correction_success / total_pre_argmax if total_pre_argmax > 0 else 0.0

                cell_res = {
                    "eta": eta,
                    "rank_below": rb,
                    "prob_above": pa,
                    "trigger_rate": trigger_rate_calib,
                    "false_fire_rate": false_fire_rate,
                    "success_rate": success_rate,
                    "mean_delta_p_bad": mean_delta,
                    "grower_count": grower_count,
                    "watchlist_hits": watchlist_hits
                }
                results.append(cell_res)
                print(cell_res)

    # Dump results
    with open("lile/docs/research/unlike-defaults-calibration.md", "w") as f:
        f.write("# Unlike Calibration Sweep Results\n\n")
        f.write("| eta | rank_below | prob_above | trigger | false_fire | success | delta_p | growers | watchlist_hits |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['eta']} | {r['rank_below']} | {r['prob_above']} | {r['trigger_rate']:.2f} | {r['false_fire_rate']:.2f} | {r['success_rate']:.2f} | {r['mean_delta_p_bad']:.4f} | {r['grower_count']} | {r['watchlist_hits']} |\n")

if __name__ == "__main__":
    main()

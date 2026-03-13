# -*- coding: utf-8 -*-
"""CSE 517 Final Project: Full Scale Swahili Reproduction with Qwen 2.5 7B by Saan Popović, Mariana Shuman, and Antonio Ballesteros"""

import torch
import os
import re
import gc
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import load_file, save_file
from datasets import load_dataset, Dataset, interleave_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

# ENVIRONMENT SETUP & PATHING
# Local saving directory for model weights, datasets, and final results.
# The script is fully fault-tolerant and will safely resume progress from this directory.
BASE_DIR = "./CSE517_Reproduction"
os.makedirs(BASE_DIR, exist_ok=True)
print(f"Results will save to local directory: {BASE_DIR}")

# Global Config (LoRA Ranks & Alpha match Paper Appx A.4)
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "lr": 5e-5,  # LoRA usually needs higher LR than full FT (Paper uses ~4-9e-5 for LoRA)
    "batch_size": 4,
    "grad_accum": 16,  # 4 * 16 = Effective Batch Size 64
    "r": 64,
    "alpha": 16,
    "num_train_epochs": 1,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


# Helper Functions
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )


"""Load Data:"""
# PAPER REQUIREMENT: "Datasets are controlled at 80k samples"
TARGET_LIMIT = 80000
DATA_SAVE_PATH = os.path.join(BASE_DIR, "processed_datasets")
os.makedirs(DATA_SAVE_PATH, exist_ok=True)


# Standard ChatML Formatter
def to_chatml(u, a):
    return {
        "text": f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    }


# Helper to Save/Load from Cache (Avoids re-processing every run)
def get_or_create_dataset(name, filepath, creator_fn):
    if os.path.exists(filepath):
        print(f"Loading {name} from cache at {filepath}...")
        return load_from_disk(filepath)
    print(f"Creating {name} from scratch...")
    ds = creator_fn()
    ds.save_to_disk(filepath)
    return ds


# Math Dataset
def create_math():
    print("Loading Math Dataset...")
    math_raw = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    math_fmt = math_raw.map(
        lambda x: to_chatml(x["question"], x["answer"]),
        remove_columns=math_raw.column_names,
        num_proc=4,
    )
    return math_fmt.shuffle(seed=42).select(range(TARGET_LIMIT))


math_dataset = get_or_create_dataset("Math", f"{DATA_SAVE_PATH}/math", create_math)


# Language Data Generator
def stream_and_mix(name, streams, limit):
    print(f"  > Mixing {name} streams...")
    iters = [iter(s) for s in streams]
    data = []

    with tqdm(total=limit, desc=f"Extracting {name}") as pbar:
        while len(data) < limit:
            for it in iters:
                try:
                    row = next(it)
                    u, a = None, None
                    if "inputs" in row:
                        u, a = row["inputs"], row["targets"]
                    elif "instruction" in row:
                        u, a = row["instruction"], row["output"]
                    elif "native_instruction" in row:
                        u, a = row["native_instruction"], row["native_response"]

                    if u and a:
                        data.append(to_chatml(str(u), str(a)))
                        pbar.update(1)
                    if len(data) >= limit:
                        break
                except StopIteration:
                    continue
            if all(not it for it in iters):
                break

    return Dataset.from_list(data).select_columns(["text"]).shuffle(seed=42)


# Language Experts:
def create_sw():
    streams = [
        load_dataset(
            "CohereForAI/aya_collection_language_split",
            "swahili",
            split="train",
            streaming=True,
        ),
        load_dataset("lelapa/Inkuba-instruct", split="swahili_train", streaming=True),
        load_dataset("bigscience/xP3mt", "sw", split="train", streaming=True),
    ]
    return stream_and_mix("Swahili", streams, TARGET_LIMIT)


sw_expert_train = get_or_create_dataset(
    "Swahili", f"{DATA_SAVE_PATH}/swahili", create_sw
)


def create_bn():
    streams = [
        load_dataset(
            "CohereForAI/aya_collection_language_split",
            "bengali",
            split="train",
            streaming=True,
        ),
        load_dataset("lumatic-ai/BongChat-v1-253k", split="train", streaming=True),
        load_dataset(
            "ai4bharat/indic-align", "Indic_ShareLlama", split="train", streaming=True
        ),
    ]
    return stream_and_mix("Bengali", streams, TARGET_LIMIT)


bn_expert_train = get_or_create_dataset(
    "Bengali", f"{DATA_SAVE_PATH}/bengali", create_bn
)


def create_te():
    streams = [
        load_dataset(
            "CohereForAI/aya_collection_language_split",
            "telugu",
            split="train",
            streaming=True,
        ),
        load_dataset("bigscience/xP3mt", "te", split="train", streaming=True),
        load_dataset(
            "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized",
            split="train",
            streaming=True,
        ),
    ]
    return stream_and_mix("Telugu", streams, TARGET_LIMIT)


te_expert_train = get_or_create_dataset("Telugu", f"{DATA_SAVE_PATH}/telugu", create_te)

# Mixed Dataset
print("Creating Mixed Dataset for Baseline...")
all_langs = interleave_datasets(
    [te_expert_train, bn_expert_train, sw_expert_train],
    stopping_strategy="all_exhausted",
)
mixed_dataset = (
    interleave_datasets([math_dataset, all_langs])
    .shuffle(seed=42)
    .take(TARGET_LIMIT * 2)
)

print(f"Production Data Ready. Mixed Dataset Size: {len(mixed_dataset)}")


print("Verification: config & tokenizer")
assert CONFIG["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
assert CONFIG["r"] == 64
assert CONFIG["alpha"] == 16
print("Configuration checks passed.")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tokenizer.pad_token = tokenizer.eos_token
print(f" Tokenizer loaded: {tokenizer.name_or_path}")


"""Expert Training Functions:"""


def train_expert(dataset, output_dir, run_name):
    final_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(final_model_path):
        print(f"{run_name} exists at {output_dir}. Skipping.")
        return

    print(f"\nTraining: {run_name}")
    clean_memory()

    model = get_base_model()
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=CONFIG["r"],
        lora_alpha=CONFIG["alpha"],
        target_modules=CONFIG["target_modules"],
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    last_checkpoint = (
        get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(output_dir)
    del model, trainer
    clean_memory()
    print(f"{run_name} Complete.")


def train_simultaneous(math_ds, lang_ds, output_dir):
    if os.path.exists(output_dir):
        print(f"Simultaneous model exists at {output_dir}. Skipping.")
        return

    print("\nTRAINING: Simultaneous SFT (Modular Partitioning)")
    clean_memory()

    model = prepare_model_for_kbit_training(get_base_model())
    peft_config = LoraConfig(
        r=CONFIG["r"],
        lora_alpha=CONFIG["alpha"],
        target_modules=CONFIG["target_modules"],
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    def tokenize_with_labels(batch):
        tokens = tokenizer(
            batch, truncation=True, max_length=1024, padding=True, return_tensors="pt"
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens.to(model.device)

    TOTAL_STEPS = 2000
    num_layers = model.base_model.model.config.num_hidden_layers
    lang_layers = set(list(range(0, 6)) + list(range(num_layers - 2, num_layers)))

    math_iter = iter(math_ds)
    lang_iter = iter(lang_ds)
    model.train()

    for step in range(TOTAL_STEPS):
        try:
            # Math Step (Middle Layers)
            for name, param in model.named_parameters():
                if "lora" in name:
                    match = re.search(r"\.layers\.(\d+)\.", name)
                    idx = int(match.group(1)) if match else -1
                    param.requires_grad = idx != -1 and idx not in lang_layers

            optimizer.zero_grad()
            batch_math = [next(math_iter)["text"] for _ in range(CONFIG["batch_size"])]
            loss_math = model(**tokenize_with_labels(batch_math)).loss
            loss_math.backward()
            optimizer.step()

            # Language Step (Outer Layers)
            for name, param in model.named_parameters():
                if "lora" in name:
                    match = re.search(r"\.layers\.(\d+)\.", name)
                    idx = int(match.group(1)) if match else -1
                    param.requires_grad = idx != -1 and idx in lang_layers

            optimizer.zero_grad()
            batch_lang = [next(lang_iter)["text"] for _ in range(CONFIG["batch_size"])]
            loss_lang = model(**tokenize_with_labels(batch_lang)).loss
            loss_lang.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"Step {step}/{TOTAL_STEPS} | Math Loss: {loss_math.item():.4f} | Lang Loss: {loss_lang.item():.4f}"
                )

        except StopIteration:
            break

    model.save_pretrained(output_dir)
    del model, optimizer
    clean_memory()
    print("Simultaneous Training Complete.")


"""Layer Swapping"""


def merge_layer_swapping(math_adapter_path, lang_adapter_path, output_dir):
    final_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(final_model_path):
        print(f"Merge output already exists at {output_dir}. Skipping.")
        return

    print(f"\nMerging: {math_adapter_path} + {lang_adapter_path} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    math_tensors = load_file(
        os.path.join(math_adapter_path, "adapter_model.safetensors")
    )
    lang_tensors = load_file(
        os.path.join(lang_adapter_path, "adapter_model.safetensors")
    )

    layer_indices = [
        int(re.search(r"\.layers\.(\d+)\.", k).group(1))
        for k in math_tensors.keys()
        if ".layers." in k
    ]
    num_layers = max(layer_indices) + 1

    lang_indices = set(list(range(0, 6)) + list(range(num_layers - 2, num_layers)))

    merged = {}
    for k in math_tensors.keys():
        layer_match = re.search(r"\.layers\.(\d+)\.", k)
        if layer_match:
            idx = int(layer_match.group(1))
            if idx in lang_indices:
                merged[k] = lang_tensors[k]
            else:
                merged[k] = math_tensors[k]
        else:
            merged[k] = math_tensors[k]

    save_file(merged, os.path.join(output_dir, "adapter_model.safetensors"))
    shutil.copy(
        os.path.join(math_adapter_path, "adapter_config.json"),
        os.path.join(output_dir, "adapter_config.json"),
    )
    print("Merge Complete.")


"""Math-Only"""
math_output_dir = f"{BASE_DIR}/math_expert_lora"
math_run_name = "Math-Only Expert"
final_math_file = os.path.join(math_output_dir, "adapter_model.safetensors")

if not os.path.exists(final_math_file):
    print(f"Starting Training for {math_run_name}...")
    clean_memory()
    train_expert(
        dataset=math_dataset, output_dir=math_output_dir, run_name=math_run_name
    )
else:
    print(f"{math_run_name} is already complete at {math_output_dir}. Skipping.")


"""Language-Only"""
language_experts = [
    {
        "dataset": sw_expert_train,
        "path": f"{BASE_DIR}/lang_expert_swahili_lora",
        "name": "Swahili Expert",
    },
    {
        "dataset": bn_expert_train,
        "path": f"{BASE_DIR}/lang_expert_bengali_lora",
        "name": "Bengali Expert",
    },
    {
        "dataset": te_expert_train,
        "path": f"{BASE_DIR}/lang_expert_telugu_lora",
        "name": "Telugu Expert",
    },
]

print("Starting Language Expert Training (Safe Mode)")
for expert in language_experts:
    final_model_file = os.path.join(expert["path"], "adapter_model.safetensors")
    if not os.path.exists(final_model_file):
        print(f"\n>>> Preparing to train: {expert['name']}")
        clean_memory()
        train_expert(
            dataset=expert["dataset"],
            output_dir=expert["path"],
            run_name=expert["name"],
        )
    else:
        print(
            f"\n>>>{expert['name']} is already complete at {expert['path']}. Skipping."
        )


"""Results and Tables"""


def evaluate_model(model, tokenizer, languages, batch_size=8):
    results = {}

    def prompt_fn(q):
        return f"""<|im_start|>user
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: <|im_end|>
<|im_start|>assistant
There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is 6.<|im_end|>
<|im_start|>user
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: <|im_end|>
<|im_start|>assistant
There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.<|im_end|>
<|im_start|>user
Question: {q}
Answer: <|im_end|>
<|im_start|>assistant
"""

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nBeginning Batched Evaluation (Batch Size: {batch_size})")

    for lang in languages:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception as e:
            print(f"Failed to load {lang}: {e}")
            results[lang] = 0.0
            continue

        correct = 0
        print(f"Evaluating {lang.upper()} ({len(ds)} samples)...")

        for i in tqdm(range(0, len(ds), batch_size), desc=f"Scoring {lang.upper()}"):
            batch_items = ds[i : i + batch_size]
            prompts = [prompt_fn(q) for q in batch_items["question"]]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )

            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.0,
                    do_sample=False,
                )

            for j, generated_tokens in enumerate(out):
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                parts = text.split("assistant")
                response_text = parts[-1].replace(",", "")
                numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
                pred = numbers[-1] if numbers else "0"

                try:
                    target = float(
                        str(batch_items["answer_number"][j]).replace(",", "")
                    )
                    if float(pred) == target:
                        correct += 1
                except:
                    continue

        score = (correct / len(ds)) * 100
        results[lang] = score
        print(f"   >> {lang.upper()} Score: {score:.1f}%")

    return results


if __name__ == "__main__":
    print("Beginning Reproduction...")

    CONFIG["lr"] = 5e-6
    print(f"Updated Learning Rate to {CONFIG['lr']} for Paper Compliance")

    json_path = f"{BASE_DIR}/smart_reproduction_results.json"
    if os.path.exists(json_path):
        print("Found existing results JSON. Resuming progress...")
        with open(json_path, "r") as f:
            final_results = json.load(f)
    else:
        final_results = {"Qwen2.5": {}}

    print("\n" + "=" * 60)
    print("Reproduction on qwen 2.5")
    print("=" * 60)

    qwen_base_dir = f"{BASE_DIR}/results/qwen2.5"
    os.makedirs(qwen_base_dir, exist_ok=True)

    # Route to the isolated persistent math expert directory to support fault tolerance
    math_expert_dir = f"{BASE_DIR}/math_expert_lora"
    train_expert(math_dataset, math_expert_dir, "Qwen Math-Only")

    # NOTE FOR EVALUATOR: Bengali and Telugu are commented out due to runtime constraints
    language_configs = [
        ("sw", "Swahili", sw_expert_train),
        # ("bn", "Bengali", bn_expert_train),
        # ("te", "Telugu", te_expert_train),
    ]

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    for lang_code, lang_name, lang_ds in language_configs:
        print(f"\n--- Processing Language: {lang_name.upper()} ---")
        lang_dir = f"{qwen_base_dir}/{lang_name.lower()}"
        os.makedirs(lang_dir, exist_ok=True)

        mixed_ds = (
            interleave_datasets([math_dataset, lang_ds]).shuffle(seed=42).take(160000)
        )

        # Dynamic routing to ensure fault-tolerance matches the isolated cell outputs
        if lang_code == "sw":
            lang_expert_dir = f"{BASE_DIR}/lang_expert_swahili_lora"
        elif lang_code == "bn":
            lang_expert_dir = f"{BASE_DIR}/lang_expert_bengali_lora"
        elif lang_code == "te":
            lang_expert_dir = f"{BASE_DIR}/lang_expert_telugu_lora"
        else:
            lang_expert_dir = f"{lang_dir}/lang_expert"

        data_mixing_dir = f"{lang_dir}/data_mixing"
        simultaneous_dir = f"{lang_dir}/simultaneous"
        layer_swapping_dir = f"{lang_dir}/layer_swapping"

        train_expert(lang_ds, lang_expert_dir, f"Qwen {lang_name}-Only")
        train_expert(mixed_ds, data_mixing_dir, f"Qwen Data-Mixing ({lang_name})")
        train_simultaneous(math_dataset, lang_ds, simultaneous_dir)
        merge_layer_swapping(math_expert_dir, lang_expert_dir, layer_swapping_dir)

        eval_paths = {
            "Base Model": None,
            "Math-Only": math_expert_dir,
            "Lang-Only": lang_expert_dir,
            "Data-Mixing": data_mixing_dir,
            "Simultaneous": simultaneous_dir,
            "Layer-Swapping": layer_swapping_dir,
        }

        for model_name, path in eval_paths.items():
            if model_name in final_results.get(
                "Qwen2.5", {}
            ) and lang_code.upper() in final_results["Qwen2.5"].get(model_name, {}):
                print(
                    f"Skipping {model_name} on {lang_name} (Already evaluated in JSON)"
                )
                continue

            print(f"Evaluating {model_name} on {lang_name} & English...")
            base = get_base_model()
            model = PeftModel.from_pretrained(base, path) if path else base

            scores = evaluate_model(model, tokenizer, languages=["en", lang_code])

            if model_name not in final_results["Qwen2.5"]:
                final_results["Qwen2.5"][model_name] = {}

            final_results["Qwen2.5"][model_name][lang_code.upper()] = scores.get(
                lang_code, 0.0
            )
            final_results["Qwen2.5"][model_name]["EN"] = scores.get("en", 0.0)

            with open(json_path, "w") as f:
                json.dump(final_results, f, indent=4)
            print(f"Progress safely saved to {json_path}")

            del model, base
            clean_memory()

    for model_name in final_results["Qwen2.5"]:
        scores = final_results["Qwen2.5"][model_name]
        avg_score = (
            scores.get("SW", 0) + scores.get("BN", 0) + scores.get("TE", 0)
        ) / 3
        final_results["Qwen2.5"][model_name]["AVG"] = avg_score

    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "=" * 85)
    print("Reproduction complete! Summary:")
    print("=" * 85)

    columns = [
        "Base Model",
        "Data-Mixing",
        "Math-Only",
        "Lang-Only",
        "Simultaneous",
        "Layer-Swapping",
    ]
    rows = [
        ("Bengali", "BN"),
        ("Telugu", "TE"),
        ("Swahili", "SW"),
        ("English", "EN"),
        ("AVG", "AVG"),
    ]

    header_str = f"{'':<14} | " + " | ".join([f"{col:<14}" for col in columns])
    print(header_str)
    print("-" * 115)

    for display_name, lang_key in rows:
        row_str = f"{display_name:<14} | "
        for col in columns:
            score = final_results["Qwen2.5"].get(col, {}).get(lang_key, 0.0)
            row_str += f"{score:<14.1f} | "
        print(row_str)


"""Additional Visualizations"""


def merge_layer_swapping_5050(math_adapter_path, lang_adapter_path, output_dir):
    final_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(final_model_path):
        print(f"50/50 Merge output already exists at {output_dir}. Skipping.")
        return

    print(
        f"\nMerging 50/50 Ablation: {math_adapter_path} + {lang_adapter_path} -> {output_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)

    math_tensors = load_file(
        os.path.join(math_adapter_path, "adapter_model.safetensors")
    )
    lang_tensors = load_file(
        os.path.join(lang_adapter_path, "adapter_model.safetensors")
    )

    # Qwen 2.5 7B has 28 layers. Bottom 14 (0-13) to Language syntax, Top 14 to Math reasoning.
    lang_indices = set(range(0, 14))

    merged = {}
    for k in math_tensors.keys():
        layer_match = re.search(r"\.layers\.(\d+)\.", k)
        if layer_match:
            idx = int(layer_match.group(1))
            if idx in lang_indices:
                merged[k] = lang_tensors[k]
            else:
                merged[k] = math_tensors[k]
        else:
            merged[k] = math_tensors[k]

    save_file(merged, os.path.join(output_dir, "adapter_model.safetensors"))
    shutil.copy(
        os.path.join(math_adapter_path, "adapter_config.json"),
        os.path.join(output_dir, "adapter_config.json"),
    )
    print("50/50 Merge Complete.")


def generate_results_chart(json_path):
    with open(json_path, "r") as f:
        data = json.load(f).get("Qwen2.5", {})

    languages = ["SW", "BN", "TE"]
    models = [
        "Base Model",
        "Data-Mixing",
        "Simultaneous",
        "Layer-Swapping",
        "Layer-Swapping (50/50)",
    ]

    scores = {
        model: [data.get(model, {}).get(lang, 0) for lang in languages]
        for model in models
        if model in data
    }

    x = np.arange(len(languages))
    width = 0.15
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")

    for attribute, measurement in scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt="%.1f")
        multiplier += 1

    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("MGSM Performance by Training Paradigm (Qwen 2.5 7B)")
    ax.set_xticks(x + width * (len(scores) - 1) / 2, languages)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    chart_path = os.path.join(os.path.dirname(json_path), "mgsm_results_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved publication-ready chart to: {chart_path}")
    plt.show()


# EXECUTION SCRIPT
print("Starting bonus ablation & visualization...")

json_path = f"{BASE_DIR}/smart_reproduction_results.json"
qwen_base_dir = f"{BASE_DIR}/results/qwen2.5"
# Route to the standalone math expert directory
math_expert_dir = f"{BASE_DIR}/math_expert_lora"

if os.path.exists(json_path):
    with open(json_path, "r") as f:
        final_results = json.load(f)
else:
    print(
        "ERROR: Could not find smart_reproduction_results.json. Make sure the main run finished!"
    )
    final_results = {"Qwen2.5": {}}

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tokenizer.pad_token = tokenizer.eos_token

languages = [
    ("sw", "Swahili"),
    # ("bn", "Bengali"),
    # ("te", "Telugu")
]

for lang_code, lang_name in languages:
    lang_dir = f"{qwen_base_dir}/{lang_name.lower()}"
    layer_swapping_5050_dir = f"{lang_dir}/layer_swapping_5050"
    model_name = "Layer-Swapping (50/50)"

    # Dynamic routing to the standalone language expert directories
    if lang_code == "sw":
        lang_expert_dir = f"{BASE_DIR}/lang_expert_swahili_lora"
    elif lang_code == "bn":
        lang_expert_dir = f"{BASE_DIR}/lang_expert_bengali_lora"
    elif lang_code == "te":
        lang_expert_dir = f"{BASE_DIR}/lang_expert_telugu_lora"
    else:
        lang_expert_dir = f"{lang_dir}/lang_expert"

    if os.path.exists(math_expert_dir) and os.path.exists(lang_expert_dir):
        merge_layer_swapping_5050(
            math_expert_dir, lang_expert_dir, layer_swapping_5050_dir
        )
    else:
        print(f"Skipping 50/50 merge for {lang_name}: Expert models not found.")
        continue

    if model_name in final_results.get(
        "Qwen2.5", {}
    ) and lang_code.upper() in final_results["Qwen2.5"].get(model_name, {}):
        print(f"Skipping {model_name} on {lang_name} (Already evaluated in JSON)")
    else:
        print(f"Evaluating {model_name} on {lang_name} & English...")
        base = get_base_model()
        model = PeftModel.from_pretrained(base, layer_swapping_5050_dir)

        scores = evaluate_model(model, tokenizer, languages=["en", lang_code])

        if model_name not in final_results["Qwen2.5"]:
            final_results["Qwen2.5"][model_name] = {}

        final_results["Qwen2.5"][model_name][lang_code.upper()] = scores.get(
            lang_code, 0.0
        )
        final_results["Qwen2.5"][model_name]["EN"] = scores.get("en", 0.0)

        with open(json_path, "w") as f:
            json.dump(final_results, f, indent=4)

        del model, base
        clean_memory()

generate_results_chart(json_path)
print("ALL DONE! Check your directory for the mgsm_results_chart.png file.")

import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import glob
import random
import torch
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# ================= 配置区域 =================
class Config:
    # 1. 基础模型路径
    BASE_MODEL_PATH = f"{PEAP_ROOT}/models/base/Meta-Llama-3-8B-Instruct"
    
    # 2. LoRA 路径 (!!! 请根据实际训练结果修改 checkpoint !!!)
    # 建议使用训练输出目录下的 final_model 或者效果最好的 checkpoint
    LORA_PATHS = {
        "image":  f"{PEAP_ROOT}/models/reward_judges/llama3/llama3_rm_output_image_optimized",
        "audio":  f"{PEAP_ROOT}/models/reward_judges/llama3/llama3_rm_output_audio_optimized",
        "assist": f"{PEAP_ROOT}/models/reward_judges/llama3/llama3_rm_output_assist_optimized"
    }

    # 3. 数据集路径
    # 用于 Test 1: 给模型打分 (总数据集)
    DATASET_ROOT = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
    # 用于 Test 2: 对齐度测试 (训练时生成的 jsonl 目录)
    TEST_DATA_DIR = f"{PEAP_ROOT}/code/training/Finetune_llama/data_output" 
    
    # 4. 采样设置
    SAMPLE_SIZE = 101
    RANDOM_SEED = 1180
    
    # 5. 待测模型列表
    TARGET_MODELS = [
        "Llama-3-8B-Instruct.txt", "minicpm_new_prompt.txt", "MiniOmni_cascade_format.txt",
        "Qwen3-0.6B.txt", "Qwen3-8B.txt", "Qwen3-32B.txt",
        "Qwen3Omni_final.txt", "StreamOmni_format.txt", "VITA.txt"
    ]
    
    # 注意：Llama3 训练时是把 instruction 放进 System Message，input 放进 User Message
    # 这里定义 Instruction 模板，用于给待测模型打分时的构建
    BASE_INSTRUCTION = """The definition of this task is as follows: A robot can proactively engage in conversations with humans based on the scenes it observes and the sounds it hears. Currently, the model under test is in a scene depicted in an image and can simultaneously hear a background audio clip."""
    
    INSTRUCTIONS = {
        "image": BASE_INSTRUCTION + """
Please judge whether the model under test **correctly identifies the visual scene information**.
Focus ONLY on the visual elements described in the [Image Context] and [Ground Truth]. Ignore audio or assistance errors if the visual description is correct.
Output "1" if the model correctly perceives the visual environment. Output "0" if it hallucinates objects or misidentifies the scene.""",

        "audio": BASE_INSTRUCTION + """
Please judge whether the model under test **correctly identifies the background audio information**.
Focus ONLY on the auditory elements described in the [Audio Context].
Output "1" if the model correctly perceives the background sound. Output "0" if it ignores obvious sounds or hallucinates sounds.""",

        "assist": BASE_INSTRUCTION + """
Please judge whether the model under test **correctly understands the assistance requirements (intent)**.
Focus ONLY on whether the robot's proposed action or response aligns with the [Logic Analysis] and [Ground Truth].
Output "1" if the intent/action direction is correct. Output "0" if the action is irrelevant or counter-productive."""
    }

cfg = Config()

# ================= 工具函数 =================

def load_model_and_tokenizer(lora_path):
    """
    每次评测不同任务时，必须重新加载基座模型和对应的 LoRA，
    以确保 SequenceClassification 的 score head 权重正确加载。
    """
    print(f"⏳ Loading Tokenizer & Model (4-bit) from {cfg.BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" # 分类任务保持与训练一致

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    
    # 这里的 num_labels=2 很重要，它会初始化一个 score 层
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.BASE_MODEL_PATH, 
        num_labels=2,
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    # 必须同步 pad token id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"📂 Loading LoRA Adapter: {lora_path}")
    # PeftModel 会自动识别并加载 modules_to_save (即 score 层)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model, tokenizer

def build_multimodal_content(case_dir: Path) -> dict:
    info = {"logic": "N/A", "gt": "N/A", "image_gt": "N/A", "audio_gt": "N/A"}
    p = case_dir / "logic.txt"
    if p.exists(): info["logic"] = p.read_text(encoding='utf-8', errors='ignore').strip()
    p = case_dir / "GT.txt"
    if p.exists(): info["gt"] = p.read_text(encoding='utf-8', errors='ignore').strip()
    for ext, key in [(".jpg", "image_gt"), (".wav", "audio_gt")]:
        files = list(case_dir.glob(f"*{ext}"))
        if files:
            gt_txt = case_dir / f"{files[0].stem}_GT.txt"
            if gt_txt.exists(): info[key] = gt_txt.read_text(encoding='utf-8', errors='ignore').strip()
    return info

def format_input(task_type: str, info: dict, model_output: str) -> str:
    """构造 User Input，必须与 make_data 时的 Prompt 结构完全一致"""
    prompt_template = ""
    if task_type == "image":
        prompt_template = "\n[Image Context]: {image_gt}\n[Ground Truth]: {gt}\n--------------------------------------------------\n[Model Response to Evaluate]:\n{model_output}\n\nBased on the visual context, is the model's visual perception correct? (Answer 0 for No, 1 for Yes)\n"
    elif task_type == "audio":
        prompt_template = "\n[Audio Context]: {audio_gt}\n[Ground Truth]: {gt}\n--------------------------------------------------\n[Model Response to Evaluate]:\n{model_output}\n\nBased on the audio context, is the model's auditory perception correct? (Answer 0 for No, 1 for Yes)\n"
    elif task_type == "assist":
        prompt_template = "\n[Logic Analysis]: {logic}\n[Ground Truth]: {gt}\n--------------------------------------------------\n[Model Response to Evaluate]:\n{model_output}\n\nBased on the scenario logic, is the model's assistance intent correct? (Answer 0 for No, 1 for Yes)\n"
    
    return prompt_template.format(
        image_gt=info["image_gt"], audio_gt=info["audio_gt"], logic=info["logic"], gt=info["gt"], model_output=model_output
    )

def predict_one_sample(model, tokenizer, instruction, user_input):
    """单条样本预测逻辑"""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048 # 与训练保持一致
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # SequenceClassification 输出的是 logits [batch_size, num_labels]
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item() # 0 or 1
    
    return pred_label

# ================= 任务 1: 人类对齐度测试 =================

def load_test_jsonl(task_name):
    path = os.path.join(cfg.TEST_DATA_DIR, f"sft_{task_name}_test.jsonl")
    if not os.path.exists(path): return []
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f: data.append(json.loads(line))
    return data

def test_1_eval_alignment():
    print("\n" + "#"*60)
    print("🧪 TEST 1: Evaluating Alignment with Human Labels (Accuracy)")
    print("#"*60)

    results_table = []

    for task_name in ["image", "audio", "assist"]:
        lora_path = cfg.LORA_PATHS.get(task_name)
        if not os.path.exists(lora_path): 
            print(f"⚠️ LoRA path not found: {lora_path}"); continue

        print(f"\nEvaluating Alignment for Task: [{task_name}]")
        test_data = load_test_jsonl(task_name)
        if not test_data: print("No test data found."); continue
        
        # 随机抽样 300 条
        if len(test_data) > 300:
            random.seed(cfg.RANDOM_SEED)
            test_data = random.sample(test_data, 300)
        
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(lora_path)
        
        true_labels = []
        pred_labels = []

        for item in tqdm(test_data, desc="Inferencing"):
            try:
                # 解析 GT
                label_str = str(item.get("output", "0")).strip()
                gt = 1 if "1" in label_str else 0
                true_labels.append(gt)

                # 读取原始 Instruction 和 Input
                # 训练时我们就是直接用 jsonl 里的这两个字段
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                
                pred = predict_one_sample(model, tokenizer, instruction, inp)
                pred_labels.append(pred)
            except Exception as e:
                pass

        # 计算指标
        acc = accuracy_score(true_labels, pred_labels)
        p, r, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
        
        print(f"✅ {task_name} Results: Acc={acc:.4f}, F1={f1:.4f}")
        results_table.append({
            "Task": task_name, "Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1, "Samples": len(true_labels)
        })

        # 清理显存
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n📊 Human Alignment Report")
    print(f"{'Task':<10} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'Samples'}")
    print("-" * 65)
    for row in results_table:
        print(f"{row['Task']:<10} | {row['Accuracy']:.4f}   | {row['Precision']:.4f}   | {row['Recall']:.4f}   | {row['F1-Score']:.4f}   | {row['Samples']}")


# ================= 任务 2: 给待测模型打分 =================

def test_2_score_models():
    print("\n" + "#"*60)
    print("🧪 TEST 2: Scoring Target Models (100 Samples)")
    print("#"*60)
    
    # 1. 采样
    all_csvs = glob.glob(os.path.join(cfg.DATASET_ROOT, "**", "QwenScore.csv"), recursive=True)
    random.seed(cfg.RANDOM_SEED)
    sampled_csvs = random.sample(all_csvs, min(cfg.SAMPLE_SIZE, len(all_csvs)))
    print(f"🎲 Sampled {len(sampled_csvs)} cases.")

    # 结果容器
    final_scores = {m: {"image": [], "audio": [], "assist": []} for m in cfg.TARGET_MODELS}

    # 2. 依次评测
    for task_name in ["image", "audio", "assist"]:
        lora_path = cfg.LORA_PATHS.get(task_name)
        if not os.path.exists(lora_path): continue

        model, tokenizer = load_model_and_tokenizer(lora_path)
        sys_prompt = cfg.INSTRUCTIONS[task_name]

        for csv_path in tqdm(sampled_csvs, desc=f"Scoring [{task_name}]"):
            case_dir = Path(csv_path).parent
            info = build_multimodal_content(case_dir)

            for target_model in cfg.TARGET_MODELS:
                output_file = case_dir / target_model
                if not output_file.exists(): continue
                
                try:
                    resp = output_file.read_text(encoding='utf-8', errors='ignore').strip()
                    if not resp: continue
                    
                    # 构造输入，必须和训练时的 make_data 逻辑对齐
                    user_input = format_input(task_name, info, resp)
                    
                    pred = predict_one_sample(model, tokenizer, sys_prompt, user_input)
                    final_scores[target_model][task_name].append(pred)

                except Exception as e:
                    pass

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. 打印报表
    print("\n📊 Model Scoring Leaderboard")
    print(f"{'Model Name':<30} | {'Image':<8} | {'Audio':<8} | {'Assist':<8} | {'Avg':<8}")
    print("-" * 75)
    
    rows = []
    for m, scores in final_scores.items():
        row = {"name": m, "avg": 0}
        total_s = 0; count = 0
        for t in ["image", "audio", "assist"]:
            if scores[t]:
                s = sum(scores[t])/len(scores[t])
                row[t] = s
                total_s += s; count += 1
            else:
                row[t] = -1
        row["avg"] = total_s / count if count > 0 else 0
        rows.append(row)
    
    rows.sort(key=lambda x: x["avg"], reverse=True)
    
    for r in rows:
        img_s = f"{r['image']:.4f}" if r['image']>=0 else "N/A"
        aud_s = f"{r['audio']:.4f}" if r['audio']>=0 else "N/A"
        ast_s = f"{r['assist']:.4f}" if r['assist']>=0 else "N/A"
        print(f"{r['name']:<30} | {img_s:<8} | {aud_s:<8} | {ast_s:<8} | {r['avg']:.4f}")

if __name__ == "__main__":
    # 执行两个任务
    test_1_eval_alignment()
    test_2_score_models()
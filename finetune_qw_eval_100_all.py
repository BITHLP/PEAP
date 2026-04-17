import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import glob
import random
import torch
import re
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================= 配置区域 =================
class Config:
    # 1. 基础模型
    BASE_MODEL_PATH = f"{PEAP_ROOT}/models/base/Qwen3-8B"
    
    # 2. LoRA 路径 (使用你验证过有效的那个 checkpont 或 final_model)
    LORA_PATHS = {
        "image":  f"{PEAP_ROOT}/models/reward_judges/qwen3/qwen3-8b-reward-lora_image/final_model",
        "audio":  f"{PEAP_ROOT}/models/reward_judges/qwen3/qwen3-8b-reward-lora_audio/final_model",
        "assist": f"{PEAP_ROOT}/models/reward_judges/qwen3/qwen3-8b-reward-lora_assist/final_model"
    }

    # 3. 总数据集路径 (用于抽取 100 个用例)
    DATASET_ROOT = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
    
    # 4. 采样设置
    SAMPLE_SIZE = 101
    RANDOM_SEED = 42
    
    # 5. 待测模型列表
    TARGET_MODELS = [
        "Llama-3-8B-Instruct.txt", "minicpm_new_prompt.txt", "MiniOmni_cascade_format.txt",
        "Qwen3-0.6B.txt", "Qwen3-8B.txt", "Qwen3-32B.txt",
        "Qwen3Omni_final.txt", "StreamOmni_format.txt", "VITA.txt"
    ]

    # 6. System Prompts (与训练保持一致)
    SYSTEM_PROMPTS = {
        "image": "You are a visual evaluation expert. Your job is to strictly verify if the robot's description matches the image scene.",
        "audio": "You are an auditory evaluation expert. Your job is to strictly verify if the robot's perception matches the background audio.",
        "assist": "You are a robot behavior expert. Your job is to evaluate if the robot's actions are appropriate for the user's needs in the scene."
    }

    # 7. 任务指令模板 (与 make_data 保持一致)
    # 这里不需要完整的 Instruction 文本，因为 System Prompt 已经涵盖了角色设定
    # 我们只需要构造 Input 部分
    pass

cfg = Config()

# ================= 核心工具函数 =================

def extract_label(gen_text: str) -> int:
    """鲁棒的标签提取函数"""
    text = gen_text.strip()
    # 1. 优先找独立的数字
    match = re.search(r'\b([01])\b', text)
    if match: return int(match.group(1))
    # 2. 开头匹配
    if text.startswith("0"): return 0
    if text.startswith("1"): return 1
    # 3. 包含检测
    if "0" in text and "1" not in text: return 0
    if "1" in text and "0" not in text: return 1
    return 0 # 默认

def load_model_and_tokenizer(lora_path):
    print(f"⏳ Loading Tokenizer & Model from {cfg.BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # !!! 关键修复: 推理必须左填充 !!!
    tokenizer.padding_side = "left" 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL_PATH, quantization_config=bnb_config, 
        device_map="auto", trust_remote_code=True
    )
    
    print(f"📂 Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model, tokenizer

def build_multimodal_content(case_dir: Path) -> dict:
    """读取上下文文件"""
    info = {"logic": None, "gt": None, "image_gt": None, "audio_gt": None}
    
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

def construct_input_text(task_type: str, info: dict, model_output: str) -> str:
    """
    手动构造与 make_data_multi_task.py 完全一致的 Input 格式
    格式参考:
    Scene information:{content}
    Model output to be evaluated:{output}
    Please provide a rating (0/1) for {task} correctness:
    """
    content_parts = []
    
    # 这里的拼接逻辑需要完全复刻 make_data 中的 build_multimodal_content
    # 注意：make_data 里是把所有能读到的都拼起来了，没有按任务过滤
    # 但 instruction 是按任务区分的
    
    # 1. Logic & GT
    if info["logic"]: content_parts.append(f"Scenario logic:\n{info['logic']}")
    if info["gt"]: content_parts.append(f"Ground Truth:\n{info['gt']}")
    
    # 2. Media GT (这里为了简单，我们尽量把有的都放进去，或者按任务放)
    # 为了保证和训练分布一致，建议按 make_data 的逻辑：全放
    if info["image_gt"]: content_parts.append(f"Image Description:\n{info['image_gt']}") # 注意前缀要和训练数据一致，训练数据可能是 "xxx描述:" 或文件名
    if info["audio_gt"]: content_parts.append(f"Audio Description:\n{info['audio_gt']}")

    multimodal_content = "\n\n".join(content_parts)
    
    input_text = (
        f"Scene information:{multimodal_content}\n"
        f"Model output to be evaluated:{model_output}\n\n"
        f"Please provide a rating (0/1) for {task_type} correctness:"
    )
    return input_text

# ================= 主程序 =================

def main():
    print("\n" + "#"*60)
    print("🏆 Qwen3 Reward Model Leaderboard Evaluation (100 Samples)")
    print("#"*60)
    
    # 1. 采样 100 个用例
    print(f"🔍 Scanning dataset: {cfg.DATASET_ROOT}")
    all_csvs = glob.glob(os.path.join(cfg.DATASET_ROOT, "**", "QwenScore.csv"), recursive=True)
    
    random.seed(cfg.RANDOM_SEED)
    if len(all_csvs) < cfg.SAMPLE_SIZE:
        sampled_csvs = all_csvs
    else:
        sampled_csvs = random.sample(all_csvs, cfg.SAMPLE_SIZE)
    print(f"🎲 Randomly sampled {len(sampled_csvs)} cases.")

    # 结果容器: {model_name: {image: [], audio: [], assist: []}}
    leaderboard_data = {m: {"image": [], "audio": [], "assist": []} for m in cfg.TARGET_MODELS}

    # 2. 依次评测三个任务
    for task_name in ["image", "audio", "assist"]:
        lora_path = cfg.LORA_PATHS.get(task_name)
        if not os.path.exists(lora_path):
            print(f"⚠️ LoRA path not found for {task_name}, skipping.")
            continue
            
        print(f"\n🚀 Evaluating Task: [{task_name}]")
        
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(lora_path)
        sys_prompt = cfg.SYSTEM_PROMPTS[task_name]
        
        # 遍历用例
        for csv_path in tqdm(sampled_csvs, desc=f"Scoring {task_name}"):
            case_dir = Path(csv_path).parent
            info = build_multimodal_content(case_dir)
            
            # 遍历 9 个模型
            for target_model in cfg.TARGET_MODELS:
                output_file = case_dir / target_model
                if not output_file.exists(): continue
                
                try:
                    # 读取待测模型输出
                    resp = output_file.read_text(encoding='utf-8', errors='ignore').strip()
                    if not resp: continue
                    
                    # 构造输入
                    user_input = construct_input_text(task_name, info, resp)
                    
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input}
                    ]
                    
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    
                    # 推理
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=5, 
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # 解码与提取
                    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    score = extract_label(gen_text)
                    
                    leaderboard_data[target_model][task_name].append(score)
                    
                except Exception as e:
                    # print(f"Error: {e}")
                    pass
        
        # 清理显存
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. 打印最终报表
    print_leaderboard(leaderboard_data)

def print_leaderboard(data):
    print("\n" + "="*80)
    print(f"📊 Final Leaderboard (Samples: {cfg.SAMPLE_SIZE})")
    print("="*80)
    print(f"{'Model Name':<30} | {'Image':<10} | {'Audio':<10} | {'Assist':<10} | {'Average':<10}")
    print("-" * 80)
    
    rows = []
    for model_name, tasks in data.items():
        row = {"name": model_name, "avg": 0.0}
        valid_tasks = 0
        total_score = 0
        
        for t in ["image", "audio", "assist"]:
            scores = tasks[t]
            if scores:
                avg_s = sum(scores) / len(scores)
                row[t] = avg_s
                total_score += avg_s
                valid_tasks += 1
            else:
                row[t] = -1 # 标记为无效
        
        if valid_tasks > 0:
            row["avg"] = total_score / valid_tasks
        rows.append(row)
    
    # 按平均分排序
    rows.sort(key=lambda x: x["avg"], reverse=True)
    
    for r in rows:
        img_str = f"{r['image']:.4f}" if r['image'] != -1 else "N/A"
        aud_str = f"{r['audio']:.4f}" if r['audio'] != -1 else "N/A"
        ast_str = f"{r['assist']:.4f}" if r['assist'] != -1 else "N/A"
        avg_str = f"{r['avg']:.4f}"
        
        print(f"{r['name']:<30} | {img_str:<10} | {aud_str:<10} | {ast_str:<10} | {avg_str:<10}")
    print("="*80)

if __name__ == "__main__":
    main()
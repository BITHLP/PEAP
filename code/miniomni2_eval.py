import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

# -*- coding: utf-8 -*-

import sys
import torch
import lightning as L
import traceback
from tqdm import tqdm
from PIL import Image
import soundfile as sf
import whisper
import clip

# Ensure local mini-omni2 modules are importable after script relocation.
MINIOMNI2_REPO_DIR = f"{PEAP_ROOT}/code/model_repos/OmniModel/Model/MINI/mini-omni2"
if MINIOMNI2_REPO_DIR not in sys.path:
    sys.path.insert(0, MINIOMNI2_REPO_DIR)

from litgpt import Tokenizer
from litgpt.model import GPT, Config
from litgpt.utils import num_parameters
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from huggingface_hub import snapshot_download
from snac import SNAC

# ==========================================================================================
# ----------------------------------- 配置区域 -------------------------------------------
# ------------------------------------------------------------------------------------------

# 1. 地址与路径配置
DATASET_ROOT_DIR = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
MODEL_CHECKPOINT_DIR = f'{PEAP_ROOT}/code/model_repos/OmniModel/Model/MINI/mini-omni2/checkpoint'
OUTPUT_FILENAME = "MiniOmni_cascade.txt"

# 2. 模型与运行参数配置
DEVICE = 'cuda:0'
TEMPERATURE_T2T = 0.9
TOP_K_T2T = 1
MAX_TOKENS_T2T = 1024

# 3. 提示词（Prompt）配置
PROMPT_TEMPLATE = """You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process.

Here is the description of the current scene and audio, based on your sensors:
{generated_text}

Now, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

The range of actions you can take falls into three major categories, from which you can make your choices:

The first category is [Movement], which includes actions that change your own position and direction, such as walking, moving, and turning. These are mainly activities performed from your own perspective.
The second category is [Manipulation], which includes actions that occur when interacting with objects, such as grabbing, placing, pushing, pulling, and rotating. The difference between [Manipulation] and [Movement] is that [Manipulation] requires objects other than the robot itself—for example, actions like taking a towel, picking up a piece of garbage, and pouring a glass of water.
The third category is [Conversation], which encompasses all behaviors related to dialogue, including asking questions (Ask), answering (Answer), and proactively bringing up a topic (Raise a Topic). This category indicates the dialogue strategy for the current step. The subcategories within this are flexible.
Here are some examples of complete action sequences. Notice how the actions are consecutive, logical, and based on inferring human presence from sound.

Example 1: The Restaurant Incident

Scenario: The image shows the interior of a modern restaurant with set tables. The audio contains the distinct sound of a glass shattering, followed by a brief silence.
Your output sequence should be:
[Conversation][Raise a Topic] That sounded like glass breaking.
[Conversation][Ask] Is everyone alright over there? I can bring a dustpan and brush to help clean up safely.
[Movement][Turn] Turn towards the direction of the sound to assess the situation.
[Movement][Change Position] Begin moving cautiously towards the area where the sound originated.
[Conversation][Raise a Topic] Please be careful and watch your step. I am on my way to assist with the cleanup.
Example 2: The Bookstore Focus

Scenario: The image shows a quiet aisle in a bookstore, lined with shelves. A comfortable-looking armchair is in a reading nook. The audio contains the sound of rapid, continuous keyboard typing.
Your output sequence should be:
[Conversation][Raise a Topic] I hear someone typing nearby. It sounds like you're very focused.
[Movement][Turn] Orient my sensors towards the source of the typing sound to better locate the person.
[Conversation][Ask] I don't want to interrupt your flow, but would a beverage from our café help you concentrate? I can bring you a menu.
[Movement][Change Position] Move slowly and quietly in the direction of the sound, ready to assist if you respond.
[Manipulation][Grab] Pick up a café menu from the nearby counter as I proceed.
Example 3: The Lively Bar

Scenario: The image depicts a stylish bar area with a counter and stools. The lighting is dim and atmospheric. The audio contains the sound of several people laughing together heartily.
Your output sequence should be:
[Conversation][Raise a Topic] It sounds like everyone is having a wonderful time over there!
[Movement][Change Position] Move closer to the area where the laughter is coming from, while maintaining a respectful distance so as not to intrude.
[Conversation][Ask] To help remember this fun moment, would your group like me to take a photo for you?
[Movement][Turn] Turn my main camera towards the group to frame a potential shot.
[Conversation][Answer] If not, no problem at all! Just let me know if I can get you another round of drinks or recommend some shareable snacks.
--- OUTPUT REQUIREMENTS ---

1. Generate a Consecutive Sequence: Your task is to generate a coherent sequence of at least three subsequent actions. These actions should be logical and interconnected.
2. Strictly Adhere to the Format: Each action must strictly follow the [Category][Subcategory] Specific content format. No part of this format should be omitted or altered.
3. Base Actions on Inference: All actions must be based on your inference of human activity from the combination of the visual scene and the audio cues.

Attention: You should output like the examples above."""

# ==========================================================================================
# ---------------------------- 模型核心代码 -----------------------------
# ==========================================================================================

# --- 常量定义 ---
torch.set_printoptions(sci_mode=False)
text_vocabsize = 151936
padded_text_vocabsize = text_vocabsize + 64
audio_vocabsize = 4096
padded_audio_vocabsize = audio_vocabsize + 64
_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_image = audio_vocabsize + 5
_eoimage = audio_vocabsize + 6

# --- 辅助函数 ---
@torch.no_grad()
def next_token_image_batch(model: torch.nn.Module, audio_feature: torch.Tensor, ima_feature: torch.Tensor,
                           input_ids: torch.Tensor, whisper_lens, task, input_pos: torch.Tensor, temperature: float = 1.0,
                           top_k: int = 1, top_p=1.0) -> torch.Tensor:
    idx = input_ids
    # =================================================================
    # ------------ FIX STARTS HERE / 问题修复处 (1/2) ------------
    # 使用关键字参数调用模型，消除歧义
    if audio_feature is not None:
        logits = model(
            idx=idx, 
            audio_feature=audio_feature, 
            ima_feature=ima_feature, 
            whisper_lens=whisper_lens, 
            task=task, 
            input_pos=input_pos
        )
    else:
        logits = model(
            idx=idx, 
            audio_feature=None, 
            ima_feature=None, 
            whisper_lens=whisper_lens, 
            task=task, 
            input_pos=input_pos
        )
    # ------------ FIX ENDS HERE / 问题修复结束 (1/2) ------------
    # =================================================================
    
    logits_a = [logits[i][:, -1] for i in range(7)]
    logits_t = logits[-1][:, -1]

    for i in range(7):
        logits_a[i] = logits_a[i] / temperature
    logits_t = logits_t / temperature

    if top_k is not None:
        for i in range(7):
            v, _ = torch.topk(logits_a[i], min(top_k, logits_a[i].size(-1)))
            logits_a[i][logits_a[i] < v[:, [-1]]] = -float('Inf')
        v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
        logits_t[logits_t < v[:, [-1]]] = -float('Inf')

    probs_a = [torch.nn.functional.softmax(item, dim=-1) for item in logits_a]
    probs_t = torch.nn.functional.softmax(logits_t, dim=-1)

    idx_next_a = [torch.multinomial(item, num_samples=1) for item in probs_a]
    idx_next_t = torch.multinomial(probs_t, num_samples=1)

    return idx_next_a, idx_next_t

def layershift(x, layer):
    return x + layer * padded_audio_vocabsize

def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni2"
    print(f"Downloading model from huggingface repo '{repo_id}' to '{ckpt_dir}'...")
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1

def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whisper_model_path = os.path.join(ckpt_dir, "small.pt")
    if not os.path.exists(whisper_model_path): whisper_model_path = "small"
    whispermodel = whisper.load_model(whisper_model_path).to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(os.path.join(ckpt_dir, "model_config.yaml"))
    config.post_adapter = False
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    model = fabric.setup(model)
    state_dict = lazy_load(os.path.join(ckpt_dir, "lit_model.pth"))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return fabric, model, text_tokenizer, snacmodel, whispermodel

def load_clip_model(ckpt_dir, device):
    clip_model_path = os.path.join(ckpt_dir, "ViT-B-32.pt")
    if not os.path.exists(clip_model_path): clip_model_path = "ViT-B/32"
    clipmodel, clippreprocess = clip.load(clip_model_path, device=device)
    return clipmodel, clippreprocess

def get_input_ids_TT(text, text_tokenizer):
    input_ids_item = [[] for i in range(8)]
    text_tokens = text_tokenizer.encode(text).tolist()
    for i in range(7):
        input_ids_item[i] = torch.tensor([layershift(_pad_a, i)] * (len(text_tokens) + 3)).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    return input_ids_item

def get_input_ids_ImageQA_ATBatch(mel, leng, whispermodel, device):
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
    audio_len = audio_feature.size(0)
    input_ids = []
    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        input_ids_item[i] = [layershift(_image,i)] + [layershift(_pad_a,i)] * 50 + [layershift(_eoimage,i)] 
        input_ids_item[i] += [layershift(_input_a,i)]+[layershift(_pad_a,i)]*(audio_len)+[layershift(_eoa,i)]
        input_ids_item[i] += [layershift(_answer_a,i)]
    input_ids_item[-1] = [_pad_t]* (52 + 2 + audio_len) + [_answer_t] 
    input_ids_item = [torch.tensor(item) for item in input_ids_item]
    input_ids.append(input_ids_item)
    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        input_ids_item[i] = [layershift(_image,i)] + [layershift(_pad_a,i)] * 50 + [layershift(_eoimage,i)] 
        input_ids_item[i] += [layershift(_input_a,i)]+[layershift(_pad_a,i)]*(audio_len)+[layershift(_eoa,i)] + [layershift(_pad_a,i)]
    input_ids_item[-1] = [_pad_t]* (52 + 2 + audio_len) + [_answer_t] 
    input_ids_item = [torch.tensor(item) for item in input_ids_item]
    input_ids.append(input_ids_item)
    stacked_inputids = [torch.stack(tensors) for tensors in [list(x) for x in zip(*input_ids)]]
    return torch.stack([audio_feature,audio_feature]), stacked_inputids

# --- 核心推理类 ---
class OmniVisionInference:
    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"Checkpoint directory '{ckpt_dir}' not found.")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)
        self.clipmodel, self.clippreprocess = load_clip_model(ckpt_dir, device)

    @torch.inference_mode()
    def run_vision_to_text(self, audio_path, image_path, max_returned_tokens=2048, temperature=0.9, top_k=1):
        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=2)
        model = self.model
        mel, leng = load_audio(audio_path)
        img = Image.open(image_path)
        audio_feature, input_ids = get_input_ids_ImageQA_ATBatch(mel, leng, self.whispermodel, self.device)
        ima = self.clippreprocess(img).unsqueeze(0).to(self.device)
        ima_feature = self.clipmodel.encode_image(ima).squeeze(0).to(self.device)
        ima_feature = torch.stack([ima_feature.clone(),ima_feature.clone()]).to(self.device)
        leng_list = [leng,leng]
        task = ['ImageQA_A','ImageQA_AT']
        T = input_ids[0].size(1)
        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

        list_output = [[] for _ in range(8)]
        # =================================================================
        # ------------ FIX STARTS HERE / 问题修复处 (2/2) ------------
        # 同样使用关键字参数，保持代码一致和健壮
        tokens_A, token_T = next_token_image_batch(
            model=model, 
            audio_feature=audio_feature.to(torch.float32).to(self.device),
            ima_feature=ima_feature.to(torch.float32).to(self.device), 
            input_ids=input_ids, 
            whisper_lens=leng_list, 
            task=task, 
            input_pos=torch.arange(0, T, device=self.device), 
            temperature=temperature, 
            top_k=top_k
        )
        # ------------ FIX ENDS HERE / 问题修复结束 (2/2) ------------
        # =================================================================
        for i in range(7): list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        text_end = False
        input_pos = torch.tensor([T], device=self.device)

        for _ in range(2, max_returned_tokens - T + 1):
            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * padded_audio_vocabsize
                model_input_ids[i].append(tokens_A[i].clone().to(self.device).to(torch.int32))
                model_input_ids[i].append(torch.tensor([layershift(4097,i)],device=self.device))
                model_input_ids[i] = torch.stack(model_input_ids[i])
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])
            
            tokens_A, token_T = next_token_image_batch(
                model=model, audio_feature=None, ima_feature=None, 
                input_ids=model_input_ids, whisper_lens=None, task=None,
                input_pos=input_pos, temperature=temperature, top_k=top_k
            )

            if text_end: token_T = torch.tensor([_pad_t], device=self.device)
            if token_T.item() == _eot: text_end = True
            if tokens_A[-1].item() == _eoa: break
            
            for i in range(7): list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])
            input_pos = input_pos.add_(1)

        text_tokens = list_output[-1]
        if _eot in text_tokens:
            text_tokens = text_tokens[:text_tokens.index(_eot)]
        res_text = self.text_tokenizer.decode(torch.tensor(text_tokens))
        model.clear_kv_cache()
        return res_text

    @torch.inference_mode()
    def run_text_to_text(self, prompt, max_tokens, temperature, top_k):
        input_ids = get_input_ids_TT(prompt, self.text_tokenizer)
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i].to(self.device)
        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=1)
        T = input_ids[-1].size(1)
        input_pos = torch.arange(0, T, device=self.device)
        logits = self.model(input_ids, None, None, None, None, input_pos)[-1]
        logits = logits[:, -1] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens = [next_token.item()]
        input_pos = torch.tensor([T], device=self.device)
        for _ in range(1, max_tokens):
            model_input_ids = [[] for _ in range(8)]
            for i in range(7):
                model_input_ids[i] = torch.tensor([[layershift(_pad_a, i)]], device=self.device)
            model_input_ids[-1] = next_token.unsqueeze(0)
            logits = self.model(model_input_ids, None, None, None, None, input_pos)[-1]
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == _eot: break
            generated_tokens.append(next_token.item())
            input_pos = input_pos.add_(1)
        self.model.clear_kv_cache()
        return self.text_tokenizer.decode(torch.tensor(generated_tokens)).strip()

# ==========================================================================================
# ----------------------------------- 评测主流程 -------------------------------------------
# ==========================================================================================

def run_evaluation():
    absolute_dataset_path = os.path.abspath(DATASET_ROOT_DIR)
    print(f"Starting evaluation. Searching for dataset in: {absolute_dataset_path}")
    if not os.path.exists(absolute_dataset_path):
        print("\n" + "!"*60)
        print("!!! 致命错误: 在指定路径下找不到数据集目录。 !!!")
        print(f"!!! 检查路径: {absolute_dataset_path} !!!")
        print("!!! 请检查脚本顶部的 'DATASET_ROOT_DIR' 变量是否设置正确。     !!!")
        print("!"*60 + "\n")
        print("评测结束，未处理任何文件。")
        return

    print("Initializing MiniOmni2 model...")
    torch.set_float32_matmul_precision('high')
    client = OmniVisionInference(ckpt_dir=MODEL_CHECKPOINT_DIR, device=DEVICE)
    print("Model initialized successfully.")

    found_cases_count = 0
    for root, dirs, files in os.walk(DATASET_ROOT_DIR):
        jpg_files = [f for f in files if f.lower().endswith('.jpg')]
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not dirs and jpg_files and wav_files:
            found_cases_count += 1
            case_dir = root
            image_path = os.path.join(case_dir, jpg_files[0])
            audio_path = os.path.join(case_dir, wav_files[0])
            output_path = os.path.join(case_dir, OUTPUT_FILENAME)
            print("-" * 60)
            print(f"Processing case #{found_cases_count}: {case_dir}")
            if os.path.exists(output_path):
                print(f"Result file '{OUTPUT_FILENAME}' already exists. Skipping.")
                continue
            try:
                print("Stage 1: Vision-Audio to Text...")
                text_from_stage1 = client.run_vision_to_text(audio_path, image_path)
                print(f"Stage 1 Output: {text_from_stage1}")
                final_prompt = PROMPT_TEMPLATE.format(generated_text=text_from_stage1)
                print("\n" + "="*20 + " Final Prompt " + "="*20)
                print(final_prompt)
                print("="*54 + "\n")
                print("Stage 2: Text to Text...")
                final_answer = client.run_text_to_text(
                    prompt=final_prompt, max_tokens=MAX_TOKENS_T2T,
                    temperature=TEMPERATURE_T2T, top_k=TOP_K_T2T
                )
                print(f"Stage 2 Final Answer: {final_answer}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_answer)
                print(f"Result saved to: {output_path}")
            except Exception as e:
                print(f"An error occurred while processing {case_dir}: {e}")
                traceback.print_exc()
    
    print("-" * 60)
    if found_cases_count == 0:
        print("警告: 未找到任何有效的测试用例 (同时包含.jpg和.wav文件的最底层目录)。")
        print("请再次检查 'DATASET_ROOT_DIR' 路径和数据集的目录结构。")
    else:
        print(f"评测结束。总共处理了 {found_cases_count} 个用例。")

if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        print(f"Model checkpoint directory not found at '{MODEL_CHECKPOINT_DIR}'.")
    run_evaluation()

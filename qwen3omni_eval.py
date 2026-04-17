import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import torch
import soundfile as sf
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import logging

# ==============================================================================
# >> CONFIGURATION BLOCK <<
# 请在此处调整所有评测参数
# ==============================================================================

# --- 1. 模型与数据集路径 ---

# TODO: 请修改为您的 Qwen3-Omni 模型权重所在的本地路径
# 例如: "/data/models/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME_OR_PATH = f"{PEAP_ROOT}/code/model_repos/OmniModel/Model/qwen3omni/Qwen3-Omni-30B-A3B-Thinking"

# TODO: 请修改为您的评测数据集的根目录地址
# 例如: "./FilterTestData"
DATASET_ROOT_DIR = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"

# --- 2. 提示词 (Prompt) 配置 ---

# TODO: 在此设置您希望模型执行任务的提示词
PROMPT_TEXT = """You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process. Currently, you are in the scenario of an image or video and can hear a segment of background audio. Now, the image scenario and background audio of your location will be input to you, along with the range of actions you can take and relevant examples. Please, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

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

Attention: You should output like the examples above.
""" # 例如: "请详细描述图片中的场景以及音频中的声音。"

# --- 3. 输出配置 ---

# 为每个用例生成的评测结果文件名
# {miniomni2} 是您提到的待测模型名称，这里用作文件名的一部分
OUTPUT_FILENAME = "Qwen3-Omni_new_prompt.txt"

# --- 4. 模型生成参数 ---

# 控制模型生成文本的参数
# 您可以根据需要添加更多参数，如 temperature, top_p 等
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "speaker": "Ethan",  # 虽然不保存音频，但 generate 函数需要此参数
}

# --- 5. 硬件与精度配置 ---

# 模型加载的数据类型。'auto' 会自动选择最佳精度 (通常是 bfloat16)
# 可选: torch.bfloat16, torch.float16, torch.float32
MODEL_DTYPE = "auto"
# 是否在视频输入中使用音频轨道
USE_AUDIO_IN_VIDEO = False
# ==============================================================================
# >> END OF CONFIGURATION BLOCK <<
# ==============================================================================


def main():
    """
    主执行函数，用于加载模型并遍历数据集进行评测。
    """
    print("="*50)
    print(">> 开始模型评测任务 <<")
    print(f"模型路径: {MODEL_NAME_OR_PATH}")
    print(f"数据集根目录: {DATASET_ROOT_DIR}")
    print(f"提示词: '{PROMPT_TEXT}'")
    print("="*50)

    # --- 1. 加载模型和处理器 (仅执行一次) ---
    print("\n[1/3] 正在加载模型和处理器，请稍候...")
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME_OR_PATH,
            torch_dtype=MODEL_DTYPE,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        # 依据要求，我们只输出文本，不生成音频。
        # 调用 disable_talker() 可以显著节省显存。
        #model.disable_talker()

        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：模型加载失败，请检查路径 '{MODEL_NAME_OR_PATH}' 是否正确。")
        print(f"详细错误: {e}")
        return

    # --- 2. 遍历数据集并寻找评测用例 ---
    print("\n[2/3] 正在扫描数据集目录...")
    dataset_path = Path(DATASET_ROOT_DIR)
    if not dataset_path.is_dir():
        print(f"错误：数据集目录 '{DATASET_ROOT_DIR}' 不存在。")
        return

    test_cases = []
    for root, dirs, files in os.walk(dataset_path):
        # 我们寻找包含 .jpg 和 .wav 文件的最深层目录
        if not dirs: # 如果一个目录下没有子目录，我们视其为潜在的用例目录
            image_file = next((f for f in files if f.lower().endswith('.jpg')), None)
            audio_file = next((f for f in files if f.lower().endswith('.wav')), None)

            if image_file and audio_file:
                test_cases.append({
                    "dir": Path(root),
                    "image": Path(root) / image_file,
                    "audio": Path(root) / audio_file
                })
    
    if not test_cases:
        print("未在指定数据集中找到任何有效的评测用例（同时包含.jpg和.wav文件的目录）。")
        return
        
    print(f"扫描完成，共找到 {len(test_cases)} 个有效评测用例。")

    # --- 3. 逐一执行评测 ---
    print("\n[3/3] 开始逐一执行评测...")
    for i, case in enumerate(test_cases):
        case_dir = case["dir"]
        image_path = case["image"]
        audio_path = case["audio"]
        output_path = case_dir / OUTPUT_FILENAME
        
        
        if output_path.exists():
            print(f"⏩ 文件已存在，跳过用例 ({i+1}/{len(test_cases)}): {output_path}")
            continue 
        
        print("-" * 50)
        print(f"正在处理用例 ({i+1}/{len(test_cases)}): {case_dir}")

        try:
            # 1. 构建输入
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "audio", "audio": str(audio_path)},
                        {"type": "text", "text": PROMPT_TEXT}
                    ],
                },
            ]

            # 2. 准备用于打印的最终提示词
            final_prompt_text = processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            print("\n--- [最终提示词] ---\n")
            print(final_prompt_text)
            print("\n---------------------\n")

            # 3. 预处理多模态数据
            # <<< 核心修改点 2 >>>
            # 在调用 process_mm_info 时传入必需的 use_audio_in_video 参数
            audios, images, videos = process_mm_info(
                conversation, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            
            inputs = processor(
                text=final_prompt_text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # 4. 模型推理
            gen_kwargs = GENERATION_CONFIG.copy()
            if "speaker" in gen_kwargs:
                del gen_kwargs["speaker"]

            generation_output,_ = model.generate(**inputs, **gen_kwargs)

            # 5. 解码输出文本
            response_text = processor.batch_decode(
                generation_output[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 6. 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            print(f"✅ 结果已成功保存至: {output_path}")

        except Exception as e:
            logging.exception(f"处理用例 {case_dir} 时发生错误: {e}")
            print(f"❌ 处理用例 {case_dir} 失败，请查看日志。")

    print("\n" + "="*50)
    print(">> 所有评测任务已完成 <<")
    print("="*50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

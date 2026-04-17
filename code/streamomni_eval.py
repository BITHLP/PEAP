import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import sys
import torch
import argparse
from PIL import Image
from io import BytesIO
import requests
import re

# Ensure Stream-Omni local package imports resolve after relocation.
STREAM_OMNI_REPO_DIR = f"{PEAP_ROOT}/code/model_repos/OmniModel/Model/Stream-Omni"
if STREAM_OMNI_REPO_DIR not in sys.path:
    sys.path.insert(0, STREAM_OMNI_REPO_DIR)

# 导入StreamOmni相关模块
from stream_omni.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from stream_omni.conversation import conv_templates, SeparatorStyle
from stream_omni.model.builder import load_pretrained_model
from stream_omni.utils import disable_torch_init
from stream_omni.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# =================================================================================
# === 1. 可配置参数、地址和Prompt ===
# (请根据您的实际情况修改以下内容)
# =================================================================================

# 数据集根目录
# 例如: "FilterTestData"
DATASET_ROOT_DIR = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"

# 模型路径和名称 (请替换为您的模型实际路径)
MODEL_PATH = f"{PEAP_ROOT}/code/model_repos/OmniModel/Model/Stream-Omni/weights/stream-omni-8b"
MODEL_BASE = None
MODEL_NAME = "stream-omni-8b" # "get_model_name_from_path" 会自动从MODEL_PATH获取, 但这里可以手动指定

# Prompt模板 (音频同名txt的内容会自动拼接到此模板前面)
# 如果只需要音频txt内容作为prompt，此处保持为空字符串""即可
PROMPT = """You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process. Currently, you are in the scenario of an image or video and can hear a segment of background audio. Now, the image scenario and background audio of your location will be input to you, along with the range of actions you can take and relevant examples. Please, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

The range of actions you can take falls into three major categories, from which you can make your choices:
• The first category is [Movement], which includes actions that change your own position and direction, such as walking, moving, and turning. These are mainly activities performed from your own perspective.
• The second category is [Manipulation], which includes actions that occur when interacting with objects, such as grabbing, placing, pushing, pulling, and rotating. The difference between [Manipulation] and [Movement] is that [Manipulation] requires objects other than the robot itself—for example, actions like taking a towel, picking up a piece of garbage, and pouring a glass of water.
• The third category is [Conversation], which encompasses all behaviors related to dialogue, including asking questions (Ask), answering (Answer), and proactively bringing up a topic (Raise a Topic). This category indicates the dialogue strategy for the current step.

When outputting your next action or speech, first output the label of the category and subcategory that the action or speech belongs to, then output the specific content. When outputting the label, present the major category label first, followed by the subcategory label. For example:
[Conversation][Raise a Topic] That drilling sound seems quite close - perhaps they're doing renovations in the adjacent unit?
[Movement][Turn] Rotate toward the direction of the sound to better assess its source.
[Manipulation][Grab] Pick up the electric kettle from the countertop.
[Conversation][Ask] Would you like me to prepare some tea or coffee while we wait? The hot water might help mask the noise somewhat.
[Movement][Change Position] Move toward the sink to fill the kettle with water.

**Attention:You must output like the example above!!! You can output one or more as needed!!!**
"""

# 模型生成参数
TEMPERATURE = 0.0
TOP_P = None
NUM_BEAMS = 1
MAX_NEW_TOKENS = 4096

# 对话模式 (设置为None可让代码根据模型名称自动推断)
# 可选值例如: "chatml_direct", "llava_v1", "mistral_instruct" 等
CONV_MODE = "chatml_direct"

# 输出结果的文件名
OUTPUT_FILENAME = "StreamOmni.txt"


# =================================================================================
# === 核心功能代码 (通常无需修改) ===
# =================================================================================

def load_image(image_file):
    """加载单张图片"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def eval_single_case(model, tokenizer, image_processor, image_path, final_prompt):
    """
    对单个用例进行评测的核心函数。

    Args:
        model: 加载好的模型。
        tokenizer: 加载好的分词器。
        image_processor: 加载好的图片处理器。
        image_path (str): 单张图片的路径。
        final_prompt (str): 拼接完成的最终提示词。

    Returns:
        str: 模型生成的文本结果。
    """
    # 准备图片输入
    image = load_image(image_path)
    image_sizes = [image.size]
    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + final_prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + final_prompt

    # 获取对话模板和最终的prompt
    # 自动推断对话模式
    if CONV_MODE is None:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower() or "stream-omni-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
    else:
        conv_mode = CONV_MODE

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_for_model = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # 模型推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            inference_type="speech_to_text", # 根据官方示例保留此参数
        )

    # 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def main():
    """
    主函数，负责遍历数据集、加载模型并调用评测函数。
    """
    print(">>> 正在初始化并加载模型，请稍候...")
    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH) if not MODEL_NAME else MODEL_NAME

    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA设备，请检查您的环境配置。")
        return

    # 在调用时添加 device_map 和 torch_dtype
    # 这是解决 "meta tensor" 问题的关键
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=MODEL_BASE,
        model_name=model_name,
        device_map="auto",          # <--- 添加这一行
        torch_dtype=torch.float16   # <--- 添加这一行
    )
    
    print(">>> 模型加载完成。")

    # 遍历数据集根目录
    for root, dirs, files in os.walk(DATASET_ROOT_DIR):
        image_file = None
        audio_file = None

        # 在当前目录中查找.jpg和.wav文件
        for file in files:
            if file.lower().endswith('.jpg'):
                image_file = file
            elif file.lower().endswith('.wav'):
                audio_file = file

        # 如果找到了.jpg和.wav文件，则认为这是一个有效的用例目录
        if image_file and audio_file:
            case_name = os.path.relpath(root, DATASET_ROOT_DIR)
            print(f"\n--- 正在处理用例: {case_name} ---")

            output_path = os.path.join(root, OUTPUT_FILENAME)

            # 4. 当检测到当前目录已经存在StreamOmni.txt时，跳过这一条
            if os.path.exists(output_path):
                print(f"结果文件 '{output_path}' 已存在，跳过此用例。")
                continue

            image_path = os.path.join(root, image_file)
            audio_basename = os.path.splitext(audio_file)[0]
            prompt_txt_path = os.path.join(root, audio_basename + '.txt')

            # 1. 输入图片，检测音频文件同名txt拼接在提示词前面
            prompt_from_file = ""
            if os.path.exists(prompt_txt_path):
                with open(prompt_txt_path, 'r', encoding='utf-8') as f:
                    prompt_from_file = f.read().strip()
            else:
                print(f"警告：未找到对应的prompt文件 '{prompt_txt_path}'，将使用空prompt。")

            final_prompt = prompt_from_file + PROMPT

            # 3. 跑每一个用例的时候都在命令行窗口中输出拼接完成的的最终提示词
            print(f"最终提示词 (Final Prompt): {final_prompt}")

            try:
                # 执行评测
                result_text = eval_single_case(
                    model, tokenizer, image_processor, image_path, final_prompt
                )

                # 2. 输出文本保存在数据集对应用例的目录下
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result_text)

                print(f"结果已成功保存到: {output_path}")
                print(f"模型输出: {result_text[:200]}...") # 打印部分输出预览

            except Exception as e:
                print(f"处理用例 {case_name} 时发生错误: {e}")

    print("\n--- 所有用例处理完毕 ---")




if __name__ == "__main__":
    main()


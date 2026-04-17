import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

# -*- coding: utf-8 -*-

import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import librosa
import glob

# #############################################################################
# 1. 可更改参数配置
# (将所有可更改参数、地址和全部prompt（先置空）全部内置在代码中)
# #############################################################################

# 数据集总目录地址
DATASET_ROOT_DIR = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData" 

# 模型和分词器路径
MODEL_PATH = f'{PEAP_ROOT}/code/model_repos/OmniModel/Model/MINICPM/MiniCPM-o-2_6'

# 推理相关参数
CHAT_PARAMS = {
    'sampling': True,
    'temperature': 0.7,
    'max_new_tokens': 2048,
}

# 系统提示词语言 ('en' 或 'zh')
SYSTEM_PROMPT_LANGUAGE = 'en'

# 用户输入的总提示词 (可根据评测需求修改, 当前为空)
# 例如: "请描述这张图片和这段音频的内容。"
USER_PROMPT = """You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process. Currently, you are in the scenario of an image or video and can hear a segment of background audio. Now, the image scenario and background audio of your location will be input to you, along with the range of actions you can take and relevant examples. Please, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

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
"""

# 输出结果的文件名
OUTPUT_FILENAME = "minicpm_new_prompt.txt"

# #############################################################################
# 2. 模型初始化
# (此部分代码源于官方示例，用于加载模型)
# #############################################################################

print("开始初始化模型...")
try:
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation='sdpa',  # or 'flash_attention_2'
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 初始化TTS处理器 (即使不生成音频，Omni模式也需要)
    model.init_tts()
    print("模型初始化完成。")
except Exception as e:
    print(f"模型初始化失败，请检查环境和模型路径配置。错误: {e}")
    exit()

# #############################################################################
# 3. 评测执行
# (遍历数据集，执行推理并保存结果)
# #############################################################################

# 搜索所有用例目录
# 我们假设每个包含.jpg文件的最底层目录是一个用例
search_pattern = os.path.join(DATASET_ROOT_DIR, '**', '*.jpg')
all_image_files = glob.glob(search_pattern, recursive=True)

if not all_image_files:
    print(f"在目录 '{DATASET_ROOT_DIR}' 下未找到任何 .jpg 文件。请检查数据集路径。")

for image_path in all_image_files:
    case_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 寻找对应的.wav文件
    # 假设.wav文件和.jpg文件在同一目录下，且可能同名或有唯一的一个
    wav_files = glob.glob(os.path.join(case_dir, '*.wav'))
    
    if not wav_files:
        print(f"警告: 在目录 {case_dir} 中未找到 .wav 文件，跳过此用例。")
        continue
    
    audio_path = wav_files[0] # 使用找到的第一个.wav文件
    
    print("-" * 50)
    print(f"正在处理用例: {case_dir}")
    print(f"  - 图片: {image_path}")
    print(f"  - 音频: {audio_path}")

    try:
        # 1. 准备输入数据
        image = Image.open(image_path).convert('RGB')
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 2. 构建输入消息 (msgs)
        # 系统消息
        sys_msg = model.get_sys_prompt(mode='omni', language=SYSTEM_PROMPT_LANGUAGE)
        
        # 用户消息，包含图片、音频和文本提示
        # 评测方式: 输入图片音频和提示词
        contents = [image, audio_np, USER_PROMPT]
        
        user_msg = {"role": "user", "content": contents}
        msgs = [sys_msg, user_msg]
        
        # 3. 打印最终提示词到命令行窗口
        # 要求: 跑每一个用例的时候都在命令行窗口中输出拼接完成的的最终提示词
        final_prompt_for_display = f'图片: [已加载], 音频: [已加载], 文本提示: "{USER_PROMPT}"'
        print(f"\n拼接完成的最终提示词:\n{final_prompt_for_display}\n")

        # 4. 执行模型推理
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            omni_input=True,         # Omni模式必须为True
            use_tts_template=False,  # 我们只进行文本生成，不遵循TTS模板
            generate_audio=False,    # 不生成音频输出
            **CHAT_PARAMS,
            return_dict=True 
        )
        
        # 提取生成的文本结果
        generated_text = res.get('text', '错误：未能从模型输出中提取文本。')
        print(f"模型输出: {generated_text}")

        # 5. 将结果保存为txt到用例自身的目录下
        # 要求: 输出文本保存在数据集对应用例的目录下，要且仅要这一个结果,结果文件名为minicpm.txt
        output_txt_path = os.path.join(case_dir, OUTPUT_FILENAME)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        
        print(f"结果已成功保存到: {output_txt_path}")

    except Exception as e:
        print(f"处理用例 {case_dir} 时发生错误: {e}")

print("-" * 50)
print("所有用例处理完毕。")


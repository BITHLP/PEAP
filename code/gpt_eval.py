import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import glob
import base64
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# 1. 全局配置 (Config)
# ==========================================
class EvalConfig:
    # --- API 密钥与地址 ---
    API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 从环境变量读取
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    
    # --- 模型选择 ---
    # 使用 OpenAI 官方最强多模态模型
    MODEL_NAME = os.environ.get("OPENAI_MODEL", "o4-mini")
    
    # --- 数据集路径 ---
    DATASET_ROOT = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
    
    # --- 输出文件 ---
    OUTPUT_FILENAME = "O4mini.txt"
    
    # --- 提示词 (Prompt) ---
    USER_INSTRUCTION = """You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process. Currently, you are in the scenario of an image or video and can hear a segment of background audio. Now, the image scenario and background audio of your location will be input to you, along with the range of actions you can take and relevant examples. Please, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

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
    
    # --- 运行参数 ---
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    TEMPERATURE = 0.5  # 控制生成的随机性

cfg = EvalConfig()

# ==========================================
# 2. 核心工具函数
# ==========================================

def get_client():
    """初始化 OpenAI 客户端"""
    return OpenAI(
        api_key=cfg.API_KEY,
        base_url=cfg.BASE_URL
    )

def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(client, image_path, audio_path):
    """
    构造请求并调用 GPT-4o API (Standard Chat Completion)
    """
    # 1. 准备 Prompt 文本
    prompt_text = cfg.USER_INSTRUCTION
    
    # 注入音频提示
    # 注意：标准 chat.completions 接口不支持直接上传音频文件，
    # 因此这里将音频信息作为文本提示的一部分传给模型。
    if audio_path:
        audio_name = os.path.basename(audio_path)
        prompt_text += f"\n\n[System Context Note: An audio file named '{audio_name}' is detected in this scene. Please consider the auditory context implied by this file.]"

    # 2. 构造多模态内容 (Content List)
    content_list = []
    
    # 添加文本
    content_list.append({
        "type": "text", 
        "text": prompt_text
    })

    # 添加图片
    if image_path:
        base64_image = encode_image(image_path)
        if base64_image:
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    # 3. 构造 Messages
    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    # 4. 调用 API
    for attempt in range(cfg.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=cfg.MODEL_NAME,
                messages=messages,
                temperature=cfg.TEMPERATURE,
                max_tokens=4096
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ Attempt {attempt + 1} failed: {error_msg}")
            
            # 如果出现 400 错误（例如图片太大或格式不支持），尝试降级为纯文本
            if "400" in error_msg and image_path:
                print("   Retrying with text-only prompt...")
                try:
                    # 降级：仅发送文本
                    response = client.chat.completions.create(
                        model=cfg.MODEL_NAME,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=cfg.TEMPERATURE
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    print(f"   Text-only fallback failed: {e2}")

            if attempt == cfg.MAX_RETRIES - 1:
                return f"[Error] API Call Failed: {str(e)}"
            
            time.sleep(cfg.RETRY_DELAY)

# ==========================================
# 3. 主流程逻辑
# ==========================================

def find_media_files(case_dir):
    """在用例文件夹中寻找 jpg 和 wav"""
    case_path = Path(case_dir)
    jpg_files = list(case_path.glob("*.jpg"))
    wav_files = list(case_path.glob("*.wav"))
    
    img_path = str(jpg_files[0]) if jpg_files else None
    aud_path = str(wav_files[0]) if wav_files else None
    
    return img_path, aud_path

def main():
    if "sk-" not in cfg.API_KEY:
        print("❌ 请先在 Config 中填入有效的 Key")
        return

    print(f"🚀 初始化客户端: {cfg.MODEL_NAME}")
    try:
        client = get_client()
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return

    root = Path(cfg.DATASET_ROOT)
    case_dirs = []
    
    if not root.exists():
        print(f"❌ 数据集路径不存在: {cfg.DATASET_ROOT}")
        return

    print("🔍 正在扫描数据集目录...")
    # 遍历三级目录结构: Root/Category/Sub/CaseID
    for cat_dir in root.iterdir():
        if cat_dir.is_dir():
            for sub_dir in cat_dir.iterdir():
                if sub_dir.is_dir():
                    for case_dir in sub_dir.iterdir():
                        if case_dir.is_dir():
                            case_dirs.append(case_dir)
    
    print(f"📂 找到 {len(case_dirs)} 个待测用例")
    
    # 使用 tqdm 显示进度
    for case_dir in tqdm(case_dirs, desc="Evaluating"):
        output_path = case_dir / cfg.OUTPUT_FILENAME
        
        # 断点续传
        if output_path.exists():
            continue
            
        img_path, aud_path = find_media_files(case_dir)
        
        # 如果没有媒体文件，跳过
        if not img_path and not aud_path:
            continue
            
        # 生成回复
        response_text = generate_response(client, img_path, aud_path)
        
        # 保存结果
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(str(response_text))
        except Exception as e:
            print(f"❌ 写入文件失败 {output_path}: {e}")

    print("\n✅ 所有评测完成！")

if __name__ == "__main__":
    main()
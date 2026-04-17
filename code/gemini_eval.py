import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import time
import glob
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types

# ==========================================
# 全局配置 (Config)
# ==========================================
class EvalConfig:
    # --- API 设置 ---
    # 替换为您的 DMXAPI 或 Google API Key
    API_KEY = os.environ.get("GEMINI_API_KEY", "") 
    # DMXAPI 基础 URL
    BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://www.dmxapi.cn")
    
    # 使用 Gemini 3 Pro 预览版
    MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3-pro-preview")
    
    # --- 路径设置 ---
    # 指向采样后的数据集目录
    DATASET_ROOT = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
    
    # 输出文件名
    OUTPUT_FILENAME = "Gemini3pro.txt"
    
    # --- 提示词 (Prompt) ---
    # 指南建议：输入提示应简洁明了，直接、清晰
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
    
    # --- 运行设置 ---
    MAX_RETRIES = 3
    RETRY_DELAY = 2 

cfg = EvalConfig()

# ==========================================
# 核心逻辑
# ==========================================

def get_client():
    """初始化 Gemini 3 客户端"""
    return genai.Client(
        api_key=cfg.API_KEY,
        http_options={'base_url': cfg.BASE_URL}
    )

def read_file_bytes(path):
    """读取文件二进制数据"""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

def generate_response(client, image_path, audio_path):
    """
    构造请求并调用 Gemini 3 API
    """
    # 准备内容部分 (Parts)
    parts = []
    
    # 1. 添加文本指令
    parts.append(types.Part(text=cfg.USER_INSTRUCTION))
    
    # 2. 添加图片 (如果存在)
    if image_path:
        image_data = read_file_bytes(image_path)
        if image_data:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg", # 假设图片为jpg，如果是png请修改
                        data=image_data
                    ),
                    # 指南推荐：大多数图片分析任务使用 high 获得最佳质量
                    media_resolution={"level": "media_resolution_high"} 
                )
            )

    # 3. 添加音频 (如果存在)
    if audio_path:
        audio_data = read_file_bytes(audio_path)
        if audio_data:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="audio/wav", 
                        data=audio_data
                    ),
                    # 对于音频/视频，通常 low/medium 足够，这里用 medium 平衡
                    media_resolution={"level": "media_resolution_medium"}
                )
            )

    # 构造请求配置
    # 指南强烈建议：Gemini 3 的 Temperature 保持默认 (1.0)，不要调低，否则可能影响推理
    generate_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="high") # 显式开启高推理模式
    )

    # 执行重试逻辑
    for attempt in range(cfg.MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=cfg.MODEL_NAME,
                contents=[types.Content(parts=parts)],
                config=generate_config
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ Attempt {attempt + 1} failed: {error_msg}")
            
            # 400 错误通常是参数问题，重试无用，直接抛出
            if "400" in error_msg:
                return f"[Error 400] Bad Request: {error_msg}"
            
            if attempt == cfg.MAX_RETRIES - 1:
                return f"[Error] API Call Failed after retries: {error_msg}"
            
            time.sleep(cfg.RETRY_DELAY)

# ==========================================
# 主流程
# ==========================================

def find_media_files(case_dir):
    """在用例文件夹中寻找 jpg 和 wav"""
    # 转换路径对象
    case_path = Path(case_dir)
    
    jpg_files = list(case_path.glob("*.jpg"))
    wav_files = list(case_path.glob("*.wav"))
    
    img_path = str(jpg_files[0]) if jpg_files else None
    aud_path = str(wav_files[0]) if wav_files else None
    
    return img_path, aud_path

def main():
    if "sk-" not in cfg.API_KEY:
        print("❌ 请先在 Config 中填入有效的 API Key")
        return

    print(f"🚀 初始化客户端: {cfg.MODEL_NAME}")
    try:
        client = get_client()
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return

    # 扫描所有最底层的用例文件夹
    # 结构: Root/Category/Sub/CaseID
    root = Path(cfg.DATASET_ROOT)
    case_dirs = []
    
    if not root.exists():
        print(f"❌ 数据集路径不存在: {cfg.DATASET_ROOT}")
        return

    print("🔍 正在扫描数据集目录...")
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
        
        # 断点续传：如果已经跑过，跳过
        if output_path.exists():
            continue
            
        img_path, aud_path = find_media_files(case_dir)
        
        # 简单检查：至少需要一种模态
        if not img_path and not aud_path:
            # 记录错误文件，避免下次重复扫描，或者仅打印跳过
            print(f"⚠️ 跳过 {case_dir.name}: 缺少媒体文件")
            continue
            
        # 生成回复
        response_text = generate_response(client, img_path, aud_path)
        
        # 保存结果
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_text if response_text else "[No Response]")
        except Exception as e:
            print(f"❌ 写入文件失败 {output_path}: {e}")

    print("\n✅ 所有评测完成！")

if __name__ == "__main__":
    main()
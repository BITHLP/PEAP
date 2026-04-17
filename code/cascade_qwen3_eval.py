import os
PEAP_ROOT = os.environ.get("PEAP_ROOT", "/home/twlan/EmbodiedAI/PEAP")
PEAP_BENCHMARK_ROOT = os.environ.get("PEAP_BENCHMARK_ROOT", "/home/twlan/EmbodiedAI/PEAP_Benchmark")

import os
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

def is_numbered_folder(folder_name):
    """检查文件夹名是否为纯数字编号"""
    return re.fullmatch(r'\d+', folder_name) is not None

def count_valid_folders(root_dir):
    """统计所有符合条件的数字文件夹数量"""
    count = 0
    for dirpath, dirnames, _ in os.walk(root_dir):
        current_dir = os.path.basename(dirpath)
        if is_numbered_folder(current_dir):
            # 检查是否是最后一层数字编号文件夹
            has_numbered_subfolder = any(is_numbered_folder(d) for d in dirnames)
            if not has_numbered_subfolder:
                count += 1
    return count

def read_txt_file(file_path):
    """读取txt文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return ""

def construct_prompt(scene_desc, audio_desc):
    """构造完整的prompt"""
    prompt = """You are a robot that can proactively engage in conversations with humans based on the scenes it observes and the sounds it hears. It interacts with the environment to support the conversation process, and its role is to actively assist the staff in the scene where it is located with their work or activities. For example, when at home, it helps residents perform life-related operations; when in a classroom, it assists teachers in carrying out teaching-related activities. Currently, you are in a scene depicted in an image and can simultaneously hear the sound of a background audio clip. It is important to note that the description of this scene is from your first-person perspective, and your next steps should be based on this perspective. I will provide you with a description of the image scene, a description of the audio, as well as the range of actions you can take and corresponding examples. Please, based on your combined understanding of the image scene and the audio, carry out the subsequent consecutive steps of action in accordance with the rules I have set.
The description of the image scene is:{scene_desc}
The description of the background audio is:{audio_desc}
The range of actions you can take falls into three main categories. The first category is [Movement], which includes actions that change your own position and direction, such as walking, moving, and turning. These actions are mainly activities performed from your own perspective. The second category is [Manipulation], which includes actions that occur when interacting with objects, such as grasping, placing, pushing, pulling, and rotating. The difference between [Manipulation] and [Movement] is that [Manipulation] requires objects other than the robot itself, such as taking a towel, picking up a piece of garbage, and pouring a glass of water. The third category is [Conversation], which includes all behaviors related to dialogue, including asking questions (Ask), answering questions (Answer), and proactively bringing up a topic (Raise a Topic). This category indicates the dialogue strategy for the current step.
When outputting your next action or statement, first output the label of the action or statement (starting with the major category label followed by the minor category label), and then output the specific content. For example:
[Conversation][Raise a Topic] That drilling sound seems quite close - perhaps they're doing renovations in the adjacent unit?
[Movement][Turn] Rotate toward the direction of the sound to better assess its source.
[Manipulation][Grab] Pick up the electric kettle from the countertop.
[Conversation][Ask] Would you like me to prepare some tea or coffee while we wait? The hot water might help mask the noise somewhat.
[Movement][Change Position] Move toward the sink to fill the kettle with water.
Print only the dialog itself, with each line at the beginning of the category.Don't print the rest of the dialog, but print a few more lines."""

#     prompt = """You are a robot that proactively engages in conversations with humans based on the scenes it observes and the sounds it hears. It interacts with the environment to support the conversation process, and its role is to actively assist the staff in the scene where it is located with their work or activities. For instance, when at home, it helps residents perform daily life-related tasks; when in a classroom, it assists teachers in conducting teaching-related activities. Currently, you are in a scene depicted in an image and can simultaneously hear a background audio clip. It is important to note that the description of this scene is from your first-person perspective, and your plans for the next steps should be based on this perspective. I will provide you with a description of the image scene and a description of the audio. Based on your combined understanding of the image scene and the audio, please specify what you will take the initiative to do next, clarify the purpose of your action, and outline the general content of what you will say. It is crucial to ensure that the content you output is consistent with the image scene—do not include elements or actions that do not exist in the image, and do not provide unnecessary information.
# The description of the image scene is:{scene_desc}
# The description of the background audio is:{audio_desc}"""
    return prompt.format(scene_desc=scene_desc, audio_desc=audio_desc)

def process_directory(root_dir, model, tokenizer):
    """递归处理目录，查找数字编号子文件夹中的JPG和对应TXT文件"""
    # 首先统计总共有多少个待处理的文件夹
    total_folders = count_valid_folders(root_dir)
    print(f"\n共发现 {total_folders} 个待处理的数字文件夹")
    
    processed_folders = 0
    processed_files = 0
    skipped_folders = 0
    already_done_folders = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录是否是数字编号文件夹
        current_dir = os.path.basename(dirpath)
        if not is_numbered_folder(current_dir):
            continue

        # 检查是否是最后一层数字编号文件夹（子文件夹中没有数字编号文件夹）
        has_numbered_subfolder = any(is_numbered_folder(d) for d in dirnames)
        if has_numbered_subfolder:
            continue

        # 检查是否已经存在Qwen3-32B.txt
        output_file = os.path.join(dirpath, "Qwen3-32B.txt")
        if os.path.exists(output_file):
            print(f"跳过 {dirpath}，已存在Qwen3-32B.txt")
            already_done_folders += 1
            processed_folders += 1  # 计入已处理
            continue

        # 更新进度显示
        remaining = total_folders - processed_folders - skipped_folders - already_done_folders
        print(f"\n当前进度: 已完成 {processed_folders}/{total_folders} 文件夹 (其中 {already_done_folders} 个已存在结果), 剩余 {remaining} 个")
        
        # 查找JPG和WAV文件及其对应的TXT文件
        jpg_files = [f for f in filenames if f.lower().endswith('.jpg')]
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        
        if not jpg_files or not wav_files:
            print(f"警告: {dirpath} 中没有找到JPG或WAV文件")
            skipped_folders += 1
            continue

        # 假设每个数字文件夹中只有一个JPG和一个WAV文件
        jpg_file = jpg_files[0]
        wav_file = wav_files[0]
        
        # 获取对应的TXT文件路径
        scene_txt_path = os.path.join(dirpath, os.path.splitext(jpg_file)[0] + '.txt')
        audio_txt_path = os.path.join(dirpath, os.path.splitext(wav_file)[0] + '.txt')
        
        if not os.path.exists(scene_txt_path) or not os.path.exists(audio_txt_path):
            print(f"警告: {dirpath} 中缺少场景或音频描述TXT文件")
            skipped_folders += 1
            continue

        # 读取场景描述和音频描述
        scene_desc = read_txt_file(scene_txt_path)
        audio_desc = read_txt_file(audio_txt_path)
        
        if not scene_desc or not audio_desc:
            print(f"警告: {dirpath} 中的场景或音频描述为空")
            skipped_folders += 1
            continue

        # 构造prompt
        prompt = construct_prompt(scene_desc, audio_desc)
        
        # 准备模型输入
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成响应
        print(f"正在处理 {dirpath} 中的 {jpg_file} 和 {wav_file}...")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
        
        # 解析输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            # 查找思考内容结束标记
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # 提取并清理内容
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # 保存结果到当前文件夹
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        processed_files += 1
        processed_folders += 1
        print(f"处理完成，结果已保存到 {output_file}")

    # 最终统计信息
    print(f"\n处理完成。共处理 {total_folders} 个文件夹:")
    print(f"- 本次新处理: {processed_files} 个文件对")
    print(f"- 已存在结果跳过: {already_done_folders} 个文件夹")
    print(f"- 因缺少文件或内容跳过: {skipped_folders} 个文件夹")
    print(f"实际处理成功率: {processed_files/(total_folders-already_done_folders)*100:.1f}% (排除已存在结果的文件夹)")

def main():
    # 配置路径和模型
    root_dir = f"{PEAP_BENCHMARK_ROOT}/data/FilterTestData"
    model_path = f"{PEAP_ROOT}/code/model_repos/CascadeModel/Qwen3-32B"
    
    if not os.path.exists(root_dir):
        print(f"错误: 目录 {root_dir} 不存在")
        return

    # 加载模型和tokenizer
    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    print("模型加载完成")

    # 处理目录
    process_directory(root_dir, model, tokenizer)

if __name__ == "__main__":
    main()

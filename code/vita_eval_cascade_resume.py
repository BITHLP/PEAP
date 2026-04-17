#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VITA 级联评测（官方组合版 + 安全转码 + 断点复跑 + 单卡并行 + 进度条/ETA）
- A) 官方“音频查询”：--image_path + --audio_path（不带 --question）
- B) 官方“文本查询”：--image_path + --question（不带 --audio_path）

新增：
- 断点复跑：检测已完成的用例目录（存在 <model-name>.txt 且 _status.json: ok），直接跳过
- CPU 并行预处理：音频安全转码/缓存提前做，减少 GPU 空窗
- GPU 并发：一次开 N 个 demo 子进程（同卡多任务），用 --max-gpu-jobs 控制并行度
- 进度条 + ETA：预取与GPU阶段都显示进度、速度、预计剩余时间
- 即时落盘：每完成一个 pair 就立即写入 <model-name>.txt 与 _status.json（更稳的断点续跑）
"""

import argparse
import os
import sys
import re
import csv
import json
import subprocess
import time
import shutil
import tempfile
import hashlib
import queue
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Semaphore, Lock
import threading
import random

# ===== PROMPT（可按需替换）=====
PROMPT = r"""You are a robot designed for specific scenarios. Your function is to proactively engage in conversations with humans based on the scenes you observe and the sounds you hear, and interact with the environment to support the conversation process. Currently, you are in the scenario of an image or video and can hear a segment of background audio. Now, the image scenario and background audio of your location will be input to you, along with the range of actions you can take and relevant examples. Please, based on your combined understanding of this image scenario and audio, select consecutive subsequent actions in accordance with the rules I have provided.

The range of actions you can take falls into three major categories, from which you can make your choices:
• The first category is [Movement], which includes actions that change your own position and direction, such as walking, moving, and turning. These actions are mainly activities performed from your own perspective.
• The second category is [Manipulation], which includes actions that occur when interacting with objects, such as grabbing, placing, pushing, pulling, and rotating. The difference between [Manipulation] and [Movement] is that [Manipulation] requires objects other than the robot itself—for example, actions like taking a towel, picking up a piece of garbage, and pouring a glass of water.
• The third category is [Conversation], which encompasses all behaviors related to dialogue, including asking questions (Ask), answering (Answer), and proactively bringing up a topic (Raise a Topic). This category indicates the dialogue strategy for the current step.

When outputting your next action or speech, first output the label of the category and subcategory that the action or speech belongs to, then output the specific content. When outputting the label, present the major category label first, followed by the subcategory label. For example:
[Conversation][Raise a Topic] That drilling sound seems quite close - perhaps they're doing renovations in the adjacent unit?
[Movement][Turn] Rotate toward the direction of the sound to better assess its source.
[Manipulation][Grab] Pick up the electric kettle from the countertop.
[Conversation][Ask] Would you like me to prepare some tea or coffee while we wait? The hot water might help mask the noise somewhat.
[Movement][Change Position] Move toward the sink to fill the kettle with water.
"""

DEFAULT_SCRIPT_MAIN = "video_audio_demo.py"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUD_EXTS = {".wav"}
SEP = "\n\n" + "-"*60 + "\n\n"

TEXT_FLAG_CANDS  = ["--question","--query","--text","--prompt","--query-text","--user-prompt","--instruction"]
IMAGE_FLAG_CANDS = ["--image_path","--image-file","--image","--input-image","--image-path","--img"]
AUDIO_FLAG_CANDS = ["--audio_path","--audio-file","--audio","--input-audio","--audio-path","--wav"]
MODEL_FLAG_CANDS = ["--model_path","--model-path","--model","--ckpt","--checkpoint"]
CONV_FLAG_CANDS  = ["--conv_mode","--conv-mode","--mode"]
MTYPE_FLAG_CANDS = ["--model_type","--model-type"]

# ---------- 小工具：时间/进度/安全打印 ----------
_print_lock = Lock()
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def bar(done: int, total: int, width: int = 28) -> str:
    total = max(total, 1)
    p = min(max(done / total, 0.0), 1.0)
    n_full = int(p * width)
    return "[" + "#" * n_full + "-" * (width - n_full) + f"] {done}/{total} ({p*100:5.1f}%)"

def eta(start_t: float, done: int, total: int) -> str:
    if done <= 0:
        return "ETA --:--:--"
    elapsed = time.time() - start_t
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {hms(remain)} | {rate:.2f}/s"

def pinfo(msg: str):
    with _print_lock:
        print(f"[{ts()}] {msg}", flush=True)

# ---------- 基本工具 ----------
def list_files(d: Path, exts: set) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

def find_case_dirs(root: Path) -> List[Path]:
    out = []
    for d in root.rglob("*"):
        if not d.is_dir(): continue
        if list_files(d, IMG_EXTS) and list_files(d, AUD_EXTS):
            out.append(d)
    return sorted(out)

def pick_pairs(case_dir: Path, limit: int=1) -> List[Tuple[Path, Path]]:
    imgs = list_files(case_dir, IMG_EXTS)
    wavs = list_files(case_dir, AUD_EXTS)
    pairs: List[Tuple[Path, Path]] = []
    idx = {w.stem: w for w in wavs}
    for im in imgs:
        w = idx.get(im.stem)
        if w: pairs.append((im, w))
    if not pairs and imgs and wavs:
        pairs.append((imgs[0], wavs[0]))
    return pairs[:limit] if limit>0 else pairs

def get_help(pyexe: str, script: Path, env: dict) -> str:
    try:
        r = subprocess.run([pyexe, str(script), "-h"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False, env=env, timeout=45)
        return r.stdout or ""
    except Exception:
        return ""

def detect_flag(help_txt: str, cands: List[str]) -> str:
    for f in cands:
        if f in help_txt: return f
    return cands[0]

def shortlist(help_txt: str, cands: List[str]) -> List[str]:
    hit = [c for c in cands if c in help_txt]
    return hit if hit else cands

def build_base(pyexe: str, script: Path, help_txt: str,
               ckpt: str, conv: str, mtype: str, extra: List[str]):
    mflag = detect_flag(help_txt, MODEL_FLAG_CANDS)
    cflag = detect_flag(help_txt, CONV_FLAG_CANDS)
    tflag = detect_flag(help_txt, MTYPE_FLAG_CANDS)
    base = [pyexe, str(script), mflag, ckpt, tflag, mtype, cflag, conv]
    if extra: base += extra
    flags = {
        "text":  shortlist(help_txt, TEXT_FLAG_CANDS),
        "image": shortlist(help_txt, IMAGE_FLAG_CANDS),
        "audio": shortlist(help_txt, AUDIO_FLAG_CANDS),
    }
    return base, flags

# ---------- 输出解析 ----------
def parse_outputs(stdout: str) -> str:
    s = stdout
    s = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)
    s = s.replace("\r", "")
    s = re.sub(r"(?:^|\n)Time consume:\s*[0-9.]+(?:\s*|\n)*$", "", s, flags=re.MULTILINE)
    s = s.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()

    i = s.rfind("☜")
    if i != -1:
        cand = s[i+1:].strip()
        if cand: return cand

    m = re.search(r"(LLM\s*Outputs?\s*:|LLM\s*Text\s*:|Model\s*Outputs?\s*:|Response\s*:|Answer\s*:|Assistant\s*:)\s*(.+)", s, re.I)
    if m: return m.group(2).strip()

    tag = re.compile(r"\[(Conversation|Movement|Manipulation)\]", re.I)
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    noise = ("Loading checkpoint shards","FutureWarning","Traceback (most recent call last):",
             "Please build and install Nvidia apex","Please install mamba_ssm","/site-packages/")
    filt = [ln for ln in lines if not any(t in ln for t in noise)]
    last = None
    for i in range(len(filt)-1, -1, -1):
        if tag.search(filt[i]): last = i; break
    if last is not None:
        st = last
        while st-1>=0 and (tag.search(filt[st-1]) or (len(filt[st-1])<400 and not any(t in filt[st-1] for t in noise))):
            st -= 1
        cand = "\n".join(filt[st:last+1]).strip()
        if cand: return cand

    for ln in reversed(s.splitlines()):
        t = ln.strip()
        if t and not t.startswith("Time consume") and "unrecognized arguments" not in t:
            return t
    return ""

# ---------- 子进程运行（含超时） ----------
def run_cmd(cmd: List[str], env: dict, timeout_sec: int) -> Tuple[int, str]:
    try:
        r = subprocess.run(cmd,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False, env=env, timeout=timeout_sec)
        return r.returncode, (r.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = getattr(e, "stdout", b"")
        if isinstance(out, bytes):
            try: out = out.decode("utf-8", errors="ignore")
            except Exception: out = out.decode(errors="ignore")
        elif out is None:
            out = ""
        return 124, out + "\n[TimeoutExpired]\n"

# ---------- 音频安全转码（带缓存/软链） ----------
def _has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None

def _hash_for(path: str) -> str:
    st = os.stat(path)
    key = f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()
    return hashlib.md5(key).hexdigest()[:16]

def _is_pcm16_16k_mono(sf, wav_path: str) -> bool:
    try:
        info = sf.info(wav_path)
        return (info.samplerate == 16000 and info.channels == 1 and "PCM_16" in str(info.subtype))
    except Exception:
        return False

def _verify_soundfile_readable(wav_path: str):
    try:
        import soundfile as sf
        _x, _sr = sf.read(wav_path, frames=1024, always_2d=True)
    except Exception as e:
        raise RuntimeError(
            f"[Audio backend error] 'soundfile' cannot read {wav_path}. "
            f"Install: conda install -y -c conda-forge libsndfile ffmpeg sox; pip install -U soundfile librosa\n"
            f"Details: {e}"
        )

def prepare_audio_safe(wav_path: str, tmp_root: Path) -> str:
    """
    将源音频映射到“无特殊字符的安全路径”，并做缓存：
      1) 若源已是 16k/mono/PCM_16，优先软链接到缓存文件
      2) 否则用 ffmpeg/torchaudio/soundfile 转码一次并缓存
    """
    import soundfile as sf
    tmp_root.mkdir(parents=True, exist_ok=True)
    tag = _hash_for(wav_path)
    safe_out = tmp_root / f"audio_{tag}.wav"
    if safe_out.exists() and safe_out.stat().st_size > 0:
        _verify_soundfile_readable(str(safe_out))
        return str(safe_out)

    if _is_pcm16_16k_mono(sf, wav_path):
        try:
            os.symlink(os.path.abspath(wav_path), str(safe_out))
        except Exception:
            shutil.copyfile(wav_path, str(safe_out))
        _verify_soundfile_readable(str(safe_out))
        return str(safe_out)

    if _has_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", wav_path, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            str(safe_out),
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if r.returncode == 0 and safe_out.exists() and safe_out.stat().st_size > 0:
            _verify_soundfile_readable(str(safe_out))
            return str(safe_out)

    try:
        import torchaudio, torch
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() > 1 and wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        torchaudio.save(str(safe_out), wav, sr, encoding="PCM_S", bits_per_sample=16)
        _verify_soundfile_readable(str(safe_out))
        return str(safe_out)
    except Exception:
        pass

    try:
        import soundfile as sff, librosa
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        sff.write(str(safe_out), y, sr, subtype="PCM_16")
        _verify_soundfile_readable(str(safe_out))
        return str(safe_out)
    except Exception:
        pass

    shutil.copyfile(wav_path, str(safe_out))
    _verify_soundfile_readable(str(safe_out))
    return str(safe_out)

# ---------- 级联 A / B ----------
def try_audio_desc_official(base, flags, image, audio, env, timeout, out_dir: Path, allow_audio_only=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs_audio"
    logs_dir.mkdir(parents=True, exist_ok=True)
    images = (flags.get("image") or [""])[:3] or [""]
    audios = (flags.get("audio") or [""])[:3] or [""]
    tried_cmds, attempt = [], 0
    for imf in images:
        for auf in audios:
            attempt += 1
            cmd = base[:]
            if imf: cmd += [imf, image]
            if auf: cmd += [auf, audio]
            tried_cmds.append(" ".join(cmd))
            rc, out = run_cmd(cmd, env, timeout)
            (logs_dir / f"attempt_{attempt:02d}.log").write_text(out, encoding="utf-8")
            if rc == 0 and "unrecognized arguments" not in out and "invalid choice" not in out:
                (out_dir / "cmd_audio.txt").write_text(" ".join(cmd), encoding="utf-8")
                (out_dir / "cmd_audio.tried.txt").write_text("\n".join(tried_cmds), encoding="utf-8")
                return rc, out
    if allow_audio_only:
        for auf in audios:
            attempt += 1
            cmd = base[:]
            if auf: cmd += [auf, audio]
            tried_cmds.append(" ".join(cmd))
            rc, out = run_cmd(cmd, env, timeout)
            (logs_dir / f"attempt_{attempt:02d}.log").write_text(out, encoding="utf-8")
            if rc == 0 and "unrecognized arguments" not in out and "invalid choice" not in out:
                (out_dir / "cmd_audio.txt").write_text(" ".join(cmd), encoding="utf-8")
                (out_dir / "cmd_audio.tried.txt").write_text("\n".join(tried_cmds), encoding="utf-8")
                return rc, out
    (out_dir / "cmd_audio.tried.txt").write_text("\n".join(tried_cmds), encoding="utf-8")
    return 1, "audio-desc failed"

def try_image_text_official(base, flags, image, question, env, timeout, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs_final"
    logs_dir.mkdir(parents=True, exist_ok=True)
    texts  = (flags.get("text")  or [""])[:3] or [""]
    images = (flags.get("image") or [""])[:3] or [""]
    tried_cmds, attempt = [], 0
    for tf in texts:
        for imf in images:
            attempt += 1
            cmd = base[:]
            if imf: cmd += [imf, image]
            if tf:  cmd += [tf,  question]
            tried_cmds.append(" ".join(cmd))
            rc, out = run_cmd(cmd, env, timeout)
            (logs_dir / f"attempt_{attempt:02d}.log").write_text(out, encoding="utf-8")
            if rc == 0 and "unrecognized arguments" not in out and "invalid choice" not in out:
                (out_dir / "cmd_final.txt").write_text(" ".join(cmd), encoding="utf-8")
                (out_dir / "cmd_final.tried.txt").write_text("\n".join(tried_cmds), encoding="utf-8")
                return rc, out
    (out_dir / "cmd_final.tried.txt").write_text("\n".join(tried_cmds), encoding="utf-8")
    return 1, "img+text failed"

# ---------- 断点复跑 ----------
def read_status_json(out_dir: Path) -> Dict:
    f = out_dir / "_status.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def count_segments_in_final(final_txt: Path) -> int:
    if not final_txt.exists(): return 0
    txt = final_txt.read_text(encoding="utf-8").strip()
    if not txt: return 0
    return len([seg for seg in txt.split(SEP) if seg.strip()])

def is_case_done(out_dir: Path, model_name: str,
                 pairs_expected: int,
                 require_pairs_match: bool=False,
                 min_bytes: int=10) -> bool:
    final_txt = out_dir / f"{model_name}.txt"
    if not final_txt.exists() or final_txt.stat().st_size < min_bytes:
        return False
    st = read_status_json(out_dir)
    if (st.get("status") == "ok") and (st.get("pairs_done", 0) >= pairs_expected):
        return True
    if require_pairs_match:
        segs = count_segments_in_final(final_txt)
        return segs >= pairs_expected
    return True

def write_status(out_dir: Path, **kw):
    f = out_dir / "_status.json"
    try:
        old = read_status_json(out_dir)
        old.update(kw)
        f.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="VITA Cascade Eval (Official combos + safe audio + resume + parallel + ETA).")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out-root",     type=Path, required=True)
    ap.add_argument("--repo-root",    type=Path, default=Path("."))
    ap.add_argument("--script-main",  type=str,  default=DEFAULT_SCRIPT_MAIN)
    ap.add_argument("--vita-ckpt",    type=Path, required=True)
    ap.add_argument("--conv-mode",    type=str,  default="qwen2p5_instruct")
    ap.add_argument("--model-type",   type=str,  default="qwen2p5_instruct")
    ap.add_argument("--model-name",   type=str,  default="VITA")
    ap.add_argument("--pairs-per-case", type=int, default=1)
    ap.add_argument("--max-cases",    type=int,  default=0)
    ap.add_argument("--cuda",         type=str,  default="")
    ap.add_argument("--per-call-timeout", type=int, default=300)
    ap.add_argument("--first-call-timeout", type=int, default=900)
    ap.add_argument("--audio-only-fallback", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--tmp-root", type=Path, default=None, help="转码后的临时目录（推荐 /dev/shm/...）")
    ap.add_argument("--extra-args",   nargs=argparse.REMAINDER)
    ap.add_argument("--hf-offline",   action="store_true")
    ap.add_argument("--hf-endpoint",  type=str, default="")
    ap.add_argument("--hf-cache",     type=str, default="")
    # 断点复跑
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-require-pairs-match", action="store_true")
    # 并行/流水线
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) // 2),
                    help="CPU 预处理线程数（音频转码/I-O 预取），建议 4~16")
    ap.add_argument("--max-gpu-jobs", type=int, default=1,
                    help="同一张卡上同时允许的 GPU 推理作业数。根据显存占用自行调大。")
    args = ap.parse_args()

    ds_root  = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    repo_root= args.repo_root.resolve()
    script   = (repo_root / args.script_main).resolve()
    if not ds_root.exists(): raise FileNotFoundError(ds_root)
    if not script.exists():  raise FileNotFoundError(script)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda: env["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.hf_offline:
        env["HF_HUB_OFFLINE"]="1"; env["TRANSFORMERS_OFFLINE"]="1"
    if args.hf_endpoint: env["HF_ENDPOINT"]=args.hf_endpoint
    if args.hf_cache:
        env["HF_HOME"]=args.hf_cache; env["HUGGINGFACE_HUB_CACHE"]=args.hf_cache
    env["PYTHONUNBUFFERED"]="1"
    pinfo(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")

    help_txt = get_help(sys.executable, script, env)
    base, flags = build_base(sys.executable, script, help_txt,
                             str(args.vita_ckpt), args.conv_mode, args.model_type,
                             args.extra_args or [])

    # 会话级安全临时目录
    if args.tmp_root is None:
        tmp_session_dir = Path(tempfile.mkdtemp(prefix="vita_audio_"))
    else:
        tmp_session_dir = args.tmp_root.resolve()
        tmp_session_dir.mkdir(parents=True, exist_ok=True)

    # 预热（图+文）
    if args.warmup:
        some_img = next((p for p in ds_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS), None)
        if some_img:
            pinfo("Warmup image+text ...")
            tf = (flags["text"][0] if flags["text"] else "")
            cmd = base[:] + [flags["image"][0], str(some_img)]
            if tf: cmd += [tf, "Describe this image in one sentence."]
            _rc, _out = run_cmd(cmd, env, args.first_call_timeout)
            pinfo("Warmup done.")

    case_dirs = find_case_dirs(ds_root)
    if args.max_cases>0: case_dirs = case_dirs[:args.max_cases]
    pinfo(f"Found {len(case_dirs)} case dir(s). Start evaluating...")

    rows = []
    t0 = time.time()

    # -------- 组装任务列表并剔除已完成 --------
    Task = Tuple[int, Path, Path, Path]  # (ci, out_dir, img, wav)
    all_tasks: List[Task] = []
    case_info = []  # [(ci, rel, out_dir, pairs)]
    ci2rel: Dict[int, Path] = {}

    for ci, cdir in enumerate(case_dirs, 1):
        rel = cdir.relative_to(ds_root)
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = pick_pairs(cdir, args.pairs_per_case)
        pairs_expected = len(pairs)
        case_info.append((ci, rel, out_dir, pairs))
        ci2rel[ci] = rel

        if args.resume and is_case_done(out_dir, args.model_name, pairs_expected,
                                        require_pairs_match=args.resume_require_pairs_match):
            pinfo(f"[{ci}/{len(case_dirs)}] {rel} -> {pairs_expected} pair(s)  - Skip (resume)")
            rows.append({"case_dir":str(rel), "pairs":pairs_expected, "has_answer":1, "status":"skipped"})
            continue

        pinfo(f"[{ci}/{len(case_dirs)}] {rel} -> {pairs_expected} pair(s)  - Enqueue")
        for (img, wav) in pairs:
            all_tasks.append((ci, out_dir, img, wav))

    if not all_tasks:
        pinfo("No pending tasks. Nothing to do.")
        with (out_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["case_dir","pairs","has_answer","status"])
            wr.writeheader(); wr.writerows(rows)
        pinfo(f"All done. ✅  (0 pending; {time.time()-t0:.1f}s)")
        return

    pinfo(f"Pending tasks: {len(all_tasks)} | CPU prefetch threads={args.workers} | GPU parallel={args.max_gpu_jobs}")

    # -------- 阶段A：CPU 并行预处理（音频安全转码/缓存）--------
    prefetched: "queue.Queue[Tuple[int, Path, Path, str]]" = queue.Queue()
    errors: List[str] = []
    prefetch_start = time.time()
    total_prefetch = len(all_tasks)
    prefetch_done = 0

    def _prep_one(task: Task):
        ci, out_dir, img, wav = task
        rel = ci2rel.get(ci, Path("."))
        safe_wav = prepare_audio_safe(str(wav), tmp_session_dir)
        prefetched.put((ci, out_dir, img, safe_wav))
        return (ci, rel, img.name, Path(safe_wav).name)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_prep_one, t) for t in all_tasks]
        for fut in as_completed(futs):
            try:
                ci, rel, img_name, wav_safe_name = fut.result()
                prefetch_done += 1
                if (prefetch_done % 20 == 0) or (prefetch_done == total_prefetch):
                    b = bar(prefetch_done, total_prefetch)
                    pinfo(f"[Prefetch] {b} | {eta(prefetch_start, prefetch_done, total_prefetch)}")
            except Exception as e:
                errors.append(str(e))
                prefetch_done += 1
                b = bar(prefetch_done, total_prefetch)
                pinfo(f"[Prefetch] {b} | ERROR: {e}")

    if errors:
        for e in errors: pinfo("[PrefetchError] " + e)
    pinfo(f"[Prefetch] Done. elapsed={hms(time.time()-prefetch_start)} prepared={prefetched.qsize()} errors={len(errors)}")

    # -------- 阶段B：GPU 并发消费（每个任务子进程完成 A→B，并即时落盘）--------
    answers_acc = defaultdict(list)      # 仅用于统计；真正写盘在完成时
    gpu_slots = Semaphore(max(1, int(args.max_gpu_jobs)))
    dir_locks: Dict[Path, Lock] = defaultdict(Lock)

    pairs_expected_map = {out_dir: len(pairs) for _, _, out_dir, pairs in case_info}

    def _append_and_status(out_dir: Path, model_name: str, final: str):
        """线程安全地把一个 final 片段追加到 <model>.txt，并刷新 _status.json"""
        with dir_locks[out_dir]:
            final_txt = out_dir / f"{model_name}.txt"
            if final is not None:
                if final_txt.exists() and final_txt.stat().st_size > 0:
                    with open(final_txt, "a", encoding="utf-8") as f:
                        f.write(SEP + final)
                else:
                    final_txt.write_text(final, encoding="utf-8")
            segs = count_segments_in_final(final_txt)
            write_status(
                out_dir,
                status="ok" if segs > 0 else "cascade_audio_failed",
                pairs_done=segs,
                pairs_expected=pairs_expected_map.get(out_dir, 1),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _run_one(ci: int, out_dir: Path, img: Path, safe_wav: str):
        gpu_slots.acquire()
        rel = ci2rel.get(ci, Path("."))
        try:
            # A 步
            pinfo(f"[Run] A  | CI#{ci} | {rel} | img={img.name} | wav={Path(safe_wav).name}")
            rc_a, out_a = try_audio_desc_official(
                base, flags, str(img), safe_wav, env, args.per_call_timeout, out_dir,
                allow_audio_only=args.audio_only_fallback
            )
            if rc_a != 0:
                pinfo(f"[Run] A!!| CI#{ci} | {rel} | FAILED -> {out_dir}/logs_audio")
                return (out_dir, None, ci, rel)

            audio_desc = parse_outputs(out_a)
            (out_dir / f"{args.model_name}_audio_first.txt").write_text(audio_desc, encoding="utf-8")

            # B 步
            pinfo(f"[Run] B  | CI#{ci} | {rel} | image+text")
            question = "[Audio Summary]\n" + (audio_desc.strip() if audio_desc else "(empty)") + "\n\n" + PROMPT.strip()
            rc_b, out_b = try_image_text_official(
                base, flags, str(img), question, env, args.per_call_timeout, out_dir
            )
            if rc_b != 0:
                pinfo(f"[Run] B!!| CI#{ci} | {rel} | FAILED -> {out_dir}/logs_final")
                return (out_dir, None, ci, rel)

            final = parse_outputs(out_b)
            return (out_dir, final, ci, rel)
        finally:
            gpu_slots.release()

    gpu_start = time.time()
    total_gpu = prefetched.qsize()
    gpu_done = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_gpu_jobs))) as ex:
        futures = []
        while not prefetched.empty():
            ci, out_dir, img, safe_wav = prefetched.get()
            futures.append(ex.submit(_run_one, ci, out_dir, img, safe_wav))

        for fut in as_completed(futures):
            out_dir, final, ci, rel = fut.result()
            if final is not None:
                _append_and_status(out_dir, args.model_name, final)
                answers_acc[out_dir].append(final)
                gpu_done += 1
                b = bar(gpu_done, total_gpu)
                pinfo(f"[Done] {b} | {eta(gpu_start, gpu_done, total_gpu)} | CI#{ci} {rel}")
            else:
                # 失败也计入进度（可选）；不写入 final
                gpu_done += 1
                b = bar(gpu_done, total_gpu)
                pinfo(f"[Skip] {b} | {eta(gpu_start, gpu_done, total_gpu)} | CI#{ci} {rel} (failed)")

    pinfo(f"[Run] GPU stage done. elapsed={hms(time.time()-gpu_start)}")

    # -------- 汇总 CSV（最终扫一遍状态写表） --------
    for ci, rel, out_dir, pairs in case_info:
        if not out_dir.exists():
            continue
        final_txt_path = out_dir / f"{args.model_name}.txt"
        segs = count_segments_in_final(final_txt_path)
        status = "ok" if segs >= len(pairs) else ("partial_ok" if segs > 0 else "cascade_audio_failed")
        write_status(
            out_dir,
            status=status,
            pairs_done=segs,
            pairs_expected=len(pairs),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        rows.append({"case_dir":str(rel), "pairs":len(pairs), "has_answer":int(segs>0), "status":status})

    with (out_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["case_dir","pairs","has_answer","status"])
        wr.writeheader(); wr.writerows(rows)

    pinfo(f"All done. ✅  ({len(case_dirs)} case dir(s), elapsed={hms(time.time()-t0)})")

if __name__ == "__main__":
    main()

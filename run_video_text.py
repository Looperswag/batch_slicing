import os
import json
import subprocess
import tempfile
from pathlib import Path
import time
import argparse
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
from datetime import datetime
import concurrent.futures
from functools import lru_cache
import re

# 视频处理库
import cv2
import ffmpeg
from pydub import AudioSegment

# 语音转文字
import whisper

# Google Vertex AI Gemini API
from google import genai
from google.genai import types


# 设置日志 - 修改为只输出ERROR和关键INFO
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    """
    配置日志系统。
    - 脚本自身的日志 (来自 logger = logging.getLogger(__name__)) 将写入文件 (DEBUG 及以上级别)。
    - 任何日志记录器 (脚本自身或库) 的消息均不通过日志系统输出到控制台。
    - 控制台输出应仅来自直接的 print() 语句 (例如，通过 print_progress)。
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"video_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 配置根日志记录器:
    # 移除其所有处理程序以防止任何默认的控制台输出。
    # 设置其级别 (例如，设置为 WARNING)，以便库中的 INFO 级别日志被早期忽略。
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.WARNING) # 库产生的 INFO 日志将不会被根记录器处理

    # 配置脚本自身的日志记录器 (__name__)
    # 'logger' 是在全局作用域定义的 logging.getLogger(__name__)
    script_logger = logging.getLogger(__name__) 
    script_logger.handlers = [] # 清除此特定记录器之前的任何处理程序
    script_logger.setLevel(logging.DEBUG) # 处理此记录器的所有级别日志
    script_logger.propagate = False # 不要将此记录器的日志传递给 (现在已静默的) 根记录器

    # 用于脚本自身记录器的文件处理程序
    script_file_handler = logging.FileHandler(log_file_path)
    script_file_handler.setLevel(logging.DEBUG)
    # 在日志格式中添加了 [%(name)s] 以便区分日志来源
    script_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    script_file_handler.setFormatter(script_file_formatter)
    script_logger.addHandler(script_file_handler)
    
    # 显式提高详细的 Google API 客户端等库的日志记录器的级别
    # 这可以防止它们生成可能被其他处理程序捕获的低级别日志记录。
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)
    logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('google.cloud.aiplatform').setLevel(logging.WARNING)
    # 根据 genai 库实际使用的记录器名称，可能需要添加更多条目
    # 例如：logging.getLogger('google.generativeai').setLevel(logging.WARNING)
    # 或 logging.getLogger('vertexai').setLevel(logging.WARNING)
    # 从您提供的日志样本看，涉及 aiplatform.googleapis.com，所以 google.cloud.aiplatform 和 vertexai 比较相关。
    logging.getLogger('vertexai').setLevel(logging.WARNING) # Gemini API 可能通过此记录器
    logging.getLogger('google.generativeai').setLevel(logging.WARNING) # 如果直接使用 GenAI SDK

    # 脚本中全局 'logger' 变量的日志将进入 script_file_handler。
    # 不会有日志通过 logging 系统输出到控制台。

    return log_file_path

key_path = "./your_key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# 添加时间格式化函数
def format_duration(seconds):
    """将秒数格式化为 xx min, xx s 格式"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours} hr, {minutes} min, {seconds} s"
    elif minutes > 0:
        return f"{minutes} min, {seconds} s"
    else:
        return f"{seconds} s"

# 添加一个进度打印函数
def print_progress(message, show_immediately=False): # show_immediately 参数现在主要用于语义
    """将进度信息打印到控制台，并将其作为 INFO 级别记录到日志文件。"""
    print(message)  # 直接输出到控制台
    logger.info(message) # 'logger' 是 logging.getLogger(__name__)，会记录到文件

# 添加批处理函数用于Gemini API调用
def batch_gemini_requests(client, requests, model_name, batch_size=4, max_retries=3):
    """
    批量处理Gemini API请求，并使用重试机制
    
    Args:
        client: Gemini API客户端
        requests: 请求列表，每个请求是(contents, config)元组
        model_name: Gemini模型名称
        batch_size: 并行处理的请求数
        max_retries: 最大重试次数
        
    Returns:
        结果列表
    """
    results = []
    
    # 定义处理单个请求的函数
    def process_request(request_data):
        contents, config = request_data
        
        for retry in range(max_retries):
            try:
                # 使用流式 API
                response_stream = client.models.generate_content_stream( # 注意是 client.models.generate_content_stream
                    model=model_name,
                    contents=contents,
                    config=config, # config 应包含 safety_settings
                )
                
                accumulated_text_parts = []
                final_prompt_feedback = None
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        accumulated_text_parts.append(chunk.text)
                    # 检查并记录 prompt_feedback (通常在最后一个 chunk 或整体响应中更完整)
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback:
                        final_prompt_feedback = chunk.prompt_feedback 
                
                if accumulated_text_parts:
                    full_text = "".join(accumulated_text_parts).strip()
                    logger.debug(f"Gemini API stream processed successfully (try {retry+1}).")
                    return full_text
                else:
                    logger.warning(f"Gemini API stream returned no text (尝试 {retry+1}/{max_retries}).")
                    if final_prompt_feedback:
                        logger.warning(f"Prompt feedback: {final_prompt_feedback}")
                        # 您可以进一步解析 final_prompt_feedback.block_reason 等
                    else:
                        logger.warning("No text and no prompt feedback from stream.")
                    
                    if retry < max_retries - 1:
                        time.sleep(min(2 ** retry, 10))  # 指数退避策略
                        continue 
                    else:
                        logger.error(f"Gemini API stream in {max_retries} attempts returned no text (batch_gemini_requests).")
                        return "未知主题" # 所有重试失败后的回退值

            except AttributeError as ae: # 捕获类似 'Client' object has no attribute 'generate_content_stream' 的错误
                logger.error(f"Gemini API Attribute Error (尝试 {retry+1}/{max_retries}): {ae}. Ensure using client.models.generate_content_stream.")
                # 此错误通常是编码错误，重试可能无效，但为了流程完整性保留重试
                if retry < max_retries - 1:
                    time.sleep(min(2 ** retry, 10))
                else:
                    return "API调用结构错误" # 指示编码问题
            except Exception as e:
                logger.warning(f"Gemini API stream call failed (尝试 {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    time.sleep(min(2 ** retry, 10))
        
        logger.error(f"所有{max_retries}次尝试都失败 (process_request in batch_gemini_requests)")
        return "未知主题"
    
    # 使用线程池并行处理请求 (这部分保持不变)
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(process_request, req) for req in requests]
        
        # 添加进度显示
        total = len(requests)
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            
            # 每完成10%的请求显示一次进度
            if completed % max(1, total // 10) == 0 or completed == total:
                progress = (completed / total) * 100
                print_progress(f"内容分析进度: {progress:.1f}% ({completed}/{total})", 
                              show_immediately=(completed == total))
    
    return results
    

class VideoSplitter:
    def __init__(self, project_id="your_project_id", location="your_location", model_name="gemini-2.5-flash-preview-05-20", 
                 whisper_model="base", output_dir="./output"):
        """
        初始化视频分割器
        
        Args:
            project_id: Google Cloud 项目ID
            location: Google Cloud 区域
            model_name: Gemini 模型名称
            whisper_model: Whisper 模型大小
            output_dir: 输出目录
        """
        self.temp_dir = tempfile.mkdtemp()
        logger.debug(f"使用临时目录: {self.temp_dir}")
        
        # 设置日志
        self.log_file = setup_logging(output_dir)
        
        # 初始化 Vertex AI Gemini 客户端
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        
        self.model_name = model_name
        self.frame_interval = 5.0  # 默认帧间隔（秒）
        self.chunk_duration = 30.0  # 默认文本块大小（秒）
        self.max_workers = 4  # 默认并行任务数量
        
        # 加载 Whisper 模型
        print_progress("正在加载语音识别模型...", True)
        self.whisper_model = whisper.load_model(whisper_model)
        print_progress("语音识别模型加载完成", True)
        
        # 初始化时间记录属性
        self.start_time = None
        self.end_time = None
        
        # 添加LRU缓存，避免重复分析相似内容
        self._topic_cache = {}
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """获取视频元数据，包括时长、帧率等"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # 优化策略：针对长视频的调整
        # 如果视频较长，增加帧间隔和文本块大小
        frame_interval = self.frame_interval
        chunk_duration = self.chunk_duration
        
        if duration > 3600:  # 超过1小时
            frame_interval = max(10.0, frame_interval)
            chunk_duration = max(45.0, chunk_duration)
            logger.debug(f"视频较长 ({format_duration(duration)}), 已调整帧间隔为 {frame_interval}秒，文本块大小为 {chunk_duration}秒")
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_interval": frame_interval,
            "chunk_duration": chunk_duration
        }
    
    def extract_audio(self, video_path: str) -> str:
        """
        从视频中提取音频，针对长视频进行优化
        - 使用更高效的编码器
        - 降低采样率
        """
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        
        # 针对长视频，降低音频质量以加快处理速度
        command = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000",  # 降低采样率
            "-ac", "1",      # 单声道
            "-q:a", "0", "-map", "a", audio_path, "-y"
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug(f"音频提取完成: {audio_path}")
        return audio_path
    
    def transcribe_audio(self, audio_path: str, metadata: Dict) -> Dict:
        """
        使用Whisper转写音频为带时间戳的文本
        - 对于长视频，分段处理
        """
        print_progress("开始音频转写...", True)
        
        video_duration = metadata["duration"]
        
        # 如果视频很长（超过30分钟），考虑分段处理
        if video_duration > 1800:
            print_progress(f"视频较长，将分段进行转写，总时长：{format_duration(video_duration)}", True)
            return self._transcribe_long_audio(audio_path, video_duration)
        else:
            # 标准处理
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
            print_progress("音频转写完成", True)
            return result
    
    def _transcribe_long_audio(self, audio_path: str, duration: float) -> Dict:
        """
        处理长音频的转写，分段处理后合并结果
        """
        # 分段长度（秒）
        segment_length = 10 * 60  # 10分钟一段
        
        # 将音频分割成多个片段
        audio = AudioSegment.from_file(audio_path)
        segments_results = []
        
        # 计算段数
        num_segments = int(np.ceil(duration / segment_length))
        print_progress(f"将音频分为 {num_segments} 段进行转写", True)
        
        for i in range(num_segments):
            start_time = i * segment_length * 1000  # 毫秒
            end_time = min((i + 1) * segment_length * 1000, len(audio))
            
            segment_audio = audio[start_time:end_time]
            segment_path = os.path.join(self.temp_dir, f"segment_{i}.wav")
            segment_audio.export(segment_path, format="wav")
            
            print_progress(f"转写音频段 {i+1}/{num_segments}...", True)
            result = self.whisper_model.transcribe(segment_path, word_timestamps=True)
            
            # 调整时间戳
            time_offset = i * segment_length
            for segment in result["segments"]:
                segment["start"] += time_offset
                segment["end"] += time_offset
                if "words" in segment:
                    for word in segment["words"]:
                        word["start"] += time_offset
                        word["end"] += time_offset
            
            segments_results.append(result)
            
            # 删除临时文件
            os.remove(segment_path)
        
        # 合并结果
        merged_result = {
            "text": " ".join(r["text"] for r in segments_results),
            "segments": []
        }
        
        for result in segments_results:
            merged_result["segments"].extend(result["segments"])
        
        print_progress("音频转写完成", True)
        return merged_result
    
    def extract_frames(self, video_path: str, metadata: Dict) -> List[Dict]:
        """
        每隔一定时间提取视频帧，针对长视频进行优化
        - 增加帧间隔
        - 降低图像分辨率
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = metadata["fps"]
        duration = metadata["duration"]
        frame_interval = metadata["frame_interval"]
        frame_interval_frames = int(fps * frame_interval)
        
        # 为长视频调整分辨率
        scale_factor = 1.0
        if duration > 3600:  # 1小时以上视频
            scale_factor = 0.5  # 降低分辨率
            
        print_progress(f"开始提取视频帧，间隔: {frame_interval}秒，总时长: {format_duration(duration)}", True)
        
        # 计算要提取的大致帧数
        estimated_frames = int(duration / frame_interval)
        logger.debug(f"预计将提取约 {estimated_frames} 帧")
        
        # 优化：使用seek而不是逐帧读取
        frame_count = 0
        progress_interval = max(1, estimated_frames // 10)  # 每10%显示一次进度
        
        while cap.isOpened():
            # 设置到特定位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval_frames)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            timestamp = frame_count * frame_interval
            
            # 如果需要缩放
            if scale_factor != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                frame = cv2.resize(frame, (new_w, new_h))
            
            frame_path = os.path.join(self.temp_dir, f"frame_{timestamp:.2f}.jpg")
            # 使用较低的图像质量参数以减少存储空间
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            frames.append({
                "timestamp": timestamp,
                "path": frame_path
            })
            
            frame_count += 1
            
            # 每progress_interval帧输出一次进度
            if frame_count % progress_interval == 0 or frame_count == estimated_frames:
                progress = min(100, (timestamp / duration) * 100)
                if frame_count == estimated_frames:
                    print_progress(f"帧提取完成: 共 {len(frames)} 帧", True)
                else:
                    logger.debug(f"帧提取进度: {progress:.1f}%, 当前时间点: {format_duration(timestamp)}")
                
            # 检查是否到达视频末尾
            if timestamp >= duration:
                break
                
        cap.release()
        return frames
    
    def _simple_image_hash(self, image_path: str) -> str:
        """创建图像的简单哈希值，用于内容相似性检测"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "invalid_image"
                
            # 缩小图像为8x8并转为灰度
            img = cv2.resize(img, (8, 8))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 计算平均值，然后生成二进制哈希
            avg = gray.mean()
            binary_hash = ''.join('1' if px > avg else '0' for px in gray.flatten())
            return binary_hash
        except Exception as e:
            logger.error(f"计算图像哈希时出错: {e}")
            return "error_hash"
    
    def analyze_segment(self, transcript_segment: Dict, frame_path: str) -> str:
        """
        使用Gemini分析文本和图像来确定语义内容
        - 添加缓存以避免相似内容的重复分析
        """
        try:
            # 准备文本
            text = transcript_segment["text"]
            
            # 检查缓存
            image_hash = self._simple_image_hash(frame_path)
            cache_key = f"{text}:{image_hash}"
            
            if cache_key in self._topic_cache:
                topic = self._topic_cache[cache_key]
                logger.debug(f"使用缓存结果: {topic[:30]}...")
                return topic
            
            # 加载图像
            with open(frame_path, "rb") as f:
                image_data = f.read()
            
            # 构建提示
            prompt = f"""
            分析以下视频片段的内容：
            
            这是从****截取的N帧图像，和对应时间的文本: "{text}"
            请根据以上信息，输出一个标签，，格式如下：'******'。
            
            """
            
            # 准备请求内容
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg",
            )
            text_part = types.Part.from_text(text=prompt)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        image_part,
                        text_part
                    ]
                )
            ]
            
            safety_settings_off = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]



            generate_content_config = types.GenerateContentConfig(
                temperature=0.9,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                safety_settings=safety_settings_off
            )
            
            max_retries_analyze = 3
            for attempt in range(max_retries_analyze):
                try:
                    if attempt > 0:
                        time.sleep(min(2 ** (attempt -1), 10))
                        logger.debug(f"Retrying API call (analyze_segment attempt {attempt+1}/{max_retries_analyze})...")

                    # 使用流式 API: self.client.models.generate_content_stream
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=generate_content_config,
                    )

                    accumulated_text_parts = []
                    current_prompt_feedback = None
                    for chunk in response_stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            accumulated_text_parts.append(chunk.text)
                        if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback:
                            current_prompt_feedback = chunk.prompt_feedback
                    
                    if accumulated_text_parts:
                        topic = "".join(accumulated_text_parts).strip()
                        logger.debug(f"识别的主题 (attempt {attempt+1}): {topic} (文本: {text[:30]}...)")
                        self._topic_cache[cache_key] = topic
                        return topic
                    else:
                        logger.warning(f"Gemini API stream (analyze_segment attempt {attempt+1}) returned no text.")
                        if current_prompt_feedback:
                            logger.warning(f"Prompt feedback (analyze_segment attempt {attempt+1}): {current_prompt_feedback}")
                        else:
                             logger.warning(f"Stream completed without text (analyze_segment attempt {attempt+1}).")
                        
                        if attempt == max_retries_analyze - 1:
                            logger.error(f"All {max_retries_analyze} attempts in analyze_segment failed to get text from stream.")
                            return "未知主题"
                
                except AttributeError as ae:
                    logger.error(f"Gemini API Attribute Error (analyze_segment attempt {attempt+1}): {ae}. Ensure using self.client.models.generate_content_stream.")
                    if attempt == max_retries_analyze - 1:
                        return "API调用结构错误"
                except Exception as e:
                    logger.error(f"Gemini API stream call or processing error (analyze_segment attempt {attempt+1}): {e}")
                    if attempt == max_retries_analyze - 1:
                        logger.error(f"All {max_retries_analyze} attempts in analyze_segment failed due to exception.")
                        return "未知主题"
            
            return "未知主题" # Should be covered by inner returns

        except Exception as general_e: # Catch errors in setup before API call
            logger.error(f"Error in analyze_segment before API call: {general_e}")
            return "未知主题 (setup error)"
            
            

    
    def identify_topic_changes(self, transcript: Dict, frames: List[Dict], metadata: Dict) -> List[Dict]:
        """
        识别主题变化点，使用并行处理和批量API调用
        """
        video_duration = metadata["duration"]
        chunk_duration = metadata["chunk_duration"]
        
        # 按照较大的粒度分割文本
        text_chunks = []
        current_chunk = {"text": "", "start": 0, "end": 0}
        
        print_progress(f"开始分析视频内容，按 {chunk_duration} 秒的间隔处理...", True)
        
        # 每chunk_duration秒左右形成一个文本块
        for segment in transcript["segments"]:
            if current_chunk["text"] == "":
                current_chunk["start"] = segment["start"]
                
            current_chunk["text"] += segment["text"] + " "
            current_chunk["end"] = segment["end"]
            
            if segment["end"] - current_chunk["start"] >= chunk_duration:
                text_chunks.append(current_chunk)
                current_chunk = {"text": "", "start": 0, "end": 0}
                
        # 添加最后一个块
        if current_chunk["text"]:
            text_chunks.append(current_chunk)
            
        logger.debug(f"将音频分成了 {len(text_chunks)} 个分析块")
        
        # 准备批量处理请求
        analysis_requests = []
        chunk_frame_pairs = []
        
        for chunk in text_chunks:
            # 找到最近的帧
            closest_frame = min(frames, key=lambda f: abs(f["timestamp"] - chunk["start"]))
            chunk_frame_pairs.append((chunk, closest_frame))
            
            # 加载图像
            with open(closest_frame["path"], "rb") as f:
                image_data = f.read()
                
            # 构建提示
            prompt = f"""
            分析以下视频片段的内容：
            
            这是从***视频中截取的N帧图像，和对应的文本: "{chunk["text"]}"
            请根据以上信息，输出一个标签，格式如下：'********'。
            """
            
            # 准备请求内容
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg",
            )
            text_part = types.Part.from_text(text=prompt)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        image_part,
                        text_part
                    ]
                )
            ]
            safety_settings_off = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=0.9,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                safety_settings=safety_settings_off
            )
            
            analysis_requests.append((contents, generate_content_config))
        
        # 批量处理API请求
        print_progress(f"开始分析视频内容 ({len(analysis_requests)} 个片段)...", True)
        
        batch_size = min(self.max_workers, 10)  # 限制最大并行请求数
        topics = batch_gemini_requests(self.client, analysis_requests, self.model_name, batch_size=batch_size)
        
        print_progress("内容分析完成，识别主题变化...", True)
        
        # 处理主题并识别变化点
        segments = []
        current_segment = None
        current_topic = None
        
        for i, ((chunk, _), topic) in enumerate(zip(chunk_frame_pairs, topics)):
            # 如果是新主题，创建新分段
            if topic != current_topic:
                if current_segment:
                    segments.append(current_segment)
                    
                current_segment = {
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "topic": topic
                }
                current_topic = topic
            else:
                # 扩展当前分段
                current_segment["end"] = chunk["end"]
                
        # 添加最后一个分段
        if current_segment:
            segments.append(current_segment)
            
        print_progress(f"识别出 {len(segments)} 个不同主题分段", True)
        
        # 合并过短的分段（小于10秒）
        if len(segments) > 1:
            merged_segments = []
            current = segments[0]
            
            for next_segment in segments[1:]:
                # 如果当前分段很短，尝试合并
                if next_segment["end"] - next_segment["start"] < 10:
                    logger.debug(f"合并短分段: {next_segment['topic']} (时长 {next_segment['end'] - next_segment['start']:.2f}秒)")
                    current["end"] = next_segment["end"]
                else:
                    merged_segments.append(current)
                    current = next_segment
                    
            merged_segments.append(current)
            
            logger.debug(f"合并短分段后，共有 {len(merged_segments)} 个分段")
            segments = merged_segments
            
        return segments
    
    def split_video(self, video_path: str, segments: List[Dict], output_dir: str):
        """
        根据分段信息切分视频，使用并行处理加速
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 为这个视频创建一个子文件夹
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print_progress(f"开始切分视频到 {video_output_dir}", True)
        
        # 准备切分任务
        split_tasks = []
        
        for i, segment in enumerate(segments):
            # 清理主题名称，用作文件名
            topic = re.sub(r'[\\/*?:"<>|]', "_", segment["topic"])
            topic = topic[:100]  # 限制文件名长度
            output_file = os.path.join(video_output_dir, f"{i+1:02d}_{topic}.mp4")
            
            split_tasks.append({
                "index": i,
                "total": len(segments),
                "start": segment["start"],
                "end": segment["end"],
                "output_file": output_file,
                "segment": segment
            })
        
        # 定义切分函数
        def split_segment(task):
            index = task["index"]
            total = task["total"]
            start = task["start"]
            end = task["end"]
            output_file = task["output_file"]
            segment = task["segment"]
            
            # 使用ffmpeg切分视频
            command = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264", "-preset", "faster",  # 使用更快的预设
                "-c:a", "aac",
                "-strict", "experimental", 
                output_file, "-y"
            ]
            
            logger.debug(f"切分视频段 {index+1}/{total}: {start:.2f}s 到 {end:.2f}s, 主题: {segment['topic'][:50]}...")
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 只在关键节点显示进度
            if (index + 1) % max(1, total // 5) == 0 or index == 0 or index == total - 1:
                progress = ((index + 1) / total) * 100
                print_progress(f"视频切分进度: {progress:.1f}% ({index+1}/{total})", True)
                
            return output_file
        
        # 并行处理切分任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = [executor.submit(split_segment, task) for task in split_tasks]
            output_files = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    output_file = future.result()
                    output_files.append(output_file)
                except Exception as e:
                    logger.error(f"切分视频时出错: {e}")
        
        # 打印最终结果 - 分段视频列表
        print_progress(f"\n视频 {video_name} 切分完成，共 {len(segments)} 个片段:", True)
        for i, segment in enumerate(segments):
            duration = segment["end"] - segment["start"]
            print_progress(f"片段 {i+1}: {segment['topic']} (时长: {format_duration(duration)})", True)
        
        # 保存分段信息
        with open(os.path.join(video_output_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
            
        print_progress(f"所有分段视频已保存到: {video_output_dir}", True)
        
        return video_output_dir
    
    def process_video(self, video_path: str, output_dir: str):
        """处理单个视频切分流程，针对长视频进行优化"""
        try:
            print_progress(f"开始处理视频: {os.path.basename(video_path)}", True)
            
            # 清理临时目录，为新视频准备
            self.clean_temp_dir()
            
            # 0. 获取视频元数据，为长视频调整参数
            metadata = self.get_video_metadata(video_path)
            print_progress(f"视频元数据: 时长={format_duration(metadata['duration'])}, 分辨率={metadata['width']}x{metadata['height']}", True)
            
            # 记录处理开始时间
            processing_start = time.time()
            
            # 1. 提取音频
            print_progress("步骤 1/5: 提取音频", True)
            audio_path = self.extract_audio(video_path)
            
            # 2. 音频转文字
            print_progress("步骤 2/5: 音频转文字", True)
            transcript = self.transcribe_audio(audio_path, metadata)
            
            # 3. 提取视频帧
            print_progress("步骤 3/5: 提取视频帧", True)
            frames = self.extract_frames(video_path, metadata)
            
            # 4. 识别主题变化点
            print_progress("步骤 4/5: 识别主题变化点", True)
            segments = self.identify_topic_changes(transcript, frames, metadata)
            
            # 5. 切分视频
            print_progress("步骤 5/5: 切分视频", True)
            video_output_dir = self.split_video(video_path, segments, output_dir)
            
            # 计算处理时间
            processing_end = time.time()
            processing_duration = processing_end - processing_start
            print_progress(f"视频处理完成: {os.path.basename(video_path)}, 耗时: {format_duration(processing_duration)}", True)
            
            return video_output_dir
            
        except Exception as e:
            # 错误仅记录到文件，不在控制台显示详细错误信息
            logger.error(f"处理视频 {video_path} 时出错: {e}") 
            # 原来的 print_progress(f"处理视频时出错: ...") 已移除，以保持控制台清洁
            # 如果需要，可以在这里 print 一个非常通用的错误提示，但根据要求“删除所有warning和报错”，保持沉默
            # print(f"处理视频 {os.path.basename(video_path)} 时发生错误。详情请查看日志文件。") # 例如
            return None

    def process_video_folder(self, input_folder: str, output_dir: str):
        """
        处理文件夹中的所有视频文件
        """
        # 记录开始时间
        self.start_time = time.time()
        
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        # 查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
            video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
        
        # 按文件大小排序，先处理较小的视频
        video_files.sort(key=lambda x: os.path.getsize(x))
        
        if not video_files:
            print_progress(f"在文件夹 {input_folder} 中未找到视频文件", True)
            return
        
        print_progress(f"在文件夹 {input_folder} 中找到 {len(video_files)} 个视频文件", True)
        
        # 创建主输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 并行处理视频，但需要控制并发数量，避免资源竞争
        max_concurrent = min(self.max_workers, 2)  # 限制并行视频处理数量
        
        if len(video_files) > 1 and max_concurrent > 1:
            print_progress(f"将使用 {max_concurrent} 个并行任务处理视频", True)
            results = self._process_videos_parallel(video_files, output_dir, max_concurrent)
        else:
            # 串行处理视频
            print_progress("将串行处理视频文件", True)
            results = self._process_videos_serial(video_files, output_dir)
        
        # 完成处理
        self.finalize_processing(results, output_dir)
        
        return results

    def _process_videos_serial(self, video_files, output_dir):
        """串行处理视频文件"""
        results = {}
        for i, video_path in enumerate(video_files):
            print_progress(f"处理视频 {i+1}/{len(video_files)}: {os.path.basename(video_path)}", True)
            try:
                video_output_dir = self.process_video(video_path, output_dir)
                if video_output_dir:
                    results[os.path.basename(video_path)] = {
                        "status": "success",
                        "output_dir": video_output_dir
                    }
                else:
                    results[os.path.basename(video_path)] = {
                        "status": "failed",
                        "error": "处理失败"
                    }
            except Exception as e:
                logger.error(f"处理视频时出错: {video_path} - {e}")
                results[os.path.basename(video_path)] = {
                    "status": "failed",
                    "error": str(e)
                }
        return results

    def _process_videos_parallel(self, video_files, output_dir, max_workers):
        """并行处理视频文件"""
        results = {}
        
        def process_single_video(video_info):
            index, total, video_path = video_info
            print_progress(f"处理视频 {index+1}/{total}: {os.path.basename(video_path)}", True)
            try:
                video_output_dir = self.process_video(video_path, output_dir)
                if video_output_dir:
                    return os.path.basename(video_path), {
                        "status": "success",
                        "output_dir": video_output_dir
                    }
                else:
                    return os.path.basename(video_path), {
                        "status": "failed",
                        "error": "处理失败"
                    }
            except Exception as e:
                logger.error(f"处理视频时出错: {video_path} - {e}")
                return os.path.basename(video_path), {
                    "status": "failed",
                    "error": str(e)
                }
        
        # 准备并发处理任务
        tasks = [(i, len(video_files), path) for i, path in enumerate(video_files)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    video_name, result = future.result()
                    results[video_name] = result
                except Exception as e:
                    logger.error(f"视频处理任务失败: {e}")
        
        return results

    def finalize_processing(self, results, output_dir):
        """完成处理并生成报告"""
        # 记录结束时间
        self.end_time = time.time()
        
        # 保存处理结果摘要
        with open(os.path.join(output_dir, "processing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 计算并显示总处理时间
        total_duration = self.end_time - self.start_time
        formatted_duration = format_duration(total_duration)
        print_progress(f"所有视频处理完成，总处理时间: {formatted_duration}", True)
        print_progress(f"结果保存在 {output_dir}", True)
        print_progress(f"详细日志已保存到 {self.log_file}", True)
        
        # 生成HTML报告
        report_path = self.generate_html_report(results, output_dir)
        print_progress(f"HTML报告已生成: {report_path}", True)

    def generate_html_report(self, results: Dict, output_dir: str):
        """生成HTML处理报告，包含性能统计信息"""
        html_path = os.path.join(output_dir, "processing_report.html")
        
        # 计算总处理时间
        if hasattr(self, 'start_time') and hasattr(self, 'end_time'):
            total_duration = self.end_time - self.start_time
            formatted_duration = format_duration(total_duration)
            
            # 格式化开始和结束时间
            start_time_str = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted_duration = "未记录"
            start_time_str = "未记录"
            end_time_str = "未记录"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>视频切分处理报告</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .video-item {{
                    margin-bottom: 20px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .success {{
                    border-left: 5px solid #4CAF50;
                }}
                .failed {{
                    border-left: 5px solid #F44336;
                }}
                .summary {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #e8f5e9;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .time-info {{
                    background-color: #e3f2fd;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .performance-info {{
                    background-color: #fff8e1;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .tips {{
                    background-color: #f3e5f5;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>视频切分处理报告</h1>
                
                <div class="time-info">
                    <h3>处理时间信息</h3>
                    <p>开始时间: {start_time_str}</p>
                    <p>结束时间: {end_time_str}</p>
                    <p>总处理时长: {formatted_duration}</p>
                </div>
                
                <div class="summary">
                    <h2>处理摘要</h2>
                    <p>总视频数: {len(results)}</p>
                    <p>成功处理: {sum(1 for r in results.values() if r['status'] == 'success')}</p>
                    <p>处理失败: {sum(1 for r in results.values() if r['status'] == 'failed')}</p>
                </div>
                
                <h2>视频处理详情</h2>
        """
        
        # 添加每个视频的处理详情
        for video_name, result in results.items():
            status_class = "success" if result["status"] == "success" else "failed"
            html_content += f"""
                <div class="video-item {status_class}">
                    <h3>{video_name}</h3>
                    <p>处理状态: {result["status"]}</p>
            """
            
            if result["status"] == "success":
                # 读取分段信息
                segments_file = os.path.join(result["output_dir"], "segments.json")
                if os.path.exists(segments_file):
                    try:
                        with open(segments_file, "r", encoding="utf-8") as f:
                            segments = json.load(f)
                        
                        html_content += """
                        <h4>切分片段:</h4>
                        <table>
                            <tr>
                                <th>序号</th>
                                <th>主题</th>
                                <th>开始时间</th>
                                <th>结束时间</th>
                                <th>时长(秒)</th>
                            </tr>
                        """
                        
                        for i, segment in enumerate(segments):
                            start = segment["start"]
                            end = segment["end"]
                            duration = end - start
                            html_content += f"""
                            <tr>
                                <td>{i+1}</td>
                                <td>{segment["topic"]}</td>
                                <td>{self.format_time(start)}</td>
                                <td>{self.format_time(end)}</td>
                                <td>{duration:.2f}</td>
                            </tr>
                            """
                        
                        html_content += """
                        </table>
                        """
                    except Exception as e:
                        html_content += f"<p>无法读取分段信息: {e}</p>"
                else:
                    html_content += "<p>未找到分段信息文件</p>"
                
                html_content += f"""
                    <p>输出目录: {result["output_dir"]}</p>
                """
            else:
                html_content += f"""
                    <p>错误信息: {result.get("error", "未知错误")}</p>
                """
            
            html_content += """
                </div>
            """
        
        # 添加优化建议
        html_content += """
                <div class="tips">
                    <h3>处理长视频的优化提示</h3>
                    <ul>
                        <li>此脚本已针对长视频进行了优化，自动调整处理参数</li>
                        <li>超过1小时的视频会增加帧间隔和文本块大小以加速处理</li>
                        <li>如需进一步提高性能，可考虑使用GPU加速的视频和图像处理</li>
                        <li>增加并行处理的线程数可以提高性能，但也会增加内存占用</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path

    def format_time(self, seconds: float) -> str:
        """将秒数格式化为 HH:MM:SS 格式"""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        
    def clean_temp_dir(self):
        """清理临时目录中的文件，但保留目录"""
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {file_path} - {e}")

    def cleanup(self):
        """清理临时文件夹"""
        import shutil
        shutil.rmtree(self.temp_dir)
        print_progress("临时文件已清理", True)


def main():
    parser = argparse.ArgumentParser(description='基于语义内容的视频切分工具')
    parser.add_argument('--input_path', type=str, default='./original_text', help='输入视频文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./Result_folder_text', help='输出目录，用于保存切分后的视频片段')
    parser.add_argument('--project-id', type=str, help='Google Cloud 项目ID')
    parser.add_argument('--location', type=str, help='Google Cloud 区域')
    parser.add_argument('--whisper-model', type=str, default='base', help='Whisper模型大小 (tiny, base, small, medium, large)')
    parser.add_argument('--gemini-model', type=str, default='gemini-2.5-flash-preview-05-20', help='Gemini模型名称')
    
    args = parser.parse_args()
    user_name = input("请输入您的英文名（全部小写）：").lower()
    user_folder = os.path.join('user', user_name)
    if not os.path.exists(user_folder):
        print("请在user目录下创建自己的文件夹")
        return
    
    args.input_path = os.path.join(user_folder, 'original_text')
    args.output_dir = os.path.join(user_folder, 'Result_folder_text')
    
    start_time = time.time()
    
    # 创建输出目录（为了日志文件做准备）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 简化初始化信息
    print("\n===== 视频内容分析与切分工具 =====")
    print(f"输入目录: {args.input_path}")
    print(f"输出目录: {args.output_dir}")
    print("开始处理...\n")
    
    splitter = VideoSplitter(
        project_id=args.project_id,
        location=args.location,
        model_name=args.gemini_model,
        whisper_model=args.whisper_model,
        output_dir=args.output_dir
    )
    
    try:
        # 处理文件夹中的所有视频
        splitter.process_video_folder(args.input_path, args.output_dir)
        
        elapsed_time = time.time() - start_time
        formatted_duration = format_duration(elapsed_time)
        print_progress(f"\n总处理时间: {formatted_duration}", True)
    finally:
        # 清理临时文件
        splitter.cleanup()

if __name__ == "__main__":
    main()

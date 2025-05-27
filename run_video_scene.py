import os
import sys
import time
import json
from datetime import datetime
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import shutil
from google import genai
from google.genai import types
import cv2
import base64
import re # 导入 re 模块

key_path = "./your_key_path.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    
# Gemini API 配置
def setup_gemini_client(project_id="your_project_id", location="your_location"):
    """
    设置并返回 Gemini API 客户端
    """
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    
    return client

# 使用 Gemini 为视频片段打标签
def label_video_with_gemini(client, video_path):
    """
    使用 Gemini API 对视频进行标注
    """
    try:
        # 直接读取视频文件
        with open(video_path, "rb") as f:
            video_data = f.read()
        
        # 创建视频和文本部分
        video_part = types.Part.from_bytes(
            data=video_data,
            mime_type="video/mp4",
        )
        
        # 定义提示词
        prompt = " Your_labeling_prompt "
        
        text_part = types.Part.from_text(text=prompt)
        
        # 创建请求内容
        contents = [
            types.Content(
                role="user",
                parts=[
                    video_part,
                    text_part
                ]
            )
        ]
        
        # 配置生成参数
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            seed=0,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
        )
        
        # 指定模型
        model = "gemini-2.5-flash-preview-05-20" # 注意：这里可能需要根据实际可用的模型更新
        
        # 发送请求并获取响应
        response_text = ""
        # 使用 client.generate_content 而不是 client.models.generate_content_stream
        # 如果您的 genai 库版本较新，可能需要 client.get_generative_model(model).generate_content(...)
        # 这里假设 client.generate_content 是流式接口的替代或存在类似方法
        # 为了安全起见，我们先尝试原始的 stream 方法，如果报错再调整
        # response = client.generate_content( # 假设这是非流式接口
        #     model=f"projects/{client.project}/locations/{client.location}/models/{model}", # 完整模型路径
        #     contents=contents,
        #     generation_config=generate_content_config, # 参数名可能是 generation_config
        # )
        # response_text = response.text # 或者 response.candidates[0].content.parts[0].text

        # 恢复使用流式API，因为原始代码是这样写的
        for chunk in client.models.generate_content_stream( # 确保 client.models 存在且有此方法
            model=model, # 或者完整的模型路径如上
            contents=contents,
            config=generate_content_config, # 参数名可能是 generation_config
        ):
            if chunk.text:
                response_text += chunk.text
        
        print(f"标签: {response_text}")
        return response_text.strip()
        
    except Exception as e:
        print(f"Gemini API 调用失败: {e}")
        return "标签生成失败"

def format_time(seconds):
    """将秒数格式化为 HH:MM:SS 格式"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"

def format_duration(seconds):
    """将秒数格式化为 xx min, xx s 格式"""
    minutes, seconds = divmod(int(seconds), 60)
    if minutes > 0:
        return f"{minutes} min, {seconds} s"
    else:
        return f"{seconds} s"

def get_video_duration(video_path):
    """获取视频文件的时长（以秒为单位）"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return 0.0
        # 获取视频的帧率和帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算视频时长
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0 # 防止除以零
        # 释放视频对象
        cap.release()
        return duration
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return 0.0

def safe_filename(filename, max_length=200):
    """
    安全化文件名，替换特殊字符但保留更多有意义的字符，并控制长度
    对第三段（产品功能特点）进行特殊处理，限制在50字符内
    特别处理双引号和其他可能导致文件系统问题的字符
    """
    # 预处理：统一替换各种引号为普通引号或删除
    quote_replacements = {
        '"': '',  # 英文双引号直接删除
        '"': '',  # 中文左双引号删除
        '"': '',  # 中文右双引号删除
        "'": '',  # 英文单引号删除
        "'": '',  # 中文左单引号删除
        "'": '',  # 中文右单引号删除
        '《': '(',  # 书名号替换为括号
        '》': ')',
        '<': '(',  # 尖括号替换为括号
        '>': ')',
        '|': '-',  # 竖线替换为横线
        '*': '',   # 星号删除
        '?': '',   # 问号删除
        ':': '：', # 英文冒号替换为中文冒号
        '/': '-',  # 斜杠替换为横线
        '\\': '-', # 反斜杠替换为横线
    }
    
    # 应用引号和特殊字符替换
    for old_char, new_char in quote_replacements.items():
        filename = filename.replace(old_char, new_char)
    
    # 先处理第三段的长度限制
    parts = filename.split('-', 3)  # 分割为最多4段
    if len(parts) >= 3:
        # 第三段是产品功能特点描述，限制长度
        feature_part = parts[2]
        if len(feature_part) > 50:
            # 智能截断第三段
            truncated_feature = feature_part[:50]
            # 尝试在标点符号处截断
            for punct in ['，', '、', '；', '：', ' ']:
                last_punct = truncated_feature.rfind(punct)
                if last_punct > 35:  # 至少保留70%
                    truncated_feature = truncated_feature[:last_punct + 1]
                    break
            else:
                # 如果找不到合适的标点，直接截断
                truncated_feature = truncated_feature.rstrip()
                if len(truncated_feature) < len(feature_part):
                    truncated_feature += "..."
            
            parts[2] = truncated_feature
            filename = '-'.join(parts)
    
    # 允许的安全字符集合（扩展版本，但排除文件系统危险字符）
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_()[]（）【】，、；：？！·"
    
    result = ""
    for char in filename:
        if char.isalnum() or char in safe_chars or '\u4e00' <= char <= '\u9fff':  # 中文字符范围
            result += char
        elif char == '。':  # 将句号替换为逗号
            result += "，"
        elif char in ['\t', '\n', '\r']:  # 空白字符替换为空格
            result += " "
        else:
            result += "_"
    
    # 清理连续的下划线、空格和首尾的下划线/空格
    result = re.sub(r'_+', '_', result)  # 多个下划线合并为一个
    result = re.sub(r'\s+', ' ', result)  # 多个空格合并为一个
    result = result.strip('_ ')  # 去除首尾的下划线和空格
    
    # 控制整体文件名长度
    if len(result) > max_length:
        # 尝试在标点符号处截断，保持语义完整
        truncated = result[:max_length]
        # 查找最后一个合适的截断点（标点符号）
        for punct in ['，', '-', '、', '；', '：', ' ']:
            last_punct = truncated.rfind(punct)
            if last_punct > max_length * 0.7:  # 至少保留70%的内容
                truncated = truncated[:last_punct + 1]
                break
        else:
            # 如果找不到合适的标点，在空格处截断
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.7:
                truncated = truncated[:last_space]
            # 否则直接截断并添加省略号
            else:
                truncated = truncated.rstrip()
                if len(truncated) < len(result):
                    truncated += "..."
        
        result = truncated
    
    # 最终清理：确保没有以点结尾（避免与扩展名混淆）
    result = result.rstrip('.')
    
    # 如果结果为空或只有无效字符，返回默认值
    if not result or result.isspace():
        result = "labeled_video"
    
    return result

def generate_html_report(results, output_dir, start_time, end_time):
    """生成HTML处理报告"""
    html_path = os.path.join(output_dir, "processing_report.html")
    
    # 计算总处理时间
    total_duration = end_time - start_time
    formatted_duration = format_duration(total_duration)
    
    # 格式化开始和结束时间
    start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>视频场景切分处理报告</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>视频场景切分处理报告</h1>
            
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
            # 读取场景信息
            if "scenes" in result:
                scenes = result["scenes"]
                
                html_content += """
                <h4>切分片段:</h4>
                <table>
                    <tr>
                        <th>序号</th>
                        <th>标签</th>
                        <th>文件名</th>
                        <th>时长(秒)</th>
                    </tr>
                """
                
                for i, scene in enumerate(scenes):
                    # 确保duration是浮点数
                    try:
                        duration = scene.get("duration", 0.0)
                        if isinstance(duration, str) and duration != "未知":
                            duration = float(duration)
                        elif duration == "未知": # 保持 "未知"
                            pass 
                        else:
                            duration = float(duration)
                            
                        # 格式化时长显示
                        if isinstance(duration, (int, float)):
                            formatted_duration_scene = f"{duration:.2f}"
                        else:
                            formatted_duration_scene = duration # 应该是 "未知"
                    except (ValueError, TypeError):
                        formatted_duration_scene = "未知"
                    
                    html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{scene.get("label", "未标注")}</td>
                        <td>{scene.get("filename", "未知")}</td>
                        <td>{formatted_duration_scene}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            html_content += f"""
                <p>输出目录: {result.get("output_dir", "未知")}</p>
            """
        else:
            html_content += f"""
                <p>错误信息: {result.get("error", "未知错误")}</p>
            """
        
        html_content += """
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {html_path}")

def detect_and_split_scenes(input_folder, output_folder, threshold=30.0, min_scene_length=15, project_id="idc-ipc", location="global"):
    """
    检测视频场景，切割视频并使用 Gemini 进行标注
    """
    start_time = time.time()

    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在：{input_folder}")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        gemini_client = setup_gemini_client(project_id, location)
        print("API 客户端设置成功")
    except Exception as e:
        print(f"API 客户端设置失败: {e}")
        gemini_client = None

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    results = {}

    for filename in os.listdir(input_folder):
        if filename.startswith('.') or filename.startswith('._'):
            print(f"跳过隐藏或元数据文件: {filename}")
            continue

        video_path = os.path.join(input_folder, filename)
        print(f"正在处理视频文件：{video_path}")

        original_base_name_for_output_dir = os.path.splitext(filename)[0]
        # 清理用于ffmpeg的名称 (与原脚本一致)
        sanitized_ffmpeg_name = re.sub(r'[^\w\-_.]', '_', original_base_name_for_output_dir)
        sanitized_ffmpeg_name = sanitized_ffmpeg_name[:60]
        if not sanitized_ffmpeg_name:
            sanitized_ffmpeg_name = "untitled_video"

        result_entry = {
            "status": "failed",
            "error": "",
            "scenes": []
        }
        
        # 定义视频输出文件夹路径，但创建操作移到 try 块内
        video_output_folder = os.path.join(output_folder, original_base_name_for_output_dir)
        result_entry["output_dir"] = video_output_folder # 记录预期的输出目录

        if not os.path.isfile(video_path):
            print(f"文件不存在或无法访问: {video_path}")
            result_entry["error"] = "文件不存在或无法访问"
            results[filename] = result_entry
            continue

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print(f"文件大小为0，跳过处理: {video_path}")
            result_entry["error"] = "文件大小为0"
            results[filename] = result_entry
            continue

        print(f"开始处理视频: {video_path} (大小: {file_size/1024/1024:.2f} MB)")

        video_manager = None
        try:
            # ---- 关键改动：将目录创建和检查移入 try 块 ----
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)
                print(f"已创建视频输出文件夹: {video_output_folder}")
            elif not os.path.isdir(video_output_folder):
                # 如果路径存在但不是一个目录，这是一个错误情况
                raise OSError(f"错误: 路径 {video_output_folder} 已存在但不是一个目录。")
            # ---- 目录创建逻辑结束 ----

            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_length))
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            
            scenes_data = []

            if len(scene_list) == 0:
                print(f"未能在视频 {filename} 中检测到任何场景，将处理整个视频。")
                temp_folder = os.path.join(output_folder, "temp_" + sanitized_ffmpeg_name)
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                temp_video_filename = sanitized_ffmpeg_name + os.path.splitext(filename)[1]
                temp_video_path = os.path.join(temp_folder, temp_video_filename)
                shutil.copy2(video_path, temp_video_path)
                video_duration = get_video_duration(temp_video_path)

                raw_gemini_label = "未标注"

                if gemini_client:
                    raw_gemini_label = label_video_with_gemini(gemini_client, temp_video_path)

                # 新的命名规则：直接使用完整标签，但控制长度
                safe_label = safe_filename(raw_gemini_label, max_length=180)
                new_filename = f"完整切片-{safe_label}.mp4"
                new_path = os.path.join(video_output_folder, new_filename)
                shutil.copy2(temp_video_path, new_path)
                print(f"已标注并保存完整视频: {new_path}")
                scenes_data.append({"label": raw_gemini_label, "filename": new_filename, "duration": video_duration})
                shutil.rmtree(temp_folder)
            else: # len(scene_list) > 0
                print(f"在视频 {filename} 中检测到 {len(scene_list)} 个场景。")
                temp_folder = os.path.join(output_folder, "temp_" + sanitized_ffmpeg_name)
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                try:
                    split_video_ffmpeg(
                        video_path,
                        scene_list,
                        output_file_template=os.path.join(temp_folder, f'{sanitized_ffmpeg_name}-Scene-$SCENE_NUMBER.mp4'),
                        video_name=sanitized_ffmpeg_name
                    )
                    print(f"视频 {filename} 的场景已切割并暂存到 {temp_folder}。")
                    if not any(f.endswith('.mp4') for f in os.listdir(temp_folder)):
                         raise RuntimeError(f"FFmpeg did not produce any .mp4 files in {temp_folder} for {filename}")
                except Exception as e:
                    print(f"FFmpeg 切割视频 {filename} 失败: {e}")
                    # result_entry["error"] 已在主 try-except 中处理，这里可以补充信息或直接抛出
                    raise RuntimeError(f"FFmpeg切割失败: {e}") # 重新抛出，由外层捕获

                scene_files_in_temp = sorted([f for f in os.listdir(temp_folder) if f.startswith(sanitized_ffmpeg_name) and f.endswith('.mp4')])

                for i, scene_obj in enumerate(scene_list):
                    if i < len(scene_files_in_temp):
                        scene_file = scene_files_in_temp[i]
                        scene_path = os.path.join(temp_folder, scene_file)
                        scene_duration = get_video_duration(scene_path)

                        raw_gemini_label = "未标注"

                        if gemini_client:
                            raw_gemini_label = label_video_with_gemini(gemini_client, scene_path)

                        # 新的命名规则：片段XXX + 完整标签，智能控制长度
                        scene_num_str = f"{i+1:03d}"
                        safe_label = safe_filename(raw_gemini_label, max_length=180)
                        new_filename = f"片段{scene_num_str}-{safe_label}.mp4"
                        new_path = os.path.join(video_output_folder, new_filename)

                        try:
                            # 在复制前再次确保目标目录存在 (防御性编程)
                            if not os.path.isdir(video_output_folder):
                                print(f"警告: 目标目录 {video_output_folder} 在复制前似乎消失了。尝试重新创建。")
                                os.makedirs(video_output_folder, exist_ok=True) # exist_ok=True 避免已存在时报错

                            shutil.copy2(scene_path, new_path)
                            print(f"已标注并保存视频片段: {new_path}")
                            scenes_data.append({"label": raw_gemini_label, "filename": new_filename, "duration": scene_duration})
                        except Exception as copy_err:
                            print(f"无法复制/重命名文件 {scene_path} 到 {new_path}: {copy_err}")
                            scenes_data.append({"label": raw_gemini_label + " (保存失败)", "filename": scene_file + " (未重命名)", "duration": scene_duration})
                        time.sleep(1)
                    else:
                        print(f"警告: 场景 {i+1} 对应的切片文件未在 {temp_folder} 中找到。")
                
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                print(f"视频 {filename} 的标注完成，结果保存到 {video_output_folder}。")

            result_entry["status"] = "success"
            result_entry["scenes"] = scenes_data
            # output_dir 已经在 result_entry 初始化时设置
            # result_entry["output_dir"] = video_output_folder 

        except Exception as e:
            print(f"处理视频 {filename} 时出错：{e}")
            result_entry["error"] = str(e)
            # output_dir 已经在 result_entry 初始化时设置
            # result_entry["output_dir"] = video_output_folder # 记录尝试的输出目录
        finally:
            if video_manager:
                video_manager.release()
        
        results[filename] = result_entry


    end_time = time.time()
    generate_html_report(results, output_folder, start_time, end_time) # 确保此函数已定义并可用

    summary_path = os.path.join(output_folder, "processing_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"无法保存处理结果摘要到 {summary_path}: {e}")

    total_duration_seconds = end_time - start_time
    formatted_total_duration = format_duration(total_duration_seconds) 
    print(f"所有视频处理完成，总处理时间: {formatted_total_duration}")
    print(f"结果保存在 {output_folder}")



if __name__ == "__main__":
    user_name = input("请输入您的英文名（全部小写）：").lower()
    user_folder = os.path.join('user', user_name)

    if not os.path.exists(user_folder):
        print(f"错误：用户文件夹 {user_folder} 不存在。请在 'user' 目录下创建您的文件夹。")
        # 尝试创建用户文件夹，如果用户希望如此
        try:
            os.makedirs(user_folder, exist_ok=True)
            print(f"已创建用户文件夹: {user_folder}")
        except OSError as e:
            print(f"无法创建用户文件夹 {user_folder}: {e}")
            exit()


    input_folder = os.path.join(user_folder, 'original_scene')
    output_folder = os.path.join(user_folder, 'Result_folder_scene')

    # 自动创建输入和输出文件夹（如果不存在）
    if not os.path.exists(input_folder):
        try:
            os.makedirs(input_folder)
            print(f"已创建输入文件夹: {input_folder}")
            print(f"请将需要处理的视频文件放入 {input_folder} 中。")
            # 如果输入文件夹是新创建的，可能没有文件，可以考虑退出或提示用户
            # exit() 
        except OSError as e:
            print(f"无法创建输入文件夹 {input_folder}: {e}")
            exit()
            
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"已创建输出文件夹: {output_folder}")
        except OSError as e:
            print(f"无法创建输出文件夹 {output_folder}: {e}")
            exit()

    print(f"input_folder is {input_folder}")
    print(f"output_folder is {output_folder}")
    
    project_id = "idc-ipc"
    location = "us-central1"
    
    detect_and_split_scenes(input_folder, output_folder, project_id=project_id, location=location)

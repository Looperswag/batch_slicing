# 智能视频分析与切分工具集 🎬✨✂️

本项目包含两款强大的 Python 脚本，旨在利用先进的 AI 技术自动化视频内容的分析、切分与标注，从而极大地提升视频素材管理和再利用的效率。

1.  **`run_video_scene.py` (基于视觉场景切分)**:
    *   适合对短视频、广告视频进行切片。
    *   通过分析视频画面的视觉变化，自动检测场景切换点。
    *   将视频按检测到的场景切分成独立的片段。
    *   利用 Google Gemini 多模态模型为每个**场景片段**生成描述性标签（侧重于画面内容）。

2.  **`run_video_text.py` (基于语义内容切分)**:
    *   适合对长视频、直播进行切片。
    *   结合语音识别 (OpenAI Whisper) 和关键帧提取技术。
    *   分析视频的语音文本和关键帧图像，理解其**语义主题**。
    *   根据识别到的主题变化点，将视频切分成具有连贯语义的片段。
    *   利用 Google Gemini 多模态模型为每个**语义片段**生成包含人物、产品功能、场景的详细标签。
    *   特别优化了对长视频的处理流程和性能。

两款脚本都旨在为内容创作者、营销团队和视频编辑提供高效的自动化工具。

## 🌟 核心功能

### 通用功能:
*   **多模态 AI 标注**: 集成 Google Gemini API，进行智能内容理解和标签生成。
*   **用户友好交互**: 启动时提示输入用户名，为不同用户创建独立的输入和输出文件夹结构。
*   **自动化文件夹管理**: 自动创建所需的输入和输出文件夹。
*   **详细处理报告**:
    *   生成 HTML 格式的处理报告 (`processing_report.html`)，直观展示处理结果。
    *   生成 JSON 格式的处理摘要 (`processing_summary.json`)，方便程序化分析。
*   **FFmpeg 依赖**: 使用 FFmpeg 进行核心的视频切分操作。

### `run_video_scene.py` (视觉场景切分) 特有功能:
*   **自动场景检测**: 使用 `PySceneDetect` 库，基于内容变化（如镜头切换、画面突变）检测场景。
*   **直接视频标注**: 主要依据视频片段的整体视觉内容进行标注。
*   **安全文件名生成**: 对 Gemini 生成的标签进行处理，创建合法且易读的文件名，并智能控制长度。
*   **`PySceneDetect`库的threshold参数**: 一般来说30符合大部分短视频投放场景，越低越敏感【切的越细】，越高约迟钝【切的越粗】。

### `run_video_text.py` (语义内容切分) 特有功能:
*   **语音转文字 (ASR)**: 使用 OpenAI Whisper 模型【后续可以更新openai最新的ASR模型】将视频中的语音精准转录为带时间戳的文本。
*   **关键帧提取**: 智能提取能代表视频片段内容的视觉关键帧。
*   **语义主题识别**: 结合文本和图像信息，利用 Gemini 模型分析并识别视频内容的主题变化。
*   **长视频优化**: 针对长视频在音频提取、转写、帧提取等环节进行了性能优化（如分段处理、调整参数）。
*   **批量 API 调用**: 高效地批量处理对 Gemini API 的请求，带有重试机制。
*   **精细化日志系统**: 将详细的调试和处理信息记录到日志文件中，控制台输出简洁进度。
*   **内容缓存**: 对相似的图像和文本分析结果进行缓存，减少重复 API 调用。
*   **命令行参数配置**: 支持通过命令行参数配置输入/输出路径、模型名称等。

## 🛠️ 技术栈

*   **Python 3.x**
*   **Google Generative AI SDK (`google-generai`)**: 与 Google Gemini API 交互。
*   **FFmpeg**: 核心视频处理和切分工具 (需单独安装)。
*   **OpenCV (`cv2`)**: 视频帧读取、元数据获取、图像处理。

### `run_video_scene.py` 特定库:
*   **PySceneDetect (`scenedetect`)**: 视频场景检测。

### `run_video_text.py` 特定库:
*   **OpenAI Whisper (`whisper`)**: 语音识别。
*   **Pydub**: 音频处理和分段。
*   **Numpy**: 数值计算，辅助处理。
*   `argparse`: 命令行参数解析。
*   `concurrent.futures`: 并行处理任务。

## ⚙️ 环境准备

1.  **Python 环境**:
    确保已安装 Python 3.7 或更高版本。

2.  **安装 FFmpeg**:
    两个脚本都依赖 FFmpeg。请根据您的操作系统从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并安装，确保 `ffmpeg` 命令在系统的 PATH 环境变量中可用。
    *   **Windows**: 下载预编译包，解压并将 `bin` 目录添加到系统 PATH。
    *   **macOS**: 可以使用 Homebrew: `brew install ffmpeg`
    *   **Linux**: 通常可以通过包管理器安装: `sudo apt update && sudo apt install ffmpeg`

3.  **Google Cloud 项目和 Vertex AI API**:
    *   目前key挂在IDC aigc下，如果后续迁移至DAM或者DOM最好换掉。
    *   创建一个服务账号 (Service Account) 并下载其 JSON 密钥文件。

4.  **Python 依赖库**:
    在项目根目录下，根据您要使用的脚本安装对应的依赖：

    *   **为 `run_video_scene.py` 安装依赖**:
        ```bash
        pip install scenedetect[opencv] google-generativeai
        ```

    *   **为 `run_video_text.py` 安装依赖**:
        ```bash
        pip install openai-whisper pydub google-generativeai opencv-python numpy
        ```
        (注意: `google-generativeai` 和 `opencv-python` 与上面的有重叠，如果同时使用两个脚本，只需安装一次。)

    *   **或者，为两个脚本一次性安装 (推荐)**:
        创建一个 `requirements.txt` 文件 (如果项目提供了，则直接使用):
        ```txt
        scenedetect[opencv]
        google-generativeai
        openai-whisper
        pydub
        opencv-python
        numpy
        ```
        然后运行:
        ```bash
        pip install -r requirements.txt
        ```

## 🚀 安装与配置

1.  **克隆或下载项目**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    或者直接下载 `run_video_scene.py` 和 `run_video_text.py` 文件。

2.  **配置 Google Cloud 认证**:
    *   将您下载的服务账号 JSON 密钥文件（例如 `your-service-account-key.json`）放置在脚本所在的目录。
    *   **重要**: 两个脚本都通过以下方式引用密钥文件。请确保 `key_path` 指向您的实际文件名：
        ```python
        # 在 run_video_scene.py 和 run_video_text.py 脚本顶部附近
        key_path = "./your-service-account-key.json" # 修改这里，确保与您的文件名一致
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        ```

3.  **模型和 API 参数配置 (可选)**:
    *   **`run_video_scene.py`**:
        *   可以在 `setup_gemini_client` 函数和 `detect_and_split_scenes` 函数调用中修改 `project_id` 和 `location`。
        *   可以在 `label_video_with_gemini` 函数中修改 `prompt` 和 Gemini `model`。
        *   可以在 `detect_and_split_scenes` 函数中调整 `threshold` (场景检测灵敏度) 和 `min_scene_length` (最小场景帧数)。
    *   **`run_video_text.py`**:
        *   **命令行参数**: 推荐通过命令行参数进行配置 (详见“使用方法”部分)。
        *   脚本内默认值: 可以在 `VideoSplitter` 类的 `__init__` 方法或 `main` 函数中修改 `project_id`, `location`, `gemini-model`, `whisper-model` 的默认值。
        *   可以在 `analyze_segment` 或 `identify_topic_changes` 方法中修改 Gemini 的 `prompt`。

## 🏃‍♂️ 使用方法

两个脚本都会提示输入用户英文名，并据此创建用户专属的文件夹结构。假设您输入 `testuser`。

### 预期文件结构

    ```
    .
    ├── run_video_scene.py
    ├── run_video_text.py
    ├── your-service-account-key.json
    └── user/
        └── testuser/
            ├── original_scene/ <-- run_video_scene.py 的输入视频
            ├── Result_folder_scene/ <-- run_video_scene.py 的输出结果
            ├── original_text/ <-- run_video_text.py 的输入视频
            └── Result_folder_text/ <-- run_video_text.py 的输出结果
    ```

### 1. 使用 `run_video_scene.py` (基于视觉场景切分)

1.  **创建用户目录和输入目录**:
    如果 `user/testuser/original_scene/` 目录不存在，脚本会尝试创建它。

2.  **放置视频文件**:
    将需要按视觉场景切分的视频文件放入 `user/testuser/original_scene/` 文件夹中。

3.  **运行脚本**:
    打开终端，导航到脚本所在目录，然后运行：
    ```bash
    python run_video_scene.py
    ```
    脚本会提示您输入英文名。

4.  **查看结果**:
    处理完成后，切分并标注好的视频片段将保存在 `user/testuser/Result_folder_scene/<original_video_name>/` 目录下。
    同时，在 `user/testuser/Result_folder_scene/` 目录下会生成 `processing_report.html` 和 `processing_summary.json`。

### 2. 使用 `run_video_text.py` (基于语义内容切分)

1.  **创建用户目录和输入目录**:
    如果 `user/testuser/original_text/` 目录不存在，脚本会提示用户创建（或您需要手动创建）。

2.  **放置视频文件**:
    将需要按语义内容切分的视频文件放入 `user/testuser/original_text/` 文件夹中。

3.  **运行脚本**:
    打开终端，导航到脚本所在目录。此脚本使用命令行参数进行配置。
    基本运行方式（使用默认参数，但指定用户路径）：
    ```bash
    python run_video_text.py
    ```
    脚本会提示输入英文名，然后 `input_path` 会被设置为 `user/yourusername/original_text`，`output_dir` 会被设置为 `user/yourusername/Result_folder_text`。

    自定义参数运行示例：
    ```bash
    python run_video_text.py \
        --project-id "your-gcp-project-id" \
        --location "your-location" \
        --whisper-model "base" \
        --gemini-model "your-latest-gemini-model-name"
    ```
    (脚本内部会处理用户文件夹路径的拼接)

    **常用命令行参数**:
    *   `--input_path`: (通常由脚本根据用户名自动设置) 输入视频文件夹路径。
    *   `--output_dir`: (通常由脚本根据用户名自动设置) 输出目录。
    *   `--project-id`: Google Cloud 项目ID
    *   `--location`: Google Cloud 区域 (默认: `global`)。
    *   `--whisper-model`: Whisper 模型大小 (可选: `tiny`, `base`, `small`, `medium`, `large`; 默认: `base`)。
    *   `--gemini-model`: Gemini 模型名称 (默认: `gemini-2.5-flash-preview-05-20`)。

4.  **查看结果**:
    处理完成后，切分并标注好的视频片段将保存在 `user/testuser/Result_folder_text/<original_video_name>/` 目录下。每个视频会有一个子文件夹。
    在该子文件夹内，除了视频片段，还会有一个 `segments.json` 文件，记录了每个片段的主题和时间戳。
    在 `user/testuser/Result_folder_text/` 目录下会生成全局的 `processing_report.html` 和 `processing_summary.json`，以及一个详细的日志文件 `video_processing_YYYYMMDD_HHMMSS.log`。

## 📄 输出说明

### `run_video_scene.py` 输出:
*   **切分的视频片段**:
    *   位于 `user/<username>/Result_folder_scene/<原始视频名>/`。
    *   命名格式:
        *   `片段XXX-<性别标签>-<产品标签>-<功能特点描述>-<场景描述>.mp4`
        *   `完整切片-<性别标签>-<产品标签>-<功能特点描述>-<场景描述>.mp4` (若未检测到场景)
*   **HTML 报告 (`processing_report.html`)**: 位于 `user/<username>/Result_folder_scene/`。
*   **JSON 摘要 (`processing_summary.json`)**: 位于 `user/<username>/Result_folder_scene/`。

### `run_video_text.py` 输出:
*   **切分的视频片段**:
    *   位于 `user/<username>/Result_folder_text/<原始视频名>/`。
    *   命名格式: `XX_<主题描述>.mp4` (XX 是序号)。
*   **分段信息 (`segments.json`)**: 位于每个视频的输出子目录 `user/<username>/Result_folder_text/<原始视频名>/`，包含各片段的主题、起止时间。
*   **HTML 报告 (`processing_report.html`)**: 位于 `user/<username>/Result_folder_text/`。
*   **JSON 摘要 (`processing_summary.json`)**: 位于 `user/<username>/Result_folder_text/`。
*   **日志文件 (`video_processing_*.log`)**: 位于 `user/<username>/Result_folder_text/`，记录详细处理过程。

## 🔍 核心逻辑概览

### `run_video_scene.py` (视觉场景切分)
1.  **用户输入与路径设置**: 获取用户名，构建输入输出路径。
2.  **Gemini 客户端初始化**。
3.  **遍历视频**:
    *   **场景检测 (`PySceneDetect`)**: 使用 `ContentDetector` 分析视频帧，找出场景切换点。
    *   **视频切分 (`FFmpeg`)**:
        *   若无场景，复制整个视频。
        *   若有场景，调用 `split_video_ffmpeg` 将视频按场景列表切分到临时目录。
    *   **片段标注 (`Gemini`)**: 对每个切分出的（或完整的）视频片段，调用 `label_video_with_gemini`，发送视频给 Gemini API 并获取标签。
    *   **重命名与保存**: 使用 `safe_filename` 清理标签并作为文件名，将片段从临时目录复制到最终输出目录。
4.  **生成报告**: 创建 HTML 和 JSON 报告。

### `run_video_text.py` (语义内容切分)
1.  **参数解析与路径设置**: 处理命令行参数，获取用户名，构建路径。
2.  **`VideoSplitter` 类初始化**: 加载 Whisper 模型，初始化 Gemini 客户端，设置日志。
3.  **遍历视频 (可并行或串行)**:
    *   **元数据获取**: 使用 OpenCV 获取视频时长、帧率等，并据此调整后续处理参数（特别是针对长视频）。
    *   **音频提取 (`FFmpeg`)**: 从视频中提取音频流，优化长视频的提取参数。
    *   **音频转文字 (`Whisper`)**: 将音频转录为带时间戳的文本，长音频会分段处理。
    *   **关键帧提取 (`OpenCV`)**: 按一定时间间隔提取视频帧，优化长视频的帧间隔和分辨率。
    *   **语义分析与主题识别 (`Gemini`)**:
        *   将转录文本按一定时长（如30秒）分块。
        *   为每个文本块匹配最近的关键帧。
        *   将文本块和对应关键帧图像批量发送给 Gemini API (`batch_gemini_requests`)，获取该片段的语义标签/主题。此步骤包含缓存和重试逻辑。
        *   比较相邻片段的主题，识别主题变化点，形成初步的语义分段。
        *   合并过短的语义分段。
    *   **视频切分 (`FFmpeg`)**: 根据最终确定的语义分段信息，使用 FFmpeg 将原始视频切分成多个片段，并以主题命名。此步骤可并行处理。
4.  **生成报告**: 创建 HTML、JSON 报告，并记录总处理时间。
5.  **清理**: 删除临时文件和目录。

## ⚠️ 注意事项
*   **FFmpeg 依赖**: 脚本的视频切分功能强依赖于正确安装并配置的 FFmpeg。
*   **Whisper 模型下载**: `run_video_text.py` 首次运行时，Whisper 会自动下载指定大小的模型文件，可能需要一些时间。
*   **处理时长与资源**: 视频处理（尤其是包含语音识别和多次 API 调用的 `run_video_text.py`）可能非常耗时且占用较多 CPU/内存资源。长视频处理建议在性能较好的机器上运行。
*   **文件路径与编码**: 脚本内部尽量使用 UTF-8 编码，但在处理包含非常特殊字符的文件名或路径时，仍需留意潜在的跨平台兼容性问题。


































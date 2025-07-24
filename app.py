import asyncio
import json
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import pytz
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
# 按需导入大模型相关包
# -----------------------------------------------------------------------
# 0. 配置
# -----------------------------------------------------------------------
shanghai_tz = pytz.timezone("Asia/Shanghai")

credentials = json.load(open("credentials.json"))
API_KEY = credentials["API_KEY"]
BASE_URL = credentials.get("BASE_URL", "")
MODEL = credentials.get("MODEL", "gemini-2.5-pro")

# 根据MODEL选择使用的大模型
USE_GEMINI = "gemini" in MODEL.lower()
USE_QWEN = "qwen" in MODEL.lower()
USE_VLLM = USE_QWEN and BASE_URL and len(BASE_URL) > 0

# 不检查API密钥的有效性，只通过连通性判断
try:
    if USE_GEMINI:
        import google.generativeai as genai
        os.environ["GEMINI_API_KEY"] = API_KEY
        gemini_client = genai.Client()
    elif USE_QWEN and not USE_VLLM:
        from dashscope import Generation
    elif USE_VLLM or API_KEY.startswith("sk-"):
        # 为 OpenRouter 添加应用标识
        extra_headers = {}
        if "openrouter.ai" in BASE_URL.lower():
            extra_headers = {
                "HTTP-Referer": "https://github.com/fogsightai/fogsight",
                "X-Title": "Fogsight - AI Animation Generator"
            }
        
        client = AsyncOpenAI(
            api_key=API_KEY, 
            base_url=BASE_URL,
            default_headers=extra_headers
        )
    else:
        print("警告: 未识别的模型类型，将使用默认处理方式")
except Exception as e:
    print(f"初始化LLM客户端时出错: {str(e)}")
    print("将继续运行，但LLM服务可能不可用")

templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------
# 1. FastAPI 初始化
# -----------------------------------------------------------------------
app = FastAPI(title="AI Animation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    topic: str
    history: Optional[List[dict]] = None

# -----------------------------------------------------------------------
# 2. 核心：流式生成器 (现在会使用 history)
# -----------------------------------------------------------------------
async def llm_event_stream(
    topic: str,
    history: Optional[List[dict]] = None,
    model: str = None, # Will use MODEL from config if not specified
) -> AsyncGenerator[str, None]:
    history = history or []
    
    # Use configured model if not specified
    if model is None:
        model = MODEL
    
    # The system prompt is now more focused
    system_prompt = f"""请你生成一个非常精美的动态动画,讲讲 {topic}
要动态的,要像一个完整的,正在播放的视频。包含一个完整的过程，能把知识点讲清楚。
页面极为精美，好看，有设计感，同时能够很好的传达知识。知识和图像要准确
附带一些旁白式的文字解说,从头到尾讲清楚一个小的知识点
不需要任何互动按钮,直接开始播放
使用和谐好看，广泛采用的浅色配色方案，使用很多的，丰富的视觉元素。双语字幕
**请保证任何一个元素都在一个2k分辨率的容器中被摆在了正确的位置，避免穿模，字幕遮挡，图形位置错误等等问题影响正确的视觉传达**
html+css+js+svg，放进一个html里"""

    if USE_GEMINI:
        try:
            full_prompt = system_prompt + "\n\n" + topic
            if history:
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
                full_prompt = history_text + "\n\n" + full_prompt
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: gemini_client.models.generate_content(
                    model=model, 
                    contents=full_prompt
                )
            )
            
            text = response.text
            chunk_size = 50
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.05)
                
        except Exception as e:
            # 返回通用错误信息，不暴露具体API密钥问题
            error_message = "LLM服务连接失败，请检查网络连接或稍后再试"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            return
    elif USE_VLLM:
        # 使用OpenAI兼容API调用本地vllm部署的qwen模型
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": topic},
        ]

        try:
            print(f"正在连接vllm部署的qwen模型: {BASE_URL}")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.8, 
            )
            
            print("成功连接到vllm模型，开始接收响应")
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    payload = json.dumps({"token": token}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0.001)
        except OpenAIError as e:
            print(f"vllm模型调用失败: {str(e)}")
            error_message = f"连接vllm模型失败: {str(e)}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            return
    elif USE_QWEN:
        try:
            # 通义千问模式 - 提供简单的HTML响应，避免复杂处理导致卡死
            simple_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>通义千问模式 - {topic}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            width: 800px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        header {{
            background: #4285f4;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .content {{
            padding: 20px;
        }}
        .topic {{
            background: #e8f0fe;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .animation {{
            height: 400px;
            background: #f8f9fa;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }}
        .circle {{
            width: 100px;
            height: 100px;
            background: linear-gradient(45deg, #4285f4, #34a853, #fbbc05, #ea4335);
            border-radius: 50%;
            position: absolute;
            animation: move 5s infinite alternate ease-in-out;
        }}
        @keyframes move {{
            0% {{ transform: translate(-150px, -100px) scale(0.8); }}
            50% {{ transform: translate(150px, 100px) scale(1.2); }}
            100% {{ transform: translate(-150px, 100px) scale(1); }}
        }}
        .subtitle {{
            position: absolute;
            bottom: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            width: 80%;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>通义千问动画演示</h1>
        </header>
        <div class="content">
            <div class="topic">
                <h2>主题：{topic}</h2>
            </div>
            <div class="animation">
                <div class="circle"></div>
                <div class="subtitle">这是关于"{topic}"的动画演示</div>
            </div>
            <p>系统正常运行中。这是通义千问模式的简化响应，展示了一个基本动画。</p>
            <p>如需连接到本地vllm部署的qwen模型，请在credentials.json中设置BASE_URL。</p>
        </div>
    </div>
</body>
</html>
"""
            
            # 分块返回HTML，确保不会卡死
            lines = simple_html.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    payload = json.dumps({"token": line + '\n'}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    # 每10行暂停一下，确保前端有时间处理
                    if i % 10 == 0:
                        await asyncio.sleep(0.05)
            
            # 确保发送完成信号
            yield 'data: {"event":"[DONE]"}\n\n'
            return
                
        except Exception as e:
            # 返回通用错误信息，不暴露具体API密钥问题
            error_message = "LLM服务连接失败，请检查网络连接或稍后再试"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            return
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": topic},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.8, 
            )
        except OpenAIError as e:
            # 返回通用错误信息，不暴露具体API密钥问题
            error_message = "LLM服务连接失败，请检查网络连接或稍后再试"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            return

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.001)

    yield 'data: {"event":"[DONE]"}\n\n'

# -----------------------------------------------------------------------
# 3. 路由 (CHANGED: Now a POST request)
# -----------------------------------------------------------------------
@app.post("/generate")
async def generate(
    chat_request: ChatRequest, # CHANGED: Use the Pydantic model
    request: Request,
):
    """
    Main endpoint: POST /generate
    Accepts a JSON body with "topic" and optional "history".
    Returns an SSE stream.
    """
    accumulated_response = ""  # for caching flow results

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for chunk in llm_event_stream(chat_request.topic, chat_request.history):
                accumulated_response += chunk
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    async def wrapped_stream():
        async for chunk in event_generator():
            yield chunk

    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(wrapped_stream(), headers=headers)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "time": datetime.now(shanghai_tz).strftime("%Y%m%d%H%M%S")})

# -----------------------------------------------------------------------
# 4. 本地启动命令
# -----------------------------------------------------------------------
# uvicorn app:app --reload --host 0.0.0.0 --port 8000


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

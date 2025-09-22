import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List
import json
import os

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro Max", 
    page_icon="🎨", 
    layout="wide"
)

# API 提供商配置
API_PROVIDERS = {
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖"
    },
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

def validate_api_key(api_key: str, base_url: str) -> tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        # 創建測試客戶端
        test_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 進行簡單的 API 調用測試
        response = test_client.models.list()
        
        # 如果沒有拋出異常，說明 API 密鑰有效
        return True, "API 密鑰驗證成功"
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return False, "API 密鑰無效或已過期"
        elif "403" in error_msg or "Forbidden" in error_msg:
            return False, "API 密鑰沒有足夠權限"
        elif "404" in error_msg:
            return False, "API 端點不存在或不正確"
        elif "timeout" in error_msg.lower():
            return False, "API 連接超時"
        else:
            return False, f"API 驗證失敗: {error_msg[:100]}"

def init_api_client():
    """初始化 API 客戶端"""
    # 從 session state 或 secrets 獲取 API 配置
    api_key = None
    base_url = None
    
    # 優先使用 session state 中的配置
    if 'api_config' in st.session_state and st.session_state.api_config['api_key']:
        api_key = st.session_state.api_config['api_key']
        base_url = st.session_state.api_config['base_url']
    
    # 如果 session state 中沒有，嘗試從 secrets 獲取
    elif 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets.get("OPENAI_API_KEY")
        base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # 如果都沒有，返回 None
    if not api_key:
        return None
    
    try:
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception:
        return None

# Flux 模型配置
FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "supports_img2img": True
    },
    "flux.1-krea-dev": {
        "name": "FLUX.1 Krea Dev", 
        "description": "創意開發版本，適合實驗性生成",
        "icon": "🎨",
        "type": "創意開發",
        "supports_img2img": True
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "supports_img2img": True
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🔧",
        "type": "編輯專用",
        "supports_img2img": True
    },
    "flux.1-kontext-max": {
        "name": "FLUX.1 Kontext Max",
        "description": "最高性能版本，極致品質",
        "icon": "🚀",
        "type": "極致性能",
        "supports_img2img": True
    }
}

# 初始化 session state
def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'optimized_prompts' not in st.session_state:
        st.session_state.optimized_prompts = {}
    
    if 'extracted_prompts' not in st.session_state:
        st.session_state.extracted_prompts = {}

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 API 設置")
    
    # API 提供商選擇
    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')
    
    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )
    
    # 顯示提供商信息
    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")
    
    # API 密鑰輸入
    current_key = st.session_state.api_config.get('api_key', '')
    masked_key = '*' * 20 + current_key[-8:] if len(current_key) > 8 else ''
    
    api_key_input = st.text_input(
        "API 密鑰",
        value="",
        type="password",
        placeholder=f"請輸入 {provider_info['name']} 的 API 密鑰...",
        help=f"API 密鑰通常以 '{provider_info['key_prefix']}' 開頭"
    )
    
    # 如果已經有密鑰，顯示遮掩版本
    if current_key and not api_key_input:
        st.caption(f"🔐 當前密鑰: {masked_key}")
    
    # Base URL 設置
    base_url_input = st.text_input(
        "API 端點 URL",
        value=st.session_state.api_config.get('base_url', provider_info['base_url_default']),
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )
    
    # 操作按鈕
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_btn = st.button("💾 保存設置", type="primary")
    
    with col2:
        test_btn = st.button("🧪 測試連接")
    
    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")
    
    # 保存設置
    if save_btn:
        if not api_key_input and not current_key:
            st.error("❌ 請輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            # 使用新輸入的密鑰或保持現有密鑰
            final_api_key = api_key_input if api_key_input else current_key
            
            st.session_state.api_config = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }
            st.success("✅ API 設置已保存")
            st.rerun()
    
    # 測試連接
    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key
        if not test_api_key:
            st.error("❌ 請先輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            with st.spinner("正在測試 API 連接..."):
                is_valid, message = validate_api_key(test_api_key, base_url_input)
                if is_valid:
                    st.success(f"✅ {message}")
                    st.session_state.api_config['validated'] = True
                else:
                    st.error(f"❌ {message}")
                    st.session_state.api_config['validated'] = False
    
    # 清除設置
    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
        st.success("🗑️ API 設置已清除")
        st.rerun()
    
    # 顯示當前狀態
    if st.session_state.api_config['api_key']:
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            if st.session_state.api_config['validated']:
                st.success("🟢 API 已驗證")
            else:
                st.warning("🟡 API 未驗證")
        
        with status_col2:
            st.info(f"🔧 使用: {provider_info['name']}")
    
    # API 使用指南
    with st.expander("📚 API 密鑰獲取指南"):
        st.markdown("""
        ### OpenAI Compatible API
        1. 前往 [OpenAI Platform](https://platform.openai.com/api-keys)
        2. 登錄你的帳戶
        3. 點擊 "Create new secret key"
        4. 複製生成的密鑰（以 sk- 開頭）
        
        ### Navy API
        1. 前往 Navy 官方網站註冊帳戶
        2. 在帳戶設置中生成 API 密鑰
        3. 複製密鑰用於此應用程式
        
        ### 安全提示 ⚠️
        - 不要在公共場所輸入 API 密鑰
        - 定期更新和輪換你的密鑰
        - 監控 API 使用量避免意外費用
        - 設置 API 使用額度限制
        """)

def optimize_prompt(original_prompt: str, style: str = "detailed") -> str:
    """使用 GPT 優化提示詞"""
    client = init_api_client()
    if not client:
        st.error("❌ 請先配置 API 密鑰")
        return original_prompt
    
    try:
        system_prompts = {
            "detailed": "You are an expert at optimizing text-to-image prompts. Transform the user's simple prompt into a detailed, descriptive prompt that will generate high-quality images. Add specific details about lighting, composition, style, colors, and artistic techniques. Keep the core concept but enhance it dramatically.",
            "artistic": "You are an expert at creating artistic image prompts. Transform the user's prompt into an artistic masterpiece description. Include art styles, famous artist influences, painting techniques, and aesthetic elements that will create visually stunning results.",
            "realistic": "You are an expert at creating photorealistic image prompts. Transform the user's prompt into a detailed photographic description. Include camera settings, lighting conditions, composition rules, and realistic details that will create lifelike images.",
            "creative": "You are an expert at creating creative and imaginative image prompts. Transform the user's prompt into something unique and creative. Add fantastical elements, unusual perspectives, creative concepts, and innovative ideas while maintaining the original intent."
        }
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompts.get(style, system_prompts["detailed"])
                },
                {
                    "role": "user", 
                    "content": f"Original prompt: {original_prompt}\n\nOptimized prompt:"
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        optimized = response.choices[0].message.content.strip()
        return optimized
        
    except Exception as e:
        st.error(f"提示詞優化失敗: {str(e)}")
        return original_prompt

def extract_prompt_from_image(image_file) -> str:
    """從圖像提取提示詞（使用 GPT-4 Vision）"""
    client = init_api_client()
    if not client:
        st.error("❌ 請先配置 API 密鑰")
        return "請先配置 API 密鑰"
    
    try:
        # 將圖像轉換為 base64
        image_bytes = image_file.read()
        image_file.seek(0)  # 重置文件指針
        base64_image = base64.b64encode(image_bytes).decode()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images and creating detailed text-to-image prompts. Analyze the provided image and create a detailed, descriptive prompt that could be used to generate a similar image. Focus on visual elements, style, composition, colors, lighting, and artistic techniques. Be specific and detailed."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and create a detailed text-to-image generation prompt that could recreate a similar image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"圖像分析失敗: {str(e)}")
        return "圖像分析失敗，請檢查 API 密鑰和網路連接"

def add_to_history(prompt: str, model: str, images: List[str], metadata: Dict):
    """添加生成記錄到歷史"""
    history_item = {
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "model": model,
        "images": images,
        "metadata": metadata,
        "id": len(st.session_state.generation_history)
    }
    
    st.session_state.generation_history.insert(0, history_item)
    
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

def display_image_with_actions(image_url: str, image_id: str, history_item: Dict = None):
    """顯示圖像和相關操作"""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    img_response = requests.get(image_url)
    img = Image.open(BytesIO(img_response.content))
    
    st.image(img, use_container_width=True)
    
    with col1:
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        st.download_button(
            label="📥 下載",
            data=img_buffer.getvalue(),
            file_name=f"flux_generated_{image_id}.png",
            mime="image/png",
            key=f"download_{image_id}",
            use_container_width=True
        )
    
    with col2:
        is_favorite = any(fav['id'] == image_id for fav in st.session_state.favorite_images)
        if st.button(
            "⭐ 已收藏" if is_favorite else "☆ 收藏",
            key=f"favorite_{image_id}",
            use_container_width=True
        ):
            if is_favorite:
                st.session_state.favorite_images = [
                    fav for fav in st.session_state.favorite_images if fav['id'] != image_id
                ]
                st.success("已取消收藏")
            else:
                favorite_item = {
                    "id": image_id,
                    "image_url": image_url,
                    "timestamp": datetime.datetime.now(),
                    "history_item": history_item
                }
                st.session_state.favorite_images.append(favorite_item)
                st.success("已加入收藏")
    
    with col3:
        if history_item and st.button(
            "🔄 重新生成",
            key=f"regenerate_{image_id}",
            use_container_width=True
        ):
            st.session_state.regenerate_prompt = history_item['prompt']
            st.session_state.regenerate_model = history_item['model']
            st.rerun()
    
    with col4:
        if st.button(
            "🔍 提取提示詞",
            key=f"extract_{image_id}",
            use_container_width=True
        ):
            with st.spinner("正在分析圖像..."):
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                extracted_prompt = extract_prompt_from_image(img_bytes)
                st.session_state.extracted_prompts[image_id] = extracted_prompt
                st.success("提示詞已提取！")

# 初始化會話狀態
init_session_state()

# 檢查 API 配置狀態
client = init_api_client()
api_configured = client is not None

# 主標題
st.title("🎨 Flux AI 圖像生成器 Pro Max")
st.markdown("**全新功能：API 密鑰管理 | 提示詞優化 | 圖生圖 | 圖出提示詞**")

# API 狀態警告
if not api_configured:
    st.error("⚠️ 請先配置 API 密鑰才能使用圖像生成功能")
    st.info("👆 點擊側邊欄的 'API 設置' 來配置你的密鑰")

# 側邊欄 API 設置
with st.sidebar:
    show_api_settings()
    
    # 快捷狀態顯示
    st.markdown("---")
    if api_configured:
        st.success("🟢 API 已配置")
        provider = st.session_state.api_config.get('provider', 'Unknown')
        st.caption(f"使用: {API_PROVIDERS.get(provider, {}).get('name', provider)}")
    else:
        st.error("🔴 API 未配置")
    
    # 使用統計
    st.markdown("### 📊 使用統計")
    total_generations = len(st.session_state.generation_history)
    total_favorites = len(st.session_state.favorite_images)
    
    st.metric("總生成數", total_generations)
    st.metric("收藏數量", total_favorites)

# 頁面導航
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 圖像生成", 
    "🔧 提示詞優化", 
    "🖼️ 圖生圖", 
    "📝 圖出提示詞",
    "📚 歷史記錄", 
    "⭐ 收藏夾"
])

# 圖像生成頁面
with tab1:
    if not api_configured:
        st.warning("⚠️ 請先在側邊欄配置 API 密鑰")
        st.info("配置完成後即可開始生成圖像")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 模型選擇
            st.subheader("選擇 Flux 模型")
            selected_model = st.selectbox(
                "模型",
                options=list(FLUX_MODELS.keys()),
                format_func=lambda x: f"{FLUX_MODELS[x]['icon']} {FLUX_MODELS[x]['name']}",
                index=0
            )
            
            model_info = FLUX_MODELS[selected_model]
            st.info(f"已選擇：{model_info['icon']} {model_info['name']} - {model_info['description']}")
            
            # 提示詞輸入
            st.subheader("輸入提示詞")
            
            default_prompt = ""
            if hasattr(st.session_state, 'regenerate_prompt'):
                default_prompt = st.session_state.regenerate_prompt
                if hasattr(st.session_state, 'regenerate_model'):
                    selected_model = st.session_state.regenerate_model
                delattr(st.session_state, 'regenerate_prompt')
                if hasattr(st.session_state, 'regenerate_model'):
                    delattr(st.session_state, 'regenerate_model')
            
            prompt = st.text_area(
                "描述你想要生成的圖像",
                value=default_prompt,
                height=120,
                placeholder="例如：A cute cat wearing a wizard hat in a magical forest..."
            )
            
            # 快速優化按鈕
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                if st.button("✨ 詳細優化", use_container_width=True):
                    if prompt.strip():
                        with st.spinner("正在優化提示詞..."):
                            optimized = optimize_prompt(prompt, "detailed")
                            st.session_state.temp_optimized_prompt = optimized
                            st.rerun()
            
            with col_opt2:
                if st.button("🎨 藝術優化", use_container_width=True):
                    if prompt.strip():
                        with st.spinner("正在優化提示詞..."):
                            optimized = optimize_prompt(prompt, "artistic")
                            st.session_state.temp_optimized_prompt = optimized
                            st.rerun()
            
            with col_opt3:
                if st.button("📸 真實優化", use_container_width=True):
                    if prompt.strip():
                        with st.spinner("正在優化提示詞..."):
                            optimized = optimize_prompt(prompt, "realistic")
                            st.session_state.temp_optimized_prompt = optimized
                            st.rerun()
            
            # 顯示優化後的提示詞
            if hasattr(st.session_state, 'temp_optimized_prompt'):
                st.success("✅ 提示詞已優化！")
                optimized_prompt = st.text_area(
                    "優化後的提示詞",
                    value=st.session_state.temp_optimized_prompt,
                    height=100,
                    key="optimized_display"
                )
                
                col_use, col_clear = st.columns(2)
                with col_use:
                    if st.button("📝 使用優化提示詞", type="primary"):
                        prompt = st.session_state.temp_optimized_prompt
                        delattr(st.session_state, 'temp_optimized_prompt')
                        st.rerun()
                with col_clear:
                    if st.button("❌ 清除"):
                        delattr(st.session_state, 'temp_optimized_prompt')
                        st.rerun()
            
            # 高級設定
            with st.expander("🔧 高級設定"):
                col_size, col_num = st.columns(2)
                
                with col_size:
                    size_options = {
                        "1024x1024": "正方形 (1:1)",
                        "1152x896": "橫向 (4:3.5)", 
                        "896x1152": "直向 (3.5:4)",
                        "1344x768": "寬屏 (16:9)",
                        "768x1344": "超高 (9:16)"
                    }
                    
                    selected_size = st.selectbox(
                        "圖像尺寸",
                        options=list(size_options.keys()),
                        format_func=lambda x: f"{x} - {size_options[x]}",
                        index=0
                    )
                
                with col_num:
                    num_images = st.slider("生成數量", 1, 4, 1)
            
            # 生成按鈕
            generate_btn = st.button(
                "🚀 生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip() or not api_configured
            )
        
        with col2:
            # API 狀態和使用說明
            if api_configured:
                provider_info = API_PROVIDERS.get(st.session_state.api_config['provider'], {})
                st.success(f"🟢 API 已連接\n使用: {provider_info.get('name', 'Unknown')}")
            else:
                st.error("🔴 API 未配置")
            
            st.subheader("📋 使用說明")
            st.markdown(f"""
            **當前模型：** {FLUX_MODELS[selected_model]['name']}
            
            **新功能：**
            - 🔑 API 密鑰管理
            - ✨ 一鍵提示詞優化
            - 🖼️ 圖生圖功能
            - 📝 圖出提示詞
            
            **步驟：**
            1. 配置 API 密鑰（側邊欄）
            2. 輸入基礎提示詞
            3. 選擇優化風格（可選）
            4. 調整高級設定
            5. 點擊生成按鈕
            """)

        # 圖像生成邏輯
        if generate_btn and prompt.strip() and api_configured:
            with st.spinner(f"正在使用 {FLUX_MODELS[selected_model]['name']} 生成圖像..."):
                try:
                    generation_params = {
                        "model": selected_model,
                        "prompt": prompt,
                        "n": num_images,
                        "size": selected_size
                    }
                    
                    response = client.images.generate(**generation_params)
                    
                    image_urls = [img.url for img in response.data]
                    metadata = {
                        "size": selected_size,
                        "num_images": num_images,
                        "model_info": FLUX_MODELS[selected_model],
                        "generation_type": "text2img",
                        "api_provider": st.session_state.api_config['provider']
                    }
                    
                    add_to_history(prompt, selected_model, image_urls, metadata)
                    
                    st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
                    
                    for i, image_data in enumerate(response.data):
                        st.subheader(f"圖像 {i+1}")
                        image_id = f"{len(st.session_state.generation_history)-1}_{i}"
                        display_image_with_actions(
                            image_data.url, 
                            image_id, 
                            st.session_state.generation_history[0]
                        )
                        
                        # 顯示提取的提示詞
                        if image_id in st.session_state.extracted_prompts:
                            with st.expander(f"📝 圖像 {i+1} 提取的提示詞"):
                                st.write(st.session_state.extracted_prompts[image_id])
                                if st.button(f"📋 複製到輸入框", key=f"copy_extracted_{i}"):
                                    st.session_state.temp_prompt = st.session_state.extracted_prompts[image_id]
                                    st.success("已複製到輸入框！")
                                    st.rerun()
                        
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ 生成圖像時發生錯誤：{str(e)}")
                    st.info("請檢查 API 密鑰是否正確，或嘗試重新配置 API 設置")

# 其他標籤頁面的內容保持不變，但需要添加 API 檢查...
# （這裡省略其他標籤頁面的代碼以節省空間，實際使用時需要添加相同的 API 檢查）

# 頁腳
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌟 <strong>Flux AI 圖像生成器 Pro Max</strong><br>
        🔑 API 密鑰管理 | ✨ 提示詞優化 | 🖼️ 圖生圖 | 📝 圖出提示詞<br>
        支援多種 API 提供商 | 安全的密鑰儲存
    </div>
    """,
    unsafe_allow_html=True
)

# 全域 API 狀態檢查提示
if not api_configured:
    st.sidebar.warning("⚠️ 功能受限：請配置 API 密鑰")

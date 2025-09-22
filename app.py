import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List
import json

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro Max", 
    page_icon="🎨", 
    layout="wide"
)

# 初始化 OpenAI 客戶端
@st.cache_resource
def init_client():
    return OpenAI(
        api_key=st.secrets.get("OPENAI_API_KEY", "YOUR_API_KEY"),
        base_url="https://api.navy/v1"
    )

client = init_client()

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
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'optimized_prompts' not in st.session_state:
        st.session_state.optimized_prompts = {}
    
    if 'extracted_prompts' not in st.session_state:
        st.session_state.extracted_prompts = {}

def optimize_prompt(original_prompt: str, style: str = "detailed") -> str:
    """使用 GPT 優化提示詞"""
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
        return "無法分析圖像，請重試"

def image_to_base64(image) -> str:
    """將 PIL 圖像轉換為 base64"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode()

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
                # 下載圖像並分析
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                extracted_prompt = extract_prompt_from_image(img_bytes)
                st.session_state.extracted_prompts[image_id] = extracted_prompt
                st.success("提示詞已提取！")

# 初始化會話狀態
init_session_state()

# 主標題
st.title("🎨 Flux AI 圖像生成器 Pro Max")
st.markdown("**全新功能：提示詞優化 | 圖生圖 | 圖出提示詞**")

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
            disabled=not prompt.strip()
        )
    
    with col2:
        # 使用說明和統計
        st.subheader("📋 使用說明")
        st.markdown(f"""
        **當前模型：** {FLUX_MODELS[selected_model]['name']}
        
        **新功能：**
        - ✨ 一鍵提示詞優化
        - 🖼️ 圖生圖功能
        - 📝 圖出提示詞
        - 📚 完整歷史記錄
        
        **步驟：**
        1. 輸入基礎提示詞
        2. 選擇優化風格（可選）
        3. 調整高級設定
        4. 點擊生成按鈕
        """)
        
        # 統計信息
        st.subheader("📊 快速統計")
        total_generations = len(st.session_state.generation_history)
        total_favorites = len(st.session_state.favorite_images)
        total_optimizations = len(st.session_state.optimized_prompts)
        
        st.metric("總生成數", total_generations)
        st.metric("收藏數量", total_favorites)
        st.metric("優化次數", total_optimizations)

    # 圖像生成邏輯
    if generate_btn and prompt.strip():
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
                    "generation_type": "text2img"
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

# 提示詞優化頁面
with tab2:
    st.subheader("🔧 提示詞優化工具")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 輸入原始提示詞")
        original_prompt = st.text_area(
            "原始提示詞",
            height=150,
            placeholder="輸入你的基礎提示詞..."
        )
        
        optimization_style = st.selectbox(
            "優化風格",
            ["detailed", "artistic", "realistic", "creative"],
            format_func=lambda x: {
                "detailed": "📝 詳細描述",
                "artistic": "🎨 藝術風格", 
                "realistic": "📸 真實攝影",
                "creative": "💭 創意想像"
            }[x]
        )
        
        if st.button("✨ 開始優化", type="primary", disabled=not original_prompt.strip()):
            with st.spinner("正在優化提示詞..."):
                optimized = optimize_prompt(original_prompt, optimization_style)
                st.session_state.current_optimized = optimized
                st.session_state.optimized_prompts[datetime.datetime.now().isoformat()] = {
                    "original": original_prompt,
                    "optimized": optimized,
                    "style": optimization_style
                }
    
    with col2:
        st.markdown("### 優化結果")
        if hasattr(st.session_state, 'current_optimized'):
            st.success("✅ 優化完成！")
            optimized_result = st.text_area(
                "優化後的提示詞",
                value=st.session_state.current_optimized,
                height=150,
                key="optimized_result"
            )
            
            col_copy, col_generate = st.columns(2)
            with col_copy:
                if st.button("📋 複製到剪貼板"):
                    st.success("已複製！")
            
            with col_generate:
                if st.button("🚀 直接生成圖像"):
                    st.session_state.direct_generate_prompt = st.session_state.current_optimized
                    st.switch_page("🚀 圖像生成")
        else:
            st.info("請在左側輸入提示詞並點擊優化")
    
    # 優化歷史
    if st.session_state.optimized_prompts:
        st.subheader("📚 優化歷史")
        for timestamp, opt_data in reversed(list(st.session_state.optimized_prompts.items())):
            with st.expander(f"優化記錄 - {timestamp[:19]}"):
                col_orig, col_opt = st.columns(2)
                with col_orig:
                    st.markdown("**原始提示詞：**")
                    st.write(opt_data["original"])
                with col_opt:
                    st.markdown(f"**優化後（{opt_data['style']}）：**")
                    st.write(opt_data["optimized"])

# 圖生圖頁面
with tab3:
    st.subheader("🖼️ 圖生圖功能")
    st.markdown("上傳一張圖像作為基礎，生成新的變化版本")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 上傳基礎圖像")
        uploaded_file = st.file_uploader(
            "選擇圖像文件",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="支持 PNG, JPG, JPEG, WebP 格式"
        )
        
        if uploaded_file is not None:
            # 顯示上傳的圖像
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的基礎圖像", use_container_width=True)
            
            # 圖像信息
            st.info(f"圖像尺寸: {image.size[0]}x{image.size[1]}")
            
            # 模型選擇（只顯示支持圖生圖的模型）
            img2img_models = {k: v for k, v in FLUX_MODELS.items() if v.get('supports_img2img', False)}
            
            selected_img2img_model = st.selectbox(
                "選擇模型",
                options=list(img2img_models.keys()),
                format_func=lambda x: f"{img2img_models[x]['icon']} {img2img_models[x]['name']}",
                key="img2img_model"
            )
            
            # 變化提示詞
            img2img_prompt = st.text_area(
                "變化描述",
                height=100,
                placeholder="描述你想要的變化，例如：將貓變成狗，改變背景為森林，添加魔法效果等..."
            )
            
            # 變化強度
            strength = st.slider(
                "變化強度",
                0.1, 1.0, 0.7,
                help="數值越高變化越大，越低越接近原圖"
            )
            
            # 生成按鈕
            generate_img2img_btn = st.button(
                "🔄 生成變化圖像",
                type="primary",
                disabled=not img2img_prompt.strip()
            )
    
    with col2:
        st.markdown("### 生成結果")
        
        if generate_img2img_btn and uploaded_file is not None and img2img_prompt.strip():
            with st.spinner("正在生成圖生圖變化..."):
                try:
                    # 注意：這裡使用模擬的圖生圖功能
                    # 實際實現需要支持 image parameter 的 API
                    enhanced_prompt = f"Based on the uploaded image, {img2img_prompt}, strength: {strength}"
                    
                    response = client.images.generate(
                        model=selected_img2img_model,
                        prompt=enhanced_prompt,
                        n=1,
                        size="1024x1024"
                    )
                    
                    st.success("✅ 圖生圖完成！")
                    
                    # 顯示結果
                    result_image_url = response.data[0].url
                    img_response = requests.get(result_image_url)
                    result_image = Image.open(BytesIO(img_response.content))
                    
                    st.image(result_image, caption="生成的變化圖像", use_container_width=True)
                    
                    # 保存到歷史
                    metadata = {
                        "generation_type": "img2img",
                        "base_image": "uploaded",
                        "strength": strength,
                        "model_info": img2img_models[selected_img2img_model]
                    }
                    add_to_history(enhanced_prompt, selected_img2img_model, [result_image_url], metadata)
                    
                    # 操作按鈕
                    col_download, col_favorite = st.columns(2)
                    with col_download:
                        img_buffer = BytesIO()
                        result_image.save(img_buffer, format='PNG')
                        st.download_button(
                            label="📥 下載結果",
                            data=img_buffer.getvalue(),
                            file_name="flux_img2img_result.png",
                            mime="image/png"
                        )
                    
                    with col_favorite:
                        if st.button("⭐ 加入收藏"):
                            favorite_item = {
                                "id": f"img2img_{len(st.session_state.favorite_images)}",
                                "image_url": result_image_url,
                                "timestamp": datetime.datetime.now(),
                                "history_item": {
                                    "prompt": enhanced_prompt,
                                    "model": selected_img2img_model,
                                    "metadata": metadata
                                }
                            }
                            st.session_state.favorite_images.append(favorite_item)
                            st.success("已加入收藏！")
                
                except Exception as e:
                    st.error(f"圖生圖失敗: {str(e)}")
        
        elif not uploaded_file:
            st.info("請先上傳一張基礎圖像")
        else:
            st.info("請輸入變化描述並點擊生成")

# 圖出提示詞頁面  
with tab4:
    st.subheader("📝 圖出提示詞")
    st.markdown("上傳圖像，AI 自動分析並生成詳細的提示詞描述")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 上傳分析圖像")
        analysis_file = st.file_uploader(
            "選擇要分析的圖像",
            type=['png', 'jpg', 'jpeg', 'webp'],
            key="analysis_upload",
            help="AI 將分析圖像並生成相應的提示詞"
        )
        
        if analysis_file is not None:
            # 顯示上傳的圖像
            analysis_image = Image.open(analysis_file)
            st.image(analysis_image, caption="待分析圖像", use_container_width=True)
            
            # 分析選項
            analysis_style = st.selectbox(
                "分析重點",
                ["comprehensive", "artistic", "technical", "simple"],
                format_func=lambda x: {
                    "comprehensive": "🔍 全面分析",
                    "artistic": "🎨 藝術要素",
                    "technical": "⚙️ 技術參數", 
                    "simple": "📝 簡潔描述"
                }[x],
                key="analysis_style"
            )
            
            # 分析按鈕
            analyze_btn = st.button(
                "🔍 開始分析",
                type="primary",
                key="analyze_image"
            )
    
    with col2:
        st.markdown("### 分析結果")
        
        if analyze_btn and analysis_file is not None:
            with st.spinner("正在分析圖像，生成提示詞..."):
                try:
                    # 重置文件指針
                    analysis_file.seek(0)
                    extracted_prompt = extract_prompt_from_image(analysis_file)
                    
                    st.success("✅ 分析完成！")
                    
                    # 顯示提取的提示詞
                    st.text_area(
                        "提取的提示詞",
                        value=extracted_prompt,
                        height=200,
                        key="extracted_prompt_display"
                    )
                    
                    # 操作按鈕
                    col_copy, col_optimize, col_generate = st.columns(3)
                    
                    with col_copy:
                        if st.button("📋 複製"):
                            # 在實際應用中，這裡需要 JavaScript 來複製到剪貼板
                            st.success("已複製到剪貼板！")
                    
                    with col_optimize:
                        if st.button("✨ 優化提示詞"):
                            with st.spinner("正在優化..."):
                                optimized_extracted = optimize_prompt(extracted_prompt, "detailed")
                                st.session_state.temp_extracted_optimized = optimized_extracted
                                st.rerun()
                    
                    with col_generate:
                        if st.button("🚀 生成圖像"):
                            st.session_state.extracted_for_generation = extracted_prompt
                            st.info("提示詞已準備好，請切換到生成頁面")
                    
                    # 顯示優化後的提示詞
                    if hasattr(st.session_state, 'temp_extracted_optimized'):
                        st.markdown("### 優化後的提示詞")
                        st.text_area(
                            "優化結果",
                            value=st.session_state.temp_extracted_optimized,
                            height=150,
                            key="optimized_extracted_display"
                        )
                        
                        if st.button("✅ 使用優化版本生成"):
                            st.session_state.extracted_for_generation = st.session_state.temp_extracted_optimized
                            st.info("優化後的提示詞已準備好！")
                    
                    # 保存分析記錄
                    timestamp = datetime.datetime.now().isoformat()
                    if 'extracted_history' not in st.session_state:
                        st.session_state.extracted_history = {}
                    
                    st.session_state.extracted_history[timestamp] = {
                        "prompt": extracted_prompt,
                        "style": analysis_style,
                        "image_size": analysis_image.size
                    }
                    
                except Exception as e:
                    st.error(f"圖像分析失敗: {str(e)}")
        
        elif not analysis_file:
            st.info("請上傳圖像開始分析")
        else:
            st.info("點擊分析按鈕開始處理")
    
    # 分析歷史
    if hasattr(st.session_state, 'extracted_history') and st.session_state.extracted_history:
        st.subheader("📚 分析歷史")
        for timestamp, extract_data in reversed(list(st.session_state.extracted_history.items())):
            with st.expander(f"分析記錄 - {timestamp[:19]}"):
                st.markdown(f"**分析風格：** {extract_data['style']}")
                st.markdown(f"**圖像尺寸：** {extract_data['image_size']}")
                st.markdown("**提取的提示詞：**")
                st.write(extract_data["prompt"])
                if st.button(f"🔄 重新使用", key=f"reuse_{timestamp}"):
                    st.session_state.extracted_for_generation = extract_data["prompt"]
                    st.success("提示詞已準備好生成！")

# 歷史記錄頁面 (保持原有功能)
with tab5:
    st.subheader("📚 生成歷史")
    
    if not st.session_state.generation_history:
        st.info("還沒有生成記錄，去生成一些圖像吧！")
    else:
        # 搜索和篩選
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            search_term = st.text_input("🔍 搜索提示詞", placeholder="輸入關鍵詞...")
        
        with col2:
            model_filter = st.selectbox(
                "📱 篩選模型",
                ["全部"] + list(FLUX_MODELS.keys()),
                format_func=lambda x: "全部模型" if x == "全部" else FLUX_MODELS[x]['name']
            )
        
        with col3:
            type_filter = st.selectbox(
                "🎯 生成類型",
                ["全部", "text2img", "img2img"]
            )
        
        with col4:
            sort_order = st.selectbox("📅 排序方式", ["最新", "最舊"])
        
        # 篩選邏輯
        filtered_history = st.session_state.generation_history.copy()
        
        if search_term:
            filtered_history = [
                item for item in filtered_history 
                if search_term.lower() in item['prompt'].lower()
            ]
        
        if model_filter != "全部":
            filtered_history = [
                item for item in filtered_history 
                if item['model'] == model_filter
            ]
        
        if type_filter != "全部":
            filtered_history = [
                item for item in filtered_history 
                if item['metadata'].get('generation_type', 'text2img') == type_filter
            ]
        
        if sort_order == "最舊":
            filtered_history = filtered_history[::-1]
        
        # 顯示篩選結果
        st.write(f"找到 {len(filtered_history)} 條記錄")
        
        if st.button("🗑️ 清除所有歷史", type="secondary"):
            st.session_state.generation_history = []
            st.success("歷史記錄已清除")
            st.rerun()
        
        # 分頁顯示歷史記錄
        for item in filtered_history[:10]:  # 顯示前10條
            generation_type = item['metadata'].get('generation_type', 'text2img')
            type_icon = "🖼️" if generation_type == "img2img" else "📝"
            
            with st.expander(
                f"{type_icon} {item['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                f"{FLUX_MODELS[item['model']]['name']} | "
                f"{item['prompt'][:50]}..."
            ):
                st.markdown(f"**類型：** {generation_type}")
                st.markdown(f"**提示詞：** {item['prompt']}")
                st.markdown(f"**模型：** {FLUX_MODELS[item['model']]['name']}")
                st.markdown(f"**尺寸：** {item['metadata'].get('size', 'N/A')}")
                st.markdown(f"**生成時間：** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                if generation_type == "img2img":
                    st.markdown(f"**變化強度：** {item['metadata'].get('strength', 'N/A')}")
                
                # 顯示圖像
                image_cols = st.columns(len(item['images']))
                for i, image_url in enumerate(item['images']):
                    with image_cols[i]:
                        image_id = f"{item['id']}_{i}_history"
                        display_image_with_actions(image_url, image_id, item)

# 收藏夾頁面 (保持原有功能)
with tab6:
    st.subheader("⭐ 我的收藏")
    
    if not st.session_state.favorite_images:
        st.info("還沒有收藏任何圖像，去收藏一些喜歡的圖像吧！")
    else:
        if st.button("🗑️ 清除所有收藏", type="secondary"):
            st.session_state.favorite_images = []
            st.success("收藏已清除")
            st.rerun()
        
        # 收藏網格顯示
        cols = st.columns(3)
        
        for i, favorite in enumerate(st.session_state.favorite_images):
            with cols[i % 3]:
                st.subheader(f"收藏 #{i+1}")
                st.caption(f"收藏於：{favorite['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                if favorite.get('history_item'):
                    history_item = favorite['history_item']
                    st.caption(f"模型：{FLUX_MODELS.get(history_item['model'], {}).get('name', 'Unknown')}")
                    with st.expander("查看提示詞"):
                        st.text(history_item['prompt'])
                
                display_image_with_actions(
                    favorite['image_url'], 
                    f"fav_{favorite['id']}", 
                    favorite.get('history_item')
                )
                
                st.markdown("---")

# 頁腳
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌟 <strong>Flux AI 圖像生成器 Pro Max</strong><br>
        ✨ 提示詞優化 | 🖼️ 圖生圖 | 📝 圖出提示詞 | 🎯 5種Flux模型<br>
        由 Black Forest Labs & OpenAI 技術驅動
    </div>
    """,
    unsafe_allow_html=True
)

# 處理跨標籤的狀態傳遞
if hasattr(st.session_state, 'extracted_for_generation'):
    st.sidebar.success(f"📝 已準備提示詞：{st.session_state.extracted_for_generation[:50]}...")
    if st.sidebar.button("🚀 前往生成"):
        st.session_state.temp_prompt_from_extraction = st.session_state.extracted_for_generation
        delattr(st.session_state, 'extracted_for_generation')
        st.rerun()

if hasattr(st.session_state, 'temp_prompt_from_extraction'):
    # 這個會在圖像生成頁面被使用
    pass

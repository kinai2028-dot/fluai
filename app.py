import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro", 
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
        "type": "快速生成"
    },
    "flux.1-krea-dev": {
        "name": "FLUX.1 Krea Dev", 
        "description": "創意開發版本，適合實驗性生成",
        "icon": "🎨",
        "type": "創意開發"
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本"
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🔧",
        "type": "編輯專用"
    },
    "flux.1-kontext-max": {
        "name": "FLUX.1 Kontext Max",
        "description": "最高性能版本，極致品質",
        "icon": "🚀",
        "type": "極致性能"
    }
}

# 初始化 session state
def init_session_state():
    """初始化會話狀態"""
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "生成器"

def image_to_base64(image_url: str) -> str:
    """將圖像URL轉換為base64編碼"""
    try:
        response = requests.get(image_url)
        image_bytes = response.content
        base64_encoded = base64.b64encode(image_bytes).decode()
        return base64_encoded
    except:
        return None

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
    
    # 限制歷史記錄數量
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

def display_image_with_actions(image_url: str, image_id: str, history_item: Dict = None):
    """顯示圖像和相關操作"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # 下載圖像
    img_response = requests.get(image_url)
    img = Image.open(BytesIO(img_response.content))
    
    st.image(img, use_container_width=True)
    
    with col1:
        # 下載按鈕
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
        # 收藏按鈕
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
        # 重新生成按鈕
        if history_item and st.button(
            "🔄 重新生成",
            key=f"regenerate_{image_id}",
            use_container_width=True
        ):
            st.session_state.regenerate_prompt = history_item['prompt']
            st.session_state.regenerate_model = history_item['model']
            st.session_state.current_page = "生成器"
            st.rerun()

# 初始化會話狀態
init_session_state()

# 導航欄
st.title("🎨 Flux AI 圖像生成器 Pro")

# 頁面導航
tab1, tab2, tab3, tab4 = st.tabs(["🚀 圖像生成", "📚 歷史記錄", "⭐ 收藏夾", "📊 統計"])

# 圖像生成頁面
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 模型選擇
        st.subheader("選擇 Flux 模型")
        model_cols = st.columns(len(FLUX_MODELS))
        
        selected_model = None
        for i, (model_key, model_info) in enumerate(FLUX_MODELS.items()):
            with model_cols[i]:
                if st.button(
                    f"{model_info['icon']} {model_info['name']}\n{model_info['type']}",
                    key=f"model_{model_key}",
                    use_container_width=True,
                    help=model_info['description']
                ):
                    selected_model = model_key
        
        # 如果沒有選擇，使用默認模型
        if not selected_model:
            selected_model = st.selectbox(
                "或從下拉選單選擇模型：",
                options=list(FLUX_MODELS.keys()),
                format_func=lambda x: f"{FLUX_MODELS[x]['icon']} {FLUX_MODELS[x]['name']}",
                index=0
            )
        
        # 顯示選中的模型信息
        if selected_model:
            model_info = FLUX_MODELS[selected_model]
            st.info(f"已選擇：{model_info['icon']} {model_info['name']} - {model_info['description']}")
        
        # 提示詞輸入
        st.subheader("輸入提示詞")
        
        # 檢查是否有重新生成的請求
        default_prompt = ""
        if hasattr(st.session_state, 'regenerate_prompt'):
            default_prompt = st.session_state.regenerate_prompt
            if hasattr(st.session_state, 'regenerate_model'):
                selected_model = st.session_state.regenerate_model
            # 清除重新生成標記
            delattr(st.session_state, 'regenerate_prompt')
            if hasattr(st.session_state, 'regenerate_model'):
                delattr(st.session_state, 'regenerate_model')
        
        prompt = st.text_area(
            "描述你想要生成的圖像",
            value=default_prompt,
            height=120,
            placeholder="例如：A cute cat wearing a wizard hat in a magical forest..."
        )
        
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
            
            # 品質設定（針對 Pro 版本）
            if "pro" in selected_model or "max" in selected_model:
                quality = st.select_slider(
                    "圖像品質",
                    options=["標準", "高品質", "超高品質"],
                    value="高品質"
                )
        
        # 快速提示詞
        st.subheader("💡 快速提示詞")
        prompt_categories = {
            "人物肖像": [
                "Professional headshot of a businesswoman in modern office",
                "Portrait of an elderly man with wise eyes and gentle smile",
                "Young artist with paint-splattered apron in studio"
            ],
            "自然風景": [
                "Sunset over snow-capped mountains with alpine lake",
                "Tropical beach with crystal clear water and palm trees", 
                "Autumn forest with golden leaves and morning mist"
            ],
            "科幻未來": [
                "Cyberpunk cityscape with neon lights and flying cars",
                "Space station orbiting a distant planet",
                "Robot assistant in a futuristic home"
            ],
            "藝術創意": [
                "Abstract geometric composition with vibrant colors",
                "Watercolor painting of blooming cherry blossoms",
                "Digital art of a dragon made of flowing water"
            ]
        }
        
        category = st.selectbox("選擇類別", list(prompt_categories.keys()))
        prompt_cols = st.columns(len(prompt_categories[category]))
        
        for i, quick_prompt in enumerate(prompt_categories[category]):
            with prompt_cols[i]:
                if st.button(
                    quick_prompt[:30] + "...",
                    key=f"quick_{category}_{i}",
                    use_container_width=True,
                    help=quick_prompt
                ):
                    st.session_state.quick_prompt = quick_prompt
                    st.rerun()
        
        # 如果有快速提示詞被選中
        if hasattr(st.session_state, 'quick_prompt'):
            prompt = st.session_state.quick_prompt
            delattr(st.session_state, 'quick_prompt')
        
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
        
        **步驟：**
        1. 選擇 Flux 模型
        2. 輸入詳細的圖像描述
        3. 調整高級設定（可選）
        4. 點擊生成按鈕
        5. 查看結果並保存
        
        **提示詞技巧：**
        - 使用具體的描述詞
        - 包含風格、顏色、構圖
        - 避免過於複雜的句子
        - 可以指定藝術風格
        """)
        
        # 統計信息
        st.subheader("📊 快速統計")
        total_generations = len(st.session_state.generation_history)
        total_favorites = len(st.session_state.favorite_images)
        
        st.metric("總生成數", total_generations)
        st.metric("收藏數量", total_favorites)
        
        if total_generations > 0:
            most_used_model = max(
                set(item['model'] for item in st.session_state.generation_history),
                key=lambda x: sum(1 for item in st.session_state.generation_history if item['model'] == x)
            )
            st.metric("最常用模型", FLUX_MODELS.get(most_used_model, {}).get('name', most_used_model))

    # 圖像生成邏輯
    if generate_btn and prompt.strip():
        with st.spinner(f"正在使用 {FLUX_MODELS[selected_model]['name']} 生成圖像，請稍候..."):
            try:
                # 準備生成參數
                generation_params = {
                    "model": selected_model,
                    "prompt": prompt,
                    "n": num_images,
                    "size": selected_size
                }
                
                # 調用 API
                response = client.images.generate(**generation_params)
                
                # 準備歷史記錄數據
                image_urls = [img.url for img in response.data]
                metadata = {
                    "size": selected_size,
                    "num_images": num_images,
                    "model_info": FLUX_MODELS[selected_model]
                }
                
                # 添加到歷史記錄
                add_to_history(prompt, selected_model, image_urls, metadata)
                
                # 顯示結果
                st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
                
                # 顯示圖像網格
                if num_images == 1:
                    cols = [st.container()]
                elif num_images == 2:
                    cols = st.columns(2)
                else:
                    cols = st.columns(2)
                
                for i, image_data in enumerate(response.data):
                    with cols[i % len(cols)]:
                        st.subheader(f"圖像 {i+1}")
                        image_id = f"{len(st.session_state.generation_history)-1}_{i}"
                        display_image_with_actions(
                            image_data.url, 
                            image_id, 
                            st.session_state.generation_history[0]
                        )
                        
                        if i % 2 == 1 and i < len(response.data) - 1:
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ 生成圖像時發生錯誤：{str(e)}")
                st.info("請檢查 API 密鑰是否正確，或稍後再試。")

# 歷史記錄頁面
with tab2:
    st.subheader("📚 生成歷史")
    
    if not st.session_state.generation_history:
        st.info("還沒有生成記錄，去生成一些圖像吧！")
    else:
        # 搜索和篩選
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 搜索提示詞", placeholder="輸入關鍵詞...")
        
        with col2:
            model_filter = st.selectbox(
                "📱 篩選模型",
                ["全部"] + list(FLUX_MODELS.keys()),
                format_func=lambda x: "全部模型" if x == "全部" else FLUX_MODELS[x]['name']
            )
        
        with col3:
            sort_order = st.selectbox("📅 排序方式", ["最新", "最舊"])
        
        # 篩選歷史記錄
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
        
        if sort_order == "最舊":
            filtered_history = filtered_history[::-1]
        
        # 清除歷史記錄按鈕
        if st.button("🗑️ 清除所有歷史", type="secondary"):
            st.session_state.generation_history = []
            st.success("歷史記錄已清除")
            st.rerun()
        
        # 分頁顯示
        items_per_page = 5
        total_items = len(filtered_history)
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        if total_items > 0:
            page = st.number_input(
                f"頁面 (共 {total_pages} 頁)",
                min_value=1,
                max_value=max(1, total_pages),
                value=1
            )
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            
            # 顯示歷史項目
            for item in filtered_history[start_idx:end_idx]:
                with st.expander(
                    f"🕒 {item['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                    f"{FLUX_MODELS[item['model']]['name']} | "
                    f"{item['prompt'][:50]}..."
                ):
                    st.markdown(f"**提示詞：** {item['prompt']}")
                    st.markdown(f"**模型：** {FLUX_MODELS[item['model']]['name']}")
                    st.markdown(f"**尺寸：** {item['metadata']['size']}")
                    st.markdown(f"**生成時間：** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 顯示圖像
                    image_cols = st.columns(len(item['images']))
                    for i, image_url in enumerate(item['images']):
                        with image_cols[i]:
                            image_id = f"{item['id']}_{i}_history"
                            display_image_with_actions(image_url, image_id, item)
        else:
            st.info("沒有符合條件的記錄")

# 收藏夾頁面
with tab3:
    st.subheader("⭐ 我的收藏")
    
    if not st.session_state.favorite_images:
        st.info("還沒有收藏任何圖像，去收藏一些喜歡的圖像吧！")
    else:
        # 清除收藏按鈕
        if st.button("🗑️ 清除所有收藏", type="secondary"):
            st.session_state.favorite_images = []
            st.success("收藏已清除")
            st.rerun()
        
        # 收藏網格顯示
        cols = st.columns(3)
        
        for i, favorite in enumerate(st.session_state.favorite_images):
            with cols[i % 3]:
                st.subheader(f"收藏 #{i+1}")
                
                # 顯示收藏時間
                st.caption(f"收藏於：{favorite['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                # 如果有歷史項目信息，顯示相關信息
                if favorite.get('history_item'):
                    history_item = favorite['history_item']
                    st.caption(f"模型：{FLUX_MODELS[history_item['model']]['name']}")
                    with st.expander("查看提示詞"):
                        st.text(history_item['prompt'])
                
                # 顯示圖像和操作
                display_image_with_actions(
                    favorite['image_url'], 
                    f"fav_{favorite['id']}", 
                    favorite.get('history_item')
                )
                
                st.markdown("---")

# 統計頁面
with tab4:
    st.subheader("📊 使用統計")
    
    if not st.session_state.generation_history:
        st.info("還沒有生成記錄，無法顯示統計信息。")
    else:
        # 基本統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("總生成次數", len(st.session_state.generation_history))
        
        with col2:
            total_images = sum(len(item['images']) for item in st.session_state.generation_history)
            st.metric("總圖像數", total_images)
        
        with col3:
            st.metric("收藏數量", len(st.session_state.favorite_images))
        
        with col4:
            if st.session_state.generation_history:
                avg_per_generation = total_images / len(st.session_state.generation_history)
                st.metric("平均每次生成", f"{avg_per_generation:.1f}")
        
        # 模型使用統計
        st.subheader("🔧 模型使用分佈")
        model_usage = {}
        for item in st.session_state.generation_history:
            model = item['model']
            model_usage[model] = model_usage.get(model, 0) + 1
        
        if model_usage:
            for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                model_name = FLUX_MODELS.get(model, {}).get('name', model)
                percentage = (count / len(st.session_state.generation_history)) * 100
                st.write(f"**{model_name}:** {count} 次 ({percentage:.1f}%)")
        
        # 尺寸使用統計
        st.subheader("📐 圖像尺寸分佈")
        size_usage = {}
        for item in st.session_state.generation_history:
            size = item['metadata'].get('size', '未知')
            size_usage[size] = size_usage.get(size, 0) + 1
        
        if size_usage:
            for size, count in sorted(size_usage.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(st.session_state.generation_history)) * 100
                st.write(f"**{size}:** {count} 次 ({percentage:.1f}%)")
        
        # 時間統計
        st.subheader("📅 生成時間分析")
        if len(st.session_state.generation_history) > 1:
            dates = [item['timestamp'].date() for item in st.session_state.generation_history]
            date_counts = {}
            for date in dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            recent_dates = sorted(date_counts.items(), reverse=True)[:7]  # 最近7天
            
            st.write("最近生成活動：")
            for date, count in recent_dates:
                st.write(f"**{date}:** {count} 次生成")

# 頁腳
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌟 <strong>Flux AI 圖像生成器 Pro</strong><br>
        支持 5 種 Flux 模型 | 完整歷史記錄 | 收藏管理<br>
        由 Black Forest Labs 技術驅動
    </div>
    """,
    unsafe_allow_html=True
)

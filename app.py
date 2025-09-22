import streamlit as st
from PIL import Image
from io import BytesIO
import time
import random

# 页面配置
st.set_page_config(page_title="Flux AI 生成器 - 企業進階版", page_icon="🎨", layout="wide")

# 模型定義（支援自訂模型）
FLUX_MODELS = {
    "flux.schnell": {"name": "Flux Schnell", "desc": "最快最穩定", "reliability": "高", "icon": "⚡"},
    "flux.krea-dev": {"name": "Flux Krea Dev", "desc": "創意開發", "reliability": "中", "icon": "🎨"},
    "flux.pro": {"name": "Flux Pro", "desc": "旗艦品質", "reliability": "中", "icon": "👑"},
    "custom": {"name": "自訂模型", "desc": "輸入任意模型ID", "reliability": "未知", "icon": "🛠️"},
}

# 模擬 API 客戶端（實際運行請替換成 openai.Streamlit 等真實 API 客戶端）
class MockClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, **kwargs):
        time.sleep(1)  # 模擬 API 耗時
        images = []
        for _ in range(kwargs.get("n", 1)):
            images.append({"url": "https://placedog.net/500/300"})  # 用寵物圖模擬生成圖
        return images

# 初始化 API 客戶端
@st.cache_resource
def get_client(api_key, base_url):
    if not api_key or not base_url:
        return None
    return MockClient(api_key, base_url)

def main():
    st.title("Flux AI 圖像生成器 - 支援自訂模型")

    # 初始化 session
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = "https://api.navy/v1"

    # 側欄 API 配置區
    with st.sidebar:
        st.header("API 設定")
        api_key = st.text_input("API Key", value=st.session_state["api_key"], type="password")
        base_url = st.text_input("API Base URL", value=st.session_state["base_url"])
        if st.button("儲存設定"):
            st.session_state["api_key"] = api_key
            st.session_state["base_url"] = base_url
            st.success("設定已儲存！")
    
    # 主頁面
    cols = st.columns([2, 1])
    with cols[0]:
        # 模型選擇區
        model_keys = list(FLUX_MODELS.keys())
        model_labels = [f"{FLUX_MODELS[m]['icon']} {FLUX_MODELS[m]['name']}" for m in model_keys]
        selected_model_label = st.selectbox("選擇模型", model_labels, help="選擇內建模型或自訂模型")
        selected_model = model_keys[model_labels.index(selected_model_label)]

        # 處理自訂模型輸入
        model_to_use = selected_model
        if selected_model == "custom":
            custom_model_id = st.text_input("請輸入模型ID", placeholder="例如：my-custom-model-2024")
            if custom_model_id.strip():
                model_to_use = custom_model_id.strip()
            else:
                st.warning("請輸入正確的模型 ID")
                model_to_use = None

        # 提示詞與其他參數
        prompt = st.text_area("輸入提示詞", height=120)
        num_images = st.slider("生成數量", 1, 4, 1)
        size = st.selectbox("圖像尺寸", ["512x512", "1024x1024"], index=1)

        if st.button("生成圖像"):
            client = get_client(st.session_state["api_key"], st.session_state["base_url"])
            if not client:
                st.error("請先設定正確的 API Key 和 Base URL")
            elif not model_to_use:
                st.error("請選擇或輸入正確的模型 ID")
            elif not prompt.strip():
                st.error("請輸入提示詞")
            else:
                with st.spinner("生成中..."):
                    try:
                        images = client.generate(
                            model=model_to_use,
                            prompt=prompt,
                            n=num_images,
                            size=size
                        )
                        st.success(f"生成成功！（模型：{model_to_use}）")
                        image_cols = st.columns(num_images)
                        for idx, img in enumerate(images):
                            with image_cols[idx]:
                                st.image(img["url"])
                    except Exception as e:
                        st.error(f"生成失敗：{str(e)}")

    # 右側說明與統計（彈性擴展）
    with cols[1]:
        st.header("📊 模型說明")
        for m in model_keys:
            st.markdown(f"**{FLUX_MODELS[m]['icon']} {FLUX_MODELS[m]['name']}**\n{FLUX_MODELS[m]['desc']}（可靠性：{FLUX_MODELS[m]['reliability']}）")
            st.caption("---")

        st.info("**⭐️ 特色**")
        st.write("- 支援多模型切換")
        st.write("- 可自訂任意模型ID")
        st.write("- 內建錯誤處理")
        st.write("- 靈活API設定")

if __name__ == "__main__":
    main()

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import zipfile
import io
import datetime

st.set_page_config(page_title="Flux AI 圖像生成器 (v3)", layout="wide")

st.title("🎨 Flux AI 圖像生成器 (v3)")

# API 配置區
st.sidebar.header("API 配置")
api_key = st.sidebar.text_input("API Key", type="password")
base_url = st.sidebar.text_input("Base URL", "https://api.navy/v1")

# 模型選擇
models = [
    "flux.1-schnell",
    "flux.1.1-por",
    "flux.latest",
    "flux.1-krea-dev",
    "flux.1-kontext-pro",
    "flux.1-kontext-max"
]
model = st.sidebar.selectbox("選擇模型", models, index=0)

# 品質字典
qualities_dict = {
    "flux.1-schnell": ["standard", "hd", "ultra"],
    "flux.1.1-por": ["hd", "ultra"],
    "flux.latest": ["standard", "hd"],
    "flux.1-krea-dev": ["quality1", "quality2"],
    "flux.1-kontext-pro": ["qualityA", "qualityB"],
    "flux.1-kontext-max": ["qualityX", "qualityY"]
}

if model not in qualities_dict:
    qualities_dict[model] = ["default_quality"]

quality = st.sidebar.selectbox("選擇品質", qualities_dict[model])

# 圖像尺寸和張數
sizes = ["1024x1024", "1024x1792", "1792x1024", "512x512", "256x256"]
size = st.sidebar.selectbox("圖像尺寸", sizes, index=0)
n = st.sidebar.slider("生成圖片數量", 1, 5, 1)

# 提示詞輸入
st.header("📝 輸入提示詞")
prompt = st.text_area("描述您想生成的圖像", value="A cute cat wearing a wizard hat", height=120)

btn_generate = st.button("生成圖像")

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "image_records" not in st.session_state:
    st.session_state.image_records = []

if btn_generate:
    if not api_key.strip():
        st.error("請輸入 API Key")
    elif not prompt.strip():
        st.error("請輸入提示詞")
    else:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            with st.spinner("正在生成圖像，請稍候..."):
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=n,
                    size=size,
                    quality=quality
                    # 不帶style參數
                )

                images = []
                for img_data in response.data:
                    image_url = img_data.url
                    img_resp = requests.get(image_url)
                    img_resp.raise_for_status()
                    img = Image.open(BytesIO(img_resp.content))
                    images.append(img)

                st.session_state.generated_images = images
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.image_records.append({
                    "prompt": prompt,
                    "time": timestamp,
                    "images": images
                })
                st.success(f"成功生成 {n} 張圖像！")
        except Exception as e:
            st.error(f"生成失敗: {str(e)}")

# 圖像顯示 (同之前，包含下載功能)
if st.session_state.generated_images:
    st.header("🖼️ 生成的圖像")
    num_columns = min(n, 4)
    cols = st.columns(num_columns)
    for idx, img in enumerate(st.session_state.generated_images):
        col = cols[idx % num_columns]
        with col:
            st.image(img, use_column_width=True, caption=f"Image {idx + 1}")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            st.download_button(
                label=f"下載圖像 {idx + 1}",
                data=img_byte_arr.getvalue(),
                file_name=f"image_{idx + 1}.png",
                mime="image/png"
            )

    if st.button("下載所有圖像"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for i, img in enumerate(st.session_state.generated_images):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"image_{i + 1}.png", img_byte_arr.getvalue())
        zip_buffer.seek(0)
        st.download_button(
            label="點擊下載 ZIP 文件",
            data=zip_buffer,
            file_name="images.zip",
            mime="application/zip"
        )

# 歷史記錄展示
if st.checkbox("顯示歷史生成記錄"):
    st.header("🕘 生成歷史記錄")
    for record in reversed(st.session_state.image_records):
        st.write(f"時間: {record['time']}，提示詞: {record['prompt']}")
        cols = st.columns(min(len(record["images"]), 4))
        for i, img in enumerate(record["images"]):
            with cols[i]:
                st.image(img, use_column_width=True)

import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List, Optional, Tuple
import time
import random
import json
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro", 
    page_icon="🎨", 
    layout="wide"
)

# 增強的錯誤處理類
class FluxAPIErrorHandler:
    def __init__(self):
        self.error_patterns = {
            'provider_500': {
                'keywords': ['unexpected provider error', '500'],
                'type': 'provider_error',
                'severity': 'high',
                'retry_recommended': True,
                'solutions': [
                    '服務器臨時故障，系統會自動重試',
                    '嘗試切換到其他可用模型',
                    '簡化提示詞內容',
                    '檢查 API 提供商服務狀態'
                ]
            },
            'auth_error': {
                'keywords': ['401', '403', 'unauthorized', 'forbidden'],
                'type': 'authentication',
                'severity': 'critical',
                'retry_recommended': False,
                'solutions': [
                    '檢查 API 密鑰是否正確',
                    '驗證帳戶權限和餘額',
                    '確認 API 端點配置',
                    '重新生成 API 密鑰'
                ]
            },
            'rate_limit': {
                'keywords': ['429', 'rate limit', 'too many requests'],
                'type': 'rate_limiting',
                'severity': 'medium',
                'retry_recommended': True,
                'solutions': [
                    '請求頻率過高，正在等待重試',
                    '考慮減少並發請求',
                    '升級到更高級別的 API 計劃',
                    '使用指數退避策略'
                ]
            },
            'model_error': {
                'keywords': ['404', 'model not found', 'invalid model'],
                'type': 'model_unavailable',
                'severity': 'high',
                'retry_recommended': False,
                'solutions': [
                    '選擇的模型不可用',
                    '切換到已驗證的可用模型',
                    '檢查模型名稱拼寫',
                    '聯繫 API 提供商確認模型狀態'
                ]
            },
            'network_error': {
                'keywords': ['timeout', 'connection', 'network', 'dns'],
                'type': 'network_issue',
                'severity': 'medium',
                'retry_recommended': True,
                'solutions': [
                    '網絡連接問題，正在重試',
                    '檢查網絡連接穩定性',
                    '嘗試更換網絡環境',
                    '檢查防火牆和代理設置'
                ]
            }
        }
    
    def analyze_error(self, error_msg: str) -> Dict:
        """分析錯誤並提供詳細診斷"""
        error_msg_lower = error_msg.lower()
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if any(keyword in error_msg_lower for keyword in pattern_info['keywords']):
                return {
                    'pattern': pattern_name,
                    'type': pattern_info['type'],
                    'severity': pattern_info['severity'],
                    'retry_recommended': pattern_info['retry_recommended'],
                    'solutions': pattern_info['solutions'],
                    'original_error': error_msg
                }
        
        # 未知錯誤的默認處理
        return {
            'pattern': 'unknown',
            'type': 'unknown_error',
            'severity': 'medium',
            'retry_recommended': True,
            'solutions': [
                '未知錯誤，嘗試重新生成',
                '檢查所有配置設置',
                '簡化提示詞內容',
                '聯繫技術支持'
            ],
            'original_error': error_msg
        }

# 增強的 API 客戶端類
class ResilientFluxClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.error_handler = FluxAPIErrorHandler()
        self.session_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0
        }
    
    def generate_with_resilience(self, **params) -> Tuple[bool, any, Dict]:
        """
        具有彈性的圖像生成方法
        返回 (成功狀態, 結果, 診斷信息)
        """
        max_retries = 3
        base_delay = 2
        fallback_models = ['flux.1-schnell', 'flux.1-krea-dev', 'flux.1.1-pro']
        original_model = params.get('model')
        
        # 更新統計
        self.session_stats['total_requests'] += 1
        
        # 主要生成邏輯
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.session_stats['retry_attempts'] += 1
                    st.info(f"🔄 重試生成 (第 {attempt + 1}/{max_retries} 次)")
                
                # 嘗試生成
                response = self.client.images.generate(**params)
                
                # 成功
                self.session_stats['successful_requests'] += 1
                return True, response, {
                    'status': 'success',
                    'attempts': attempt + 1,
                    'model_used': params.get('model'),
                    'message': f'成功生成 (第 {attempt + 1} 次嘗試)'
                }
                
            except Exception as e:
                error_msg = str(e)
                error_analysis = self.error_handler.analyze_error(error_msg)
                
                st.warning(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {error_analysis['type']}")
                
                # 特殊處理 500 錯誤
                if error_analysis['pattern'] == 'provider_500':
                    if attempt < max_retries - 1:
                        # 指數退避 + 隨機延遲
                        delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                        
                        st.info(f"⏱️ 檢測到提供商錯誤，{delay:.1f} 秒後重試...")
                        
                        # 顯示進度條
                        progress_bar = st.progress(0)
                        for i in range(int(delay)):
                            progress_bar.progress((i + 1) / delay)
                            time.sleep(1)
                        progress_bar.empty()
                        
                        continue
                
                # 模型回退策略
                elif error_analysis['pattern'] == 'model_error' and attempt < max_retries - 1:
                    available_fallbacks = [m for m in fallback_models if m != params.get('model')]
                    if available_fallbacks:
                        fallback_model = available_fallbacks[0]
                        params['model'] = fallback_model
                        st.info(f"🔄 嘗試回退模型: {fallback_model}")
                        continue
                
                # 速率限制處理
                elif error_analysis['pattern'] == 'rate_limit' and attempt < max_retries - 1:
                    delay = base_delay * (3 ** attempt) + random.uniform(2, 5)  # 更長延遲
                    st.warning(f"🚦 遇到速率限制，{delay:.1f} 秒後重試...")
                    time.sleep(delay)
                    continue
                
                # 不建議重試的錯誤
                elif not error_analysis['retry_recommended']:
                    self.session_stats['failed_requests'] += 1
                    return False, error_analysis, {
                        'status': 'failed',
                        'attempts': attempt + 1,
                        'error_type': error_analysis['type'],
                        'no_retry_reason': '錯誤類型不適合重試'
                    }
                
                # 最後一次嘗試失敗
                elif attempt == max_retries - 1:
                    self.session_stats['failed_requests'] += 1
                    return False, error_analysis, {
                        'status': 'failed',
                        'attempts': max_retries,
                        'error_type': error_analysis['type'],
                        'message': '所有重試嘗試均失敗'
                    }
                
                # 其他可重試錯誤
                else:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    st.info(f"⏳ {delay:.1f} 秒後重試...")
                    time.sleep(delay)
                    continue
        
        # 應該不會到達這裡
        self.session_stats['failed_requests'] += 1
        return False, {'type': 'unknown_error', 'original_error': 'Unexpected error'}, {
            'status': 'failed',
            'attempts': max_retries,
            'message': '未知錯誤'
        }

def show_error_recovery_panel(error_analysis: Dict, diagnostic_info: Dict):
    """顯示錯誤恢復面板"""
    st.subheader("🚨 錯誤診斷和恢復")
    
    # 錯誤概覽
    col_error1, col_error2, col_error3 = st.columns(3)
    
    with col_error1:
        severity_color = {
            'critical': '🔴',
            'high': '🟠', 
            'medium': '🟡',
            'low': '🟢'
        }
        st.metric(
            "錯誤嚴重程度",
            f"{severity_color.get(error_analysis['severity'], '❓')} {error_analysis['severity'].upper()}"
        )
    
    with col_error2:
        st.metric("錯誤類型", error_analysis['type'].replace('_', ' ').title())
    
    with col_error3:
        st.metric("嘗試次數", diagnostic_info.get('attempts', 'N/A'))
    
    # 詳細錯誤信息
    with st.expander("🔍 詳細錯誤信息"):
        st.code(error_analysis['original_error'])
        st.json({
            'error_pattern': error_analysis['pattern'],
            'retry_recommended': error_analysis['retry_recommended'],
            'diagnostic_status': diagnostic_info.get('status'),
            'model_attempted': diagnostic_info.get('model_used', 'N/A')
        })
    
    # 解決方案
    st.subheader("💡 推薦解決方案")
    
    for i, solution in enumerate(error_analysis['solutions'], 1):
        st.write(f"{i}. {solution}")
    
    # 快速修復按鈕
    st.subheader("⚡ 快速修復")
    
    col_fix1, col_fix2, col_fix3, col_fix4 = st.columns(4)
    
    with col_fix1:
        if st.button("🔄 重新嘗試", use_container_width=True):
            st.session_state.retry_generation = True
            st.rerun()
    
    with col_fix2:
        if st.button("🎯 自動選擇最佳模型", use_container_width=True):
            if 'model_test_results' in st.session_state:
                available_models = [
                    model for model, result in st.session_state.model_test_results.items()
                    if result.get('available', False)
                ]
                if available_models:
                    # 按響應時間排序
                    best_model = min(
                        available_models,
                        key=lambda x: st.session_state.model_test_results[x].get('response_time', 999)
                    )
                    st.session_state.auto_selected_model = best_model
                    st.success(f"已選擇最佳模型: {best_model}")
                    st.rerun()
                else:
                    st.warning("沒有已測試的可用模型")
            else:
                st.warning("請先測試模型可用性")
    
    with col_fix3:
        if st.button("✂️ 簡化提示詞", use_container_width=True):
            if 'last_prompt' in st.session_state:
                # 簡化提示詞邏輯
                original = st.session_state.last_prompt
                simplified = simplify_prompt(original)
                st.session_state.simplified_prompt = simplified
                st.info(f"原提示詞: {original[:50]}...")
                st.success(f"簡化後: {simplified[:50]}...")
                st.rerun()
    
    with col_fix4:
        if st.button("🧪 測試 API 連接", use_container_width=True):
            if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
                test_api_connection()
            else:
                st.warning("請先配置 API 密鑰")

def simplify_prompt(original_prompt: str) -> str:
    """簡化提示詞"""
    # 移除複雜的修飾詞和長句
    words = original_prompt.split()
    
    # 保留核心詞彙
    core_words = []
    skip_words = {
        'extremely', 'highly', 'very', 'incredibly', 'amazingly',
        'detailed', 'intricate', 'complex', 'sophisticated',
        'professional', 'cinematic', 'photorealistic', 'ultra-realistic'
    }
    
    for word in words[:15]:  # 限制長度
        if word.lower() not in skip_words:
            core_words.append(word)
    
    simplified = ' '.join(core_words)
    
    # 如果太短，添加基本描述
    if len(simplified) < 20:
        simplified += ", simple and clear"
    
    return simplified

def test_api_connection():
    """測試 API 連接"""
    try:
        client = OpenAI(
            api_key=st.session_state.api_config['api_key'],
            base_url=st.session_state.api_config['base_url']
        )
        
        with st.spinner("測試 API 連接..."):
            models = client.models.list()
            st.success(f"✅ API 連接正常，發現 {len(models.data)} 個模型")
            
            # 檢查 Flux 模型
            flux_models = [m.id for m in models.data if 'flux' in m.id.lower()]
            if flux_models:
                st.info(f"🎨 可用的 Flux 模型: {', '.join(flux_models[:3])}...")
            else:
                st.warning("⚠️ 未發現 Flux 模型")
                
    except Exception as e:
        st.error(f"❌ API 連接測試失敗: {str(e)}")

def show_session_diagnostics():
    """顯示會話診斷信息"""
    if 'resilient_client' in st.session_state:
        client = st.session_state.resilient_client
        stats = client.session_stats
        
        st.subheader("📊 會話診斷")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("總請求數", stats['total_requests'])
        
        with col_stat2:
            success_rate = (
                stats['successful_requests'] / stats['total_requests'] * 100
                if stats['total_requests'] > 0 else 0
            )
            st.metric("成功率", f"{success_rate:.1f}%")
        
        with col_stat3:
            st.metric("失敗次數", stats['failed_requests'])
        
        with col_stat4:
            st.metric("重試次數", stats['retry_attempts'])
        
        # 建議
        if success_rate < 50:
            st.error("🚨 成功率過低，建議檢查 API 配置")
        elif success_rate < 80:
            st.warning("⚠️ 成功率不理想，建議優化設置")
        else:
            st.success("✅ 系統運行良好")

def create_resilient_client() -> Optional[ResilientFluxClient]:
    """創建彈性客戶端"""
    if 'api_config' not in st.session_state or not st.session_state.api_config.get('api_key'):
        return None
    
    return ResilientFluxClient(
        api_key=st.session_state.api_config['api_key'],
        base_url=st.session_state.api_config['base_url']
    )

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
    
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}

# API 提供商配置
API_PROVIDERS = {
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓"
    },
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# Flux 模型配置
FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型，最穩定",
        "icon": "⚡",
        "type": "快速生成",
        "reliability": "高"
    },
    "flux.1-krea-dev": {
        "name": "FLUX.1 Krea Dev", 
        "description": "創意開發版本，適合實驗性生成",
        "icon": "🎨",
        "type": "創意開發",
        "reliability": "中"
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "reliability": "中"
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🔧",
        "type": "編輯專用",
        "reliability": "低"
    },
    "flux.1-kontext-max": {
        "name": "FLUX.1 Kontext Max",
        "description": "最高性能版本，極致品質",
        "icon": "🚀",
        "type": "極致性能",
        "reliability": "低"
    }
}

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 API 設置")
    
    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')
    
    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )
    
    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")
    
    current_key = st.session_state.api_config.get('api_key', '')
    masked_key = '*' * 20 + current_key[-8:] if len(current_key) > 8 else ''
    
    api_key_input = st.text_input(
        "API 密鑰",
        value="",
        type="password",
        placeholder=f"請輸入 {provider_info['name']} 的 API 密鑰...",
        help=f"API 密鑰通常以 '{provider_info['key_prefix']}' 開頭"
    )
    
    if current_key and not api_key_input:
        st.caption(f"🔐 當前密鑰: {masked_key}")
    
    base_url_input = st.text_input(
        "API 端點 URL",
        value=st.session_state.api_config.get('base_url', provider_info['base_url_default']),
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_btn = st.button("💾 保存設置", type="primary")
    
    with col2:
        test_btn = st.button("🧪 測試連接")
    
    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")
    
    if save_btn:
        if not api_key_input and not current_key:
            st.error("❌ 請輸入 API 密鑰")
        else:
            final_api_key = api_key_input if api_key_input else current_key
            st.session_state.api_config = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }
            st.success("✅ API 設置已保存")
            st.rerun()
    
    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key
        if test_api_key:
            test_api_connection()
    
    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
        st.success("🗑️ 設置已清除")
        st.rerun()

# 初始化
init_session_state()

# 創建彈性客戶端
resilient_client = create_resilient_client()
api_configured = resilient_client is not None

# 側邊欄
with st.sidebar:
    show_api_settings()
    
    st.markdown("---")
    if api_configured:
        st.success("🟢 增強 API 已配置")
        provider = st.session_state.api_config.get('provider', 'Unknown')
        st.caption(f"使用: {API_PROVIDERS.get(provider, {}).get('name', provider)}")
        
        # 顯示會話統計
        if 'resilient_client' in st.session_state:
            stats = st.session_state.resilient_client.session_stats
            success_rate = (
                stats['successful_requests'] / stats['total_requests'] * 100
                if stats['total_requests'] > 0 else 0
            )
            st.metric("會話成功率", f"{success_rate:.1f}%")
        
    else:
        st.error("🔴 API 未配置")
    
    st.markdown("### 🛡️ 錯誤恢復")
    if st.button("🔄 重置錯誤狀態", use_container_width=True):
        if 'error_state' in st.session_state:
            del st.session_state.error_state
        st.success("錯誤狀態已重置")
        st.rerun()

# 主頁面
st.title("🎨 Flux AI 圖像生成器 Pro - 增強錯誤恢復版")

if not api_configured:
    st.error("⚠️ 請先配置 API 密鑰")
else:
    # 存儲彈性客戶端到 session state
    st.session_state.resilient_client = resilient_client
    
    # 主要生成界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 智能模型選擇")
        
        # 按可靠性排序模型
        sorted_models = sorted(
            FLUX_MODELS.items(),
            key=lambda x: {'高': 0, '中': 1, '低': 2}.get(x[1].get('reliability', '低'), 2)
        )
        
        selected_model = st.selectbox(
            "選擇模型 (按可靠性排序)",
            options=[m[0] for m in sorted_models],
            format_func=lambda x: f"{FLUX_MODELS[x]['icon']} {FLUX_MODELS[x]['name']} - 可靠性: {FLUX_MODELS[x]['reliability']}",
            index=0
        )
        
        # 自動選擇檢查
        if 'auto_selected_model' in st.session_state:
            selected_model = st.session_state.auto_selected_model
            del st.session_state.auto_selected_model
        
        model_info = FLUX_MODELS[selected_model]
        
        reliability_color = {'高': '🟢', '中': '🟡', '低': '🔴'}
        st.info(
            f"已選擇: {model_info['icon']} {model_info['name']} "
            f"{reliability_color[model_info['reliability']]} 可靠性: {model_info['reliability']}"
        )
        
        # 提示詞輸入
        st.subheader("✏️ 輸入提示詞")
        
        # 簡化提示詞檢查
        default_prompt = ""
        if 'simplified_prompt' in st.session_state:
            default_prompt = st.session_state.simplified_prompt
            del st.session_state.simplified_prompt
            st.success("✂️ 使用簡化後的提示詞")
        
        prompt = st.text_area(
            "描述你想要生成的圖像",
            value=default_prompt,
            height=120,
            placeholder="例如: A simple cat sitting on a table"
        )
        
        # 保存提示詞到 session state
        if prompt:
            st.session_state.last_prompt = prompt
        
        # 高級設定
        with st.expander("🔧 高級設定"):
            col_size, col_num = st.columns(2)
            
            with col_size:
                size_options = {
                    "1024x1024": "正方形 (1:1) - 最穩定",
                    "1152x896": "橫向 (4:3.5)", 
                    "896x1152": "直向 (3.5:4)",
                }
                
                selected_size = st.selectbox(
                    "圖像尺寸",
                    options=list(size_options.keys()),
                    format_func=lambda x: size_options[x],
                    index=0
                )
            
            with col_num:
                num_images = st.slider("生成數量", 1, 2, 1, help="減少數量提高穩定性")
        
        # 生成按鈕
        generate_btn = st.button(
            "🚀 增強生成圖像",
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip()
        )
        
        # 重試檢查
        if 'retry_generation' in st.session_state:
            generate_btn = True
            del st.session_state.retry_generation
    
    with col2:
        st.subheader("🛡️ 錯誤恢復系統")
        
        if api_configured:
            st.success("✅ 增強錯誤處理已啟用")
            st.markdown("""
            **新功能:**
            - 🔄 智能重試機制
            - 🎯 自動模型回退
            - 📊 實時錯誤分析  
            - 💡 智能解決方案
            """)
        
        # 顯示會話診斷
        show_session_diagnostics()
        
        st.subheader("💡 使用建議")
        st.markdown("""
        **避免 500 錯誤:**
        - 使用高可靠性模型
        - 簡化提示詞內容
        - 選擇標準圖像尺寸
        - 減少生成數量
        
        **錯誤處理流程:**
        1. 自動重試 (3次)
        2. 模型自動回退
        3. 提供解決方案
        4. 用戶手動修復
        """)

    # 增強的生成邏輯
    if generate_btn and prompt.strip():
        st.subheader("🔄 生成進度")
        
        generation_params = {
            "model": selected_model,
            "prompt": prompt,
            "n": num_images,
            "size": selected_size
        }
        
        # 使用彈性客戶端生成
        success, result, diagnostic_info = resilient_client.generate_with_resilience(**generation_params)
        
        if success:
            # 成功處理
            response = result
            st.success(f"✨ 生成成功! {diagnostic_info.get('message', '')}")
            
            # 顯示圖像
            for i, image_data in enumerate(response.data):
                st.subheader(f"圖像 {i+1}")
                
                img_response = requests.get(image_data.url)
                img = Image.open(BytesIO(img_response.content))
                st.image(img, use_container_width=True)
                
                # 下載按鈕
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                st.download_button(
                    label=f"📥 下載圖像 {i+1}",
                    data=img_buffer.getvalue(),
                    file_name=f"flux_generated_{i+1}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )
            
            # 記錄成功到歷史
            # add_to_history(prompt, selected_model, [img.url for img in response.data], {...})
            
        else:
            # 失敗處理
            error_analysis = result
            st.error(f"❌ 生成失敗: {error_analysis['type']}")
            
            # 顯示錯誤恢復面板
            show_error_recovery_panel(error_analysis, diagnostic_info)
            
            # 記錄失敗到 session state
            st.session_state.error_state = {
                'error_analysis': error_analysis,
                'diagnostic_info': diagnostic_info,
                'timestamp': datetime.datetime.now()
            }

# 頁腳
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🛡️ <strong>Flux AI 圖像生成器 Pro - 增強錯誤恢復版</strong><br>
        🔄 智能重試 | 🎯 自動回退 | 📊 錯誤分析 | 💡 智能修復<br>
        專為解決 500 錯誤而設計
    </div>
    """,
    unsafe_allow_html=True
)

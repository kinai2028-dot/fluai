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
import traceback

# 设定页面配置
st.set_page_config(
    page_title="Flux AI 图像生成器 Pro", 
    page_icon="🎨", 
    layout="wide"
)

# 增强的错误处理类
class FluxAPIErrorHandler:
    def __init__(self):
        self.error_patterns = {
            'provider_500': {
                'keywords': ['unexpected provider error', '500', 'internal server error'],
                'type': 'provider_error',
                'severity': 'high',
                'retry_recommended': True,
                'solutions': [
                    '服务器临时故障，系统会自动重试',
                    '尝试切换到其他可用模型',
                    '简化提示词内容',
                    '检查 API 提供商服务状态'
                ]
            },
            'auth_error': {
                'keywords': ['401', '403', 'unauthorized', 'forbidden', 'invalid api key', 'authentication'],
                'type': 'authentication',
                'severity': 'critical',
                'retry_recommended': False,
                'solutions': [
                    '检查 API 密钥是否正确',
                    '验证账户权限和余额',
                    '确认 API 端点配置',
                    '重新生成 API 密钥'
                ]
            },
            'rate_limit': {
                'keywords': ['429', 'rate limit', 'too many requests', 'quota exceeded'],
                'type': 'rate_limiting',
                'severity': 'medium',
                'retry_recommended': True,
                'solutions': [
                    '请求频率过高，正在等待重试',
                    '考虑减少并发请求',
                    '升级到更高级别的 API 计划',
                    '使用指数退避策略'
                ]
            },
            'model_error': {
                'keywords': ['404', 'model not found', 'invalid model', 'model does not exist'],
                'type': 'model_unavailable',
                'severity': 'high',
                'retry_recommended': False,
                'solutions': [
                    '选择的模型不可用',
                    '切换到已验证的可用模型',
                    '检查模型名称拼写',
                    '联系 API 提供商确认模型状态'
                ]
            },
            'network_error': {
                'keywords': ['timeout', 'connection', 'network', 'dns', 'ssl', 'certificate'],
                'type': 'network_issue',
                'severity': 'medium',
                'retry_recommended': True,
                'solutions': [
                    '网络连接问题，正在重试',
                    '检查网络连接稳定性',
                    '尝试更换网络环境',
                    '检查防火墙和代理设置'
                ]
            },
            'parameter_error': {
                'keywords': ['invalid parameter', 'bad request', '400', 'validation error', 'malformed'],
                'type': 'parameter_invalid',
                'severity': 'high',
                'retry_recommended': False,
                'solutions': [
                    '请求参数无效',
                    '检查图像尺寸设置',
                    '验证提示词格式',
                    '确认生成数量在允许范围内'
                ]
            },
            'content_policy': {
                'keywords': ['content policy', 'inappropriate', 'unsafe', 'filtered', 'blocked'],
                'type': 'content_violation',
                'severity': 'medium',
                'retry_recommended': False,
                'solutions': [
                    '提示词违反内容政策',
                    '修改提示词避免敏感内容',
                    '使用更温和的描述',
                    '参考平台使用指南'
                ]
            },
            'openai_client_error': {
                'keywords': ['openai', 'client error', 'initialization failed', 'api client'],
                'type': 'client_initialization',
                'severity': 'critical',
                'retry_recommended': False,
                'solutions': [
                    'OpenAI 客户端初始化失败',
                    '检查 API 密钥格式',
                    '验证 API 端点 URL',
                    '重新配置 API 设置'
                ]
            }
        }
    
    def analyze_error(self, error_msg: str, context: Dict = None) -> Dict:
        """分析错误并提供详细诊断 - 增强版本"""
        if not isinstance(error_msg, str):
            error_msg = str(error_msg)
        
        error_msg_lower = error_msg.lower()
        
        # 记录原始错误和上下文
        analysis_result = {
            'pattern': 'unknown',
            'type': 'unknown_error',
            'severity': 'medium',
            'retry_recommended': True,
            'solutions': ['尝试重新生成', '检查配置设置'],
            'original_error': error_msg[:1000],
            'context': context or {},
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        # 搜索匹配的错误模式
        for pattern_name, pattern_info in self.error_patterns.items():
            try:
                keywords = pattern_info.get('keywords', [])
                if any(keyword in error_msg_lower for keyword in keywords):
                    analysis_result.update({
                        'pattern': pattern_name,
                        'type': pattern_info.get('type', 'unknown'),
                        'severity': pattern_info.get('severity', 'medium'),
                        'retry_recommended': pattern_info.get('retry_recommended', True),
                        'solutions': pattern_info.get('solutions', ['尝试重新生成']),
                        'matched_keywords': [kw for kw in keywords if kw in error_msg_lower]
                    })
                    st.info(f"🔍 匹配到错误模式: {pattern_name}")
                    return analysis_result
                    
            except Exception as match_error:
                st.warning(f"错误模式匹配失败 ({pattern_name}): {str(match_error)}")
                continue
        
        # 特殊错误检测
        if 'unexpected error in retry loop' in error_msg_lower:
            analysis_result.update({
                'pattern': 'retry_loop_error',
                'type': 'retry_mechanism_failure',
                'severity': 'high',
                'retry_recommended': False,
                'solutions': [
                    '重试机制本身出现问题',
                    '重置错误状态和客户端',
                    '切换到最稳定的模型',
                    '减少生成复杂度'
                ]
            })
        
        st.warning(f"⚠️ 未识别的错误模式: {error_msg[:100]}...")
        return analysis_result

# 增强的 API 客户端类
class ResilientFluxClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.initialization_error = None
        
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            st.success("✅ OpenAI 客户端初始化成功")
        except Exception as e:
            self.initialization_error = str(e)
            st.error(f"❌ OpenAI 客户端初始化失败: {str(e)}")
            
        self.error_handler = FluxAPIErrorHandler()
        self.session_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'error_types': {},
            'last_errors': []
        }
    
    def validate_parameters(self, **params) -> Tuple[bool, str]:
        """验证生成参数"""
        try:
            model = params.get('model')
            prompt = params.get('prompt', '')
            n = params.get('n', 1)
            size = params.get('size', '1024x1024')
            
            # 检查必需参数
            if not model:
                return False, "缺少模型参数"
            
            if not prompt or len(prompt.strip()) == 0:
                return False, "提示词不能为空"
            
            if len(prompt) > 4000:
                return False, "提示词过长，请缩短至4000字符以内"
            
            # 检查数量
            if not isinstance(n, int) or n < 1 or n > 4:
                return False, "生成数量必须在1-4之间"
            
            # 检查尺寸格式
            valid_sizes = ['1024x1024', '1152x896', '896x1152', '1344x768', '768x1344']
            if size not in valid_sizes:
                return False, f"无效的图像尺寸，支持: {', '.join(valid_sizes)}"
            
            return True, "参数验证通过"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def log_error(self, error_type: str, error_msg: str, attempt: int, model: str):
        """记录错误信息"""
        try:
            error_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': error_type,
                'message': error_msg[:500],  # 限制长度
                'attempt': attempt,
                'model': model
            }
            
            self.session_stats['last_errors'].append(error_entry)
            
            # 只保留最近10个错误
            if len(self.session_stats['last_errors']) > 10:
                self.session_stats['last_errors'] = self.session_stats['last_errors'][-10:]
            
            # 更新错误类型统计
            if error_type in self.session_stats['error_types']:
                self.session_stats['error_types'][error_type] += 1
            else:
                self.session_stats['error_types'][error_type] = 1
                
        except Exception as log_error:
            st.warning(f"记录错误信息失败: {str(log_error)}")
    
    def generate_with_resilience(self, **params) -> Tuple[bool, any, Dict]:
        """
        具有弹性的图像生成方法 - 完全重写
        """
        # 初始检查
        if self.client is None:
            error_result = {
                'type': 'client_initialization',
                'severity': 'critical',
                'retry_recommended': False,
                'solutions': ['重新配置 API 客户端', '检查 API 密钥和端点'],
                'original_error': self.initialization_error or 'OpenAI 客户端未初始化'
            }
            return False, error_result, {'status': 'failed', 'attempts': 0, 'error_type': 'client_init'}
        
        # 参数验证
        param_valid, param_msg = self.validate_parameters(**params)
        if not param_valid:
            error_result = {
                'type': 'parameter_invalid',
                'severity': 'high',
                'retry_recommended': False,
                'solutions': [f'参数验证失败: {param_msg}', '检查输入参数', '使用推荐的参数设置'],
                'original_error': param_msg
            }
            return False, error_result, {'status': 'failed', 'attempts': 0, 'error_type': 'param_validation'}
        
        # 设置重试参数
        max_retries = 3
        base_delay = 2
        fallback_models = ['flux.1-schnell', 'flux.1-krea-dev', 'flux.1.1-pro']
        original_model = params.get('model', 'flux.1-schnell')
        current_params = params.copy()
        
        # 更新统计
        self.session_stats['total_requests'] += 1
        
        st.info(f"🚀 开始生成: 模型={original_model}, 提示词长度={len(params.get('prompt', ''))}")
        
        # 重试循环
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.session_stats['retry_attempts'] += 1
                    st.info(f"🔄 第 {attempt + 1}/{max_retries} 次重试...")
                
                # 显示当前参数
                with st.expander(f"📋 第 {attempt + 1} 次尝试参数"):
                    st.json({
                        'model': current_params.get('model'),
                        'prompt_length': len(current_params.get('prompt', '')),
                        'n': current_params.get('n'),
                        'size': current_params.get('size')
                    })
                
                # 执行生成
                st.info(f"📡 调用 API...")
                response = self.client.images.generate(**current_params)
                
                # 成功处理
                self.session_stats['successful_requests'] += 1
                st.success(f"✅ 生成成功! (第 {attempt + 1} 次尝试)")
                
                return True, response, {
                    'status': 'success',
                    'attempts': attempt + 1,
                    'model_used': current_params.get('model'),
                    'message': f'成功生成 (尝试 {attempt + 1}/{max_retries})',
                    'final_params': current_params
                }
                
            except Exception as e:
                error_msg = str(e)
                error_context = {
                    'attempt': attempt + 1,
                    'max_retries': max_retries,
                    'model': current_params.get('model'),
                    'params': {k: v for k, v in current_params.items() if k != 'prompt'}  # 不记录完整提示词
                }
                
                # 记录错误
                self.log_error('generation_error', error_msg, attempt + 1, current_params.get('model'))
                
                # 分析错误
                error_analysis = self.error_handler.analyze_error(error_msg, error_context)
                
                st.error(f"❌ 第 {attempt + 1} 次尝试失败: {error_analysis.get('type', 'unknown')}")
                st.code(f"错误详情: {error_msg[:200]}...")
                
                # 决定是否继续重试
                if attempt >= max_retries - 1:
                    # 最后一次尝试失败
                    st.error("💥 所有重试尝试均已失败")
                    self.session_stats['failed_requests'] += 1
                    
                    return False, error_analysis, {
                        'status': 'failed',
                        'attempts': max_retries,
                        'error_type': error_analysis.get('type'),
                        'message': '达到最大重试次数',
                        'all_errors': self.session_stats['last_errors'][-max_retries:]
                    }
                
                # 根据错误类型决定重试策略
                pattern = error_analysis.get('pattern', 'unknown')
                
                if pattern == 'provider_500':
                    # 500错误 - 指数退避重试
                    delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                    st.warning(f"🔄 检测到服务器错误，{delay:.1f} 秒后重试...")
                    
                    progress_bar = st.progress(0)
                    for i in range(int(delay * 2)):  # 更细粒度的进度
                        progress_bar.progress((i + 1) / (delay * 2))
                        time.sleep(0.5)
                    progress_bar.empty()
                    continue
                
                elif pattern == 'model_error' and attempt < max_retries - 1:
                    # 模型错误 - 尝试回退模型
                    current_model = current_params.get('model')
                    available_fallbacks = [m for m in fallback_models if m != current_model]
                    
                    if available_fallbacks:
                        fallback_model = available_fallbacks[0]
                        current_params['model'] = fallback_model
                        st.info(f"🎯 尝试回退模型: {current_model} → {fallback_model}")
                        continue
                    else:
                        st.warning("⚠️ 没有可用的回退模型")
                
                elif pattern == 'rate_limit':
                    # 速率限制 - 长延迟重试
                    delay = base_delay * (4 ** attempt) + random.uniform(5, 10)
                    st.warning(f"🚦 遇到速率限制，{delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                
                elif pattern == 'parameter_error':
                    # 参数错误 - 尝试简化参数
                    if current_params.get('n', 1) > 1:
                        current_params['n'] = 1
                        st.info("🔧 简化参数: 减少生成数量到1")
                        continue
                    elif current_params.get('size') != '1024x1024':
                        current_params['size'] = '1024x1024'
                        st.info("🔧 简化参数: 使用标准尺寸")
                        continue
                
                elif not error_analysis.get('retry_recommended', True):
                    # 不建议重试的错误
                    st.error("🛑 检测到不适合重试的错误")
                    self.session_stats['failed_requests'] += 1
                    
                    return False, error_analysis, {
                        'status': 'failed',
                        'attempts': attempt + 1,
                        'error_type': error_analysis.get('type'),
                        'no_retry_reason': '错误类型不适合重试'
                    }
                
                # 默认重试策略
                delay = base_delay * (1.5 ** attempt) + random.uniform(0, 2)
                st.info(f"⏳ {delay:.1f} 秒后进行默认重试...")
                time.sleep(delay)
                continue
        
        # 如果循环正常结束但没有返回，这是一个意外情况
        st.error("🚨 重试循环意外结束 - 这不应该发生")
        
        # 创建详细的错误报告
        error_analysis = {
            'pattern': 'retry_loop_completion',
            'type': 'unexpected_loop_completion',
            'severity': 'critical',
            'retry_recommended': False,
            'solutions': [
                '重试循环意外完成',
                '重置客户端状态',
                '检查系统日志',
                '联系技术支持'
            ],
            'original_error': 'Retry loop completed without return',
            'context': {
                'max_retries': max_retries,
                'final_params': current_params,
                'error_history': self.session_stats['last_errors'][-max_retries:]
            }
        }
        
        self.session_stats['failed_requests'] += 1
        
        return False, error_analysis, {
            'status': 'failed',
            'attempts': max_retries,
            'error_type': 'loop_completion_error',
            'message': '重试循环意外完成'
        }

def show_error_recovery_panel(error_analysis: Dict, diagnostic_info: Dict):
    """显示错误恢复面板 - 增强版本"""
    st.subheader("🚨 详细错误诊断")
    
    # 错误概览
    error_type = error_analysis.get('type', 'unknown_error')
    severity = error_analysis.get('severity', 'medium')
    pattern = error_analysis.get('pattern', 'unknown')
    
    # 基本信息显示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        severity_colors = {
            'critical': ('🔴', 'red'),
            'high': ('🟠', 'orange'), 
            'medium': ('🟡', 'yellow'),
            'low': ('🟢', 'green')
        }
        icon, color = severity_colors.get(severity, ('❓', 'gray'))
        st.metric("严重程度", f"{icon} {severity.upper()}")
    
    with col2:
        st.metric("错误类型", error_type.replace('_', ' ').title())
    
    with col3:
        st.metric("错误模式", pattern.replace('_', ' ').title())
    
    with col4:
        attempts = diagnostic_info.get('attempts', 'N/A')
        st.metric("尝试次数", str(attempts))
    
    # 详细错误信息
    with st.expander("🔍 完整错误详情", expanded=False):
        st.markdown("### 原始错误消息")
        original_error = error_analysis.get('original_error', '未知错误')
        st.code(original_error)
        
        st.markdown("### 错误上下文")
        context = error_analysis.get('context', {})
        if context:
            st.json(context)
        else:
            st.info("无额外上下文信息")
        
        st.markdown("### 诊断信息")
        st.json(diagnostic_info)
        
        # 匹配的关键词
        matched_keywords = error_analysis.get('matched_keywords', [])
        if matched_keywords:
            st.markdown("### 匹配的错误关键词")
            st.write(", ".join(matched_keywords))
    
    # 解决方案
    st.subheader("💡 推荐解决方案")
    solutions = error_analysis.get('solutions', ['尝试重新生成'])
    
    for i, solution in enumerate(solutions, 1):
        if i == 1:
            st.success(f"**🎯 首选方案:** {solution}")
        else:
            st.info(f"**{i}.** {solution}")
    
    # 快速修复操作
    st.subheader("⚡ 快速修复操作")
    
    col_fix1, col_fix2, col_fix3, col_fix4 = st.columns(4)
    
    with col_fix1:
        if st.button("🔄 立即重试", use_container_width=True, type="primary"):
            st.session_state.retry_generation = True
            st.success("准备重试...")
            st.rerun()
    
    with col_fix2:
        if st.button("🛡️ 安全模式", use_container_width=True):
            # 设置最安全的参数
            st.session_state.safe_mode_params = {
                'model': 'flux.1-schnell',
                'size': '1024x1024',
                'n': 1
            }
            st.success("已启用安全模式")
            st.rerun()
    
    with col_fix3:
        if st.button("🔧 重置客户端", use_container_width=True):
            # 重置客户端状态
            if 'resilient_client' in st.session_state:
                del st.session_state.resilient_client
            st.success("客户端状态已重置")
            st.rerun()
    
    with col_fix4:
        if st.button("📞 获取帮助", use_container_width=True):
            st.session_state.show_help = True
            st.rerun()
    
    # 错误趋势分析
    if 'resilient_client' in st.session_state:
        client = st.session_state.resilient_client
        error_types = client.session_stats.get('error_types', {})
        
        if error_types:
            st.subheader("📈 错误趋势分析")
            
            # 显示错误类型分布
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{error_type}:** {count} 次")
            
            # 最近错误历史
            recent_errors = client.session_stats.get('last_errors', [])
            if recent_errors:
                with st.expander("📝 最近错误历史"):
                    for error in recent_errors[-5:]:  # 显示最近5个错误
                        st.write(f"**{error.get('timestamp', 'Unknown time')}** - {error.get('type', 'Unknown')} (尝试 {error.get('attempt', 'N/A')})")
                        st.caption(error.get('message', 'No message')[:100])

def show_help_panel():
    """显示帮助面板"""
    st.subheader("📞 错误解决帮助")
    
    st.markdown("""
    ### 🔧 常见问题解决方案
    
    #### 1. "Unexpected error in retry loop"
    - **原因**: 重试机制本身出现问题
    - **解决**: 使用"重置客户端"按钮，然后重新配置API
    
    #### 2. 500 服务器错误
    - **原因**: API提供商服务器临时故障
    - **解决**: 等待自动重试，或切换到更稳定的模型
    
    #### 3. 401/403 认证错误
    - **原因**: API密钥无效或权限不足
    - **解决**: 检查API密钥，确认账户余额
    
    #### 4. 模型不可用 (404)
    - **原因**: 选择的模型暂时不可用
    - **解决**: 切换到 flux.1-schnell (最稳定)
    
    #### 5. 参数验证失败
    - **原因**: 输入参数超出允许范围
    - **解决**: 使用"安全模式"或简化设置
    """)
    
    st.markdown("""
    ### 🛡️ 预防措施
    
    - **使用稳定模型**: 优先选择 flux.1-schnell
    - **合理提示词**: 保持在1000字符以内
    - **标准尺寸**: 使用 1024x1024 最稳定
    - **少量生成**: 一次生成1-2张图片
    - **定期测试**: 定期测试API连接状态
    """)
    
    if st.button("✅ 了解了", type="primary"):
        if 'show_help' in st.session_state:
            del st.session_state.show_help
        st.rerun()

# 其他函数保持不变...
def create_resilient_client() -> Optional[ResilientFluxClient]:
    """创建弹性客户端"""
    try:
        if 'api_config' not in st.session_state or not st.session_state.api_config.get('api_key'):
            return None
        
        config = st.session_state.api_config
        return ResilientFluxClient(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
    except Exception as e:
        st.error(f"创建弹性客户端失败: {str(e)}")
        return None

def init_session_state():
    """初始化会话状态"""
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

# API 和模型配置
API_PROVIDERS = {
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 图像生成服务",
        "icon": "⚓"
    }
}

FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最稳定的模型，推荐用于错误恢复",
        "icon": "⚡",
        "reliability": "高"
    },
    "flux.1-krea-dev": {
        "name": "FLUX.1 Krea Dev", 
        "description": "创意开发版本",
        "icon": "🎨",
        "reliability": "中"
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "旗舰模型",
        "icon": "👑",
        "reliability": "中"
    }
}

def show_api_settings():
    """显示API设置"""
    st.subheader("🔑 API 设置")
    
    current_key = st.session_state.api_config.get('api_key', '')
    
    api_key_input = st.text_input(
        "API 密钥",
        value="",
        type="password",
        placeholder="请输入 API 密钥...",
    )
    
    base_url_input = st.text_input(
        "API 端点",
        value=st.session_state.api_config.get('base_url', 'https://api.navy/v1'),
    )
    
    if st.button("💾 保存设置", type="primary"):
        if api_key_input or current_key:
            st.session_state.api_config = {
                'provider': 'Navy',
                'api_key': api_key_input or current_key,
                'base_url': base_url_input,
                'validated': False
            }
            st.success("✅ 设置已保存")
            st.rerun()

# 初始化
init_session_state()
resilient_client = create_resilient_client()
api_configured = resilient_client is not None

# 主界面
st.title("🎨 Flux AI 图像生成器 Pro - 详细错误修复版")

# 帮助面板检查
if st.session_state.get('show_help', False):
    show_help_panel()
else:
    # 侧边栏
    with st.sidebar:
        show_api_settings()
        
        if api_configured:
            st.success("🟢 API 已配置")
            
            # 显示客户端统计
            stats = resilient_client.session_stats
            if stats['total_requests'] > 0:
                success_rate = stats['successful_requests'] / stats['total_requests'] * 100
                st.metric("成功率", f"{success_rate:.1f}%")
                
                if stats['error_types']:
                    st.subheader("⚠️ 错误统计")
                    for error_type, count in list(stats['error_types'].items())[:3]:
                        st.write(f"• {error_type}: {count}次")
        else:
            st.error("🔴 API 未配置")
    
    # 主生成界面
    if not api_configured:
        st.error("⚠️ 请先配置 API 密钥")
    else:
        st.session_state.resilient_client = resilient_client
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 图像生成")
            
            # 安全模式检查
            if 'safe_mode_params' in st.session_state:
                safe_params = st.session_state.safe_mode_params
                st.success("🛡️ 安全模式已启用")
                
                selected_model = safe_params.get('model', 'flux.1-schnell')
                selected_size = safe_params.get('size', '1024x1024')
                num_images = safe_params.get('n', 1)
                
                del st.session_state.safe_mode_params
            else:
                # 正常选择
                selected_model = st.selectbox(
                    "选择模型",
                    options=list(FLUX_MODELS.keys()),
                    format_func=lambda x: f"{FLUX_MODELS[x]['icon']} {FLUX_MODELS[x]['name']} (可靠性: {FLUX_MODELS[x]['reliability']})",
                    index=0
                )
                
                selected_size = st.selectbox(
                    "图像尺寸",
                    options=['1024x1024', '1152x896', '896x1152'],
                    index=0
                )
                
                num_images = st.slider("生成数量", 1, 3, 1)
            
            prompt = st.text_area(
                "输入提示词",
                height=100,
                placeholder="例如: A simple cat sitting on a table"
            )
            
            generate_btn = st.button(
                "🚀 生成图像",
                type="primary",
                disabled=not prompt.strip()
            )
            
            # 重试检查
            if st.session_state.get('retry_generation', False):
                generate_btn = True
                del st.session_state.retry_generation
        
        with col2:
            st.subheader("🛡️ 系统状态")
            st.success("✅ 增强错误处理已启用")
            st.info("🔧 详细错误诊断")
            st.info("🔄 智能重试机制")
            st.info("🎯 自动参数优化")
            
            if resilient_client.session_stats['total_requests'] > 0:
                st.subheader("📊 会话统计")
                stats = resilient_client.session_stats
                st.write(f"总请求: {stats['total_requests']}")
                st.write(f"成功: {stats['successful_requests']}")
                st.write(f"失败: {stats['failed_requests']}")
                st.write(f"重试: {stats['retry_attempts']}")
        
        # 生成逻辑
        if generate_btn and prompt.strip():
            st.subheader("🔄 生成进度")
            
            generation_params = {
                "model": selected_model,
                "prompt": prompt,
                "n": num_images,
                "size": selected_size
            }
            
            success, result, diagnostic_info = resilient_client.generate_with_resilience(**generation_params)
            
            if success:
                response = result
                st.success(f"✨ 生成成功! {diagnostic_info.get('message', '')}")
                
                for i, image_data in enumerate(response.data):
                    st.subheader(f"图像 {i+1}")
                    
                    try:
                        img_response = requests.get(image_data.url)
                        img = Image.open(BytesIO(img_response.content))
                        st.image(img, use_container_width=True)
                        
                        # 下载按钮
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='PNG')
                        st.download_button(
                            label=f"📥 下载图像 {i+1}",
                            data=img_buffer.getvalue(),
                            file_name=f"flux_generated_{i+1}.png",
                            mime="image/png"
                        )
                    except Exception as img_error:
                        st.error(f"显示图像失败: {str(img_error)}")
            else:
                error_analysis = result
                st.error(f"❌ 生成失败: {error_analysis.get('type', 'unknown')}")
                show_error_recovery_panel(error_analysis, diagnostic_info)

# 页脚
st.markdown("---")
st.markdown("🛡️ **Flux AI 图像生成器 Pro - 详细错误修复版** | 🔧 完整错误诊断 | 🔄 智能重试 | 💡 详细解决方案")

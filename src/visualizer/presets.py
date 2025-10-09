import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Dict, Any, Optional
from pathlib import Path
import toml

class _ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "configs/painter.toml"
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config_cache is not None:
            return self._config_cache
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            self._config_cache = {}
            return self._config_cache
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config_cache = toml.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
            self._config_cache = {}
        
        return self._config_cache
    
    def get_cmap_config(self) -> Dict[str, Any]:
        """获取cmap配置"""
        config = self.load_config()
        return config.get('cmap', {})
    
    def get_font_config(self) -> Dict[str, Any]:
        """获取font配置"""
        config = self.load_config()
        return config.get('font', {})
    
    def get_theme_config(self) -> Dict[str, Any]:
        """获取theme配置"""
        config = self.load_config()
        return config.get('theme', {})
    
    def reload_config(self):
        """重新加载配置文件"""
        self._config_cache = None
        return self.load_config()

# 全局配置加载器实例
_config_loader = _ConfigLoader()

class CmapPresets:
    """预设的colormap配置，适用于不同的科学实验场景"""
    
    # 医学影像场景
    MEDICAL_GRAY = 'gray'  # 医学影像标准灰度
    MEDICAL_BONE = 'bone'  # 骨骼成像
    MEDICAL_HOT = 'hot'    # 热图显示
    
    # 深度学习训练场景
    LOSS_CURVE = 'viridis'     # 损失曲线
    ACCURACY_CURVE = 'plasma'   # 准确率曲线
    GRADIENT_FLOW = 'coolwarm'  # 梯度流动
    
    # 数据分析场景
    CORRELATION = 'RdBu_r'     # 相关性矩阵
    HEATMAP_BLUE = 'Blues'     # 蓝色热图
    HEATMAP_RED = 'Reds'       # 红色热图
    DIVERGING = 'RdYlBu_r'     # 发散色彩
    
    # 分类任务场景
    CLASSIFICATION = 'Set1'     # 分类标签
    MULTI_CLASS = 'tab10'      # 多分类
    CONFUSION_MATRIX = 'Blues' # 混淆矩阵
    
    # 回归任务场景
    REGRESSION_PRED = 'viridis'  # 回归预测
    RESIDUAL_PLOT = 'coolwarm'   # 残差图
    
    # 特征重要性场景
    FEATURE_IMPORTANCE = 'YlOrRd'  # 特征重要性
    ATTENTION_MAP = 'Reds'         # 注意力图
    
    # 时间序列场景
    TIME_SERIES = 'plasma'    # 时间序列
    TREND_ANALYSIS = 'viridis' # 趋势分析
    
    # 统计分析场景
    DISTRIBUTION = 'hist'      # 分布图
    DENSITY_PLOT = 'viridis'   # 密度图
    PROBABILITY = 'YlGnBu'     # 概率图
    
    # 网络分析场景
    NETWORK_NODES = 'Set2'     # 网络节点
    GRAPH_EDGES = 'viridis'    # 图边权重
    
    # 地理/空间数据场景
    TERRAIN = 'terrain'        # 地形图
    ELEVATION = 'gist_earth'   # 海拔高度
    
    @classmethod
    def _load_custom_cmaps(cls) -> Dict[str, str]:
        """从配置文件加载自定义cmap设置"""
        cmap_config = _config_loader.get_cmap_config()
        custom_cmaps = {}
        
        # 加载场景级别的重载
        for scenario, presets in cmap_config.items():
            if isinstance(presets, dict):
                for preset_name, cmap_value in presets.items():
                    key = f"{scenario}.{preset_name}"
                    custom_cmaps[key] = cmap_value
            elif isinstance(presets, str):
                # 直接场景级别的设置
                custom_cmaps[scenario] = presets
        
        return custom_cmaps
    
    @classmethod
    def get_cmap(cls, scenario: str, preset_type: str = None) -> str:
        """获取colormap，优先使用配置文件中的设置"""
        custom_cmaps = cls._load_custom_cmaps()
        
        # 构建查找键
        if preset_type:
            lookup_key = f"{scenario}.{preset_type}"
            if lookup_key in custom_cmaps:
                return custom_cmaps[lookup_key]
        
        # 查找场景级别的设置
        if scenario in custom_cmaps:
            return custom_cmaps[scenario]
        
        # 回退到默认预设
        all_presets = cls.get_all_presets()
        if scenario not in all_presets:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(all_presets.keys())}")
        
        scenario_presets = all_presets[scenario]
        
        if preset_type is None:
            return list(scenario_presets.values())[0]
        else:
            if preset_type not in scenario_presets:
                raise ValueError(f"Unknown preset_type: {preset_type}. Available for {scenario}: {list(scenario_presets.keys())}")
            return scenario_presets[preset_type]
    
    @classmethod
    def get_all_presets(cls) -> dict[str, dict[str, str]]:
        """获取所有预设配置"""
        return {
            'medical': {
                'grayscale': cls.MEDICAL_GRAY,
                'bone': cls.MEDICAL_BONE,
                'thermal': cls.MEDICAL_HOT
            },
            'ml_training': {
                'loss': cls.LOSS_CURVE,
                'accuracy': cls.ACCURACY_CURVE,
                'gradient': cls.GRADIENT_FLOW
            },
            'analysis': {
                'correlation': cls.CORRELATION,
                'heatmap_blue': cls.HEATMAP_BLUE,
                'heatmap_red': cls.HEATMAP_RED,
                'diverging': cls.DIVERGING
            },
            'classification': {
                'classes': cls.CLASSIFICATION,
                'multi_class': cls.MULTI_CLASS,
                'confusion': cls.CONFUSION_MATRIX
            },
            'regression': {
                'prediction': cls.REGRESSION_PRED,
                'residual': cls.RESIDUAL_PLOT
            },
            'feature': {
                'importance': cls.FEATURE_IMPORTANCE,
                'attention': cls.ATTENTION_MAP
            },
            'time_series': {
                'series': cls.TIME_SERIES,
                'trend': cls.TREND_ANALYSIS
            },
            'statistical': {
                'distribution': cls.DISTRIBUTION,
                'density': cls.DENSITY_PLOT,
                'probability': cls.PROBABILITY
            },
            'network': {
                'nodes': cls.NETWORK_NODES,
                'edges': cls.GRAPH_EDGES
            },
            'spatial': {
                'terrain': cls.TERRAIN,
                'elevation': cls.ELEVATION
            }
        }

class ThemePresets:
    """预设的主题配置，适用于不同的科学可视化场景"""
    
    # 基础主题
    DEFAULT = "default"
    CLASSIC = "classic"
    SEABORN = "seaborn-v0_8"
    GGPLOT = "ggplot"
    
    # 科学论文主题
    SCIENTIFIC = "scientific"
    PUBLICATION = "publication"
    MINIMAL = "minimal"
    
    # 医学影像主题
    MEDICAL = "medical"
    GRAYSCALE = "grayscale"
    HIGH_CONTRAST = "high_contrast"
    
    # 演示主题
    PRESENTATION = "presentation"
    POSTER = "seaborn-v0_8-poster"
    TALK = "seaborn-v0_8-talk"
    
    # 可访问性主题
    COLORBLIND_FRIENDLY = "colorblind_friendly"
    MONOCHROME = "monochrome"
    DARK_BACKGROUND = "dark_background"
    
    @classmethod
    def _load_custom_themes(cls) -> Dict[str, Any]:
        """从配置文件加载自定义theme设置"""
        theme_config = _config_loader.get_theme_config()
        return theme_config
    
    @classmethod
    def get_theme(cls, category: str, theme_type: str = None) -> str:
        """获取主题，优先使用配置文件中的设置"""
        custom_themes = cls._load_custom_themes()
        
        # 构建查找键
        if theme_type:
            lookup_key = f"{category}.{theme_type}"
            if lookup_key in custom_themes:
                return custom_themes[lookup_key]
        
        # 查找类别级别的设置
        if category in custom_themes:
            category_config = custom_themes[category]
            if isinstance(category_config, str):
                return category_config
            elif isinstance(category_config, dict) and theme_type:
                return category_config.get(theme_type, cls._get_default_theme(category, theme_type))
        
        # 回退到默认预设
        return cls._get_default_theme(category, theme_type)
    
    @classmethod
    def _get_default_theme(cls, category: str, theme_type: str = None) -> str:
        """获取默认主题"""
        all_themes = cls.get_all_themes()
        if category not in all_themes:
            return cls.DEFAULT
        
        category_themes = all_themes[category]
        
        if theme_type is None:
            return category_themes.get('default', cls.DEFAULT)
        else:
            return category_themes.get(theme_type, cls.DEFAULT)
    
    @classmethod
    def get_all_themes(cls) -> dict[str, dict[str, str]]:
        """获取所有预设主题"""
        return {
            'scientific': {
                'default': cls.SCIENTIFIC,
                'publication': cls.PUBLICATION,
                'minimal': cls.MINIMAL,
                'classic': cls.CLASSIC
            },
            'medical': {
                'default': cls.MEDICAL,
                'grayscale': cls.GRAYSCALE,
                'high_contrast': cls.HIGH_CONTRAST,
                'monochrome': cls.MONOCHROME
            },
            'presentation': {
                'default': cls.PRESENTATION,
                'poster': cls.POSTER,
                'talk': cls.TALK,
                'dark': cls.DARK_BACKGROUND
            },
            'accessibility': {
                'colorblind': cls.COLORBLIND_FRIENDLY,
                'monochrome': cls.MONOCHROME,
                'high_contrast': cls.HIGH_CONTRAST,
                'dark': cls.DARK_BACKGROUND
            }
        }
    
    @classmethod
    def apply_theme_with_config(cls, theme_name: str):
        """应用主题配置，支持配置文件重载"""
        custom_themes = cls._load_custom_themes()
        
        # 检查是否有自定义主题配置
        if 'custom_styles' in custom_themes and theme_name in custom_themes['custom_styles']:
            custom_style = custom_themes['custom_styles'][theme_name]
            cls._apply_custom_style(custom_style)
        else:
            # 使用预设主题
            cls.apply_custom_theme(theme_name)
    
    @classmethod
    def apply_custom_theme(cls, theme_name: str):
        """应用自定义主题配置"""
        custom_themes = {
            'scientific': {
                'style': 'seaborn-v0_8-whitegrid',
                'context': 'paper',
                'palette': 'deep',
                'font_scale': 1.0,
                'rc': {
                    'figure.figsize': (8, 6),
                    'font.family': 'serif',
                    'font.serif': ['Times New Roman'],
                    'axes.linewidth': 1.0,
                    'grid.linewidth': 0.5,
                    'lines.linewidth': 1.5
                }
            },
            'medical': {
                'style': 'seaborn-v0_8-white',
                'context': 'notebook',
                'palette': 'muted',
                'font_scale': 1.1,
                'rc': {
                    'figure.figsize': (10, 8),
                    'font.family': 'sans-serif',
                    'font.sans-serif': ['Arial', 'Helvetica'],
                    'axes.linewidth': 1.2,
                    'lines.linewidth': 2.0
                }
            },
            'presentation': {
                'style': 'seaborn-v0_8-darkgrid',
                'context': 'talk',
                'palette': 'bright',
                'font_scale': 1.3,
                'rc': {
                    'figure.figsize': (12, 9),
                    'font.family': 'sans-serif',
                    'axes.linewidth': 1.5,
                    'lines.linewidth': 2.5
                }
            }
        }
        
        if theme_name in custom_themes:
            theme_config = custom_themes[theme_name]
            cls._apply_custom_style(theme_config)
        else:
            # 使用matplotlib内置样式
            try:
                plt.style.use(theme_name)
            except OSError:
                print(f"Warning: Theme '{theme_name}' not found, using default")
                plt.style.use('default')
    
    @classmethod
    def _apply_custom_style(cls, style_config: Dict[str, Any]):
        """应用自定义样式配置"""
        # 设置seaborn样式
        if 'style' in style_config:
            sns.set_style(style_config['style'])
        
        # 设置上下文
        if 'context' in style_config:
            sns.set_context(style_config['context'], 
                           font_scale=style_config.get('font_scale', 1.0))
        
        # 设置调色板
        if 'palette' in style_config:
            sns.set_palette(style_config['palette'])
        
        # 设置matplotlib参数
        if 'rc' in style_config:
            plt.rcParams.update(style_config['rc'])
        
        # 设置自定义参数
        if 'custom_rc' in style_config:
            plt.rcParams.update(style_config['custom_rc'])

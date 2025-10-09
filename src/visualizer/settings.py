import matplotlib.pyplot as plt
from typing import Literal

MARKER = Literal[
    ".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "d", "D", "|", "_"
]
LINE_STYLE = Literal["-", "--", "-.", ":", "", " "]


class PlotSetting:
    def __init__(self):
        self._linewidth = None
        self._color = None
        self._linestyle = None
        self._marker = None
        self._markersize = None
        self._alpha = None
        self._zorder = None
        self._dashes = None
        self._label = None
    
    def linewidth(self, linewidth: float):
        self._linewidth = linewidth
        return self
    
    def color(self, color: str): # '#1ff7b4‘
        self._color = color
        return self

    # 支持LaTex
    def label(self, label: str): 
        self._label = label
        return self

    def linestyle(self, linestyle: LINE_STYLE):
        self._linestyle = linestyle
        return self
    
    def marker(self, marker: MARKER):
        self._marker = marker
        return self
    
    def markersize(self, markersize: float):
        self._markersize = markersize
        return self

    def alpha(self, alpha: float):
        self._alpha = alpha
        return self
    
    def zorder(self, zorder: int):
        self._zorder = zorder
        return self

    # 仅对虚线 -- 有效 
    # (实线长度，空白长度)
    def dashes(self, dashes: tuple[float, float]):
        self._dashes = dashes
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(linewidth: float=1.0, color: str='black', linestyle: LINE_STYLE='-', marker: MARKER=None, markersize: float=None, alpha: float=None, zorder: int=None, dashes: tuple[float, float]=None, label: str=None):
        setting = PlotSetting()
        if linewidth is not None:
            setting.linewidth(linewidth)
        if color is not None:
            setting.color(color)
        if linestyle is not None:
            setting.linestyle(linestyle)
        if marker is not None:
            setting.marker(marker)
        if markersize is not None:
            setting.markersize(markersize)
        if alpha is not None:
            setting.alpha(alpha)
        if zorder is not None:
            setting.zorder(zorder)
        if dashes is not None:
            setting.dashes(dashes)
        if label is not None:
            setting.label(label)
        return setting


class FigureSetting:
    def __init__(self):
        self._figsize = None
        self._dpi = None
        self._facecolor = None
        self._edgecolor = None
        self._frameon = None
        self._tight_layout = None
        self._constrained_layout = None

    def figsize(self, figsize: tuple[float, float]):  # (width, height) in inches
        self._figsize = figsize
        return self

    def dpi(self, dpi: float):
        self._dpi = dpi
        return self

    def facecolor(self, facecolor: str):
        self._facecolor = facecolor
        return self

    def edgecolor(self, edgecolor: str):
        self._edgecolor = edgecolor
        return self

    def frameon(self, frameon: bool):
        self._frameon = frameon
        return self

    def tight_layout(self, tight_layout: bool):
        self._tight_layout = tight_layout
        return self

    def constrained_layout(self, constrained_layout: bool):
        self._constrained_layout = constrained_layout
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(figsize: tuple[float, float]=(8, 6), dpi: float=300, facecolor: str='white', edgecolor: str='white', frameon: bool=True, tight_layout: bool=False, constrained_layout: bool=False):
        setting = FigureSetting()
        if figsize is not None:
            setting.figsize(figsize)
        if dpi is not None:
            setting.dpi(dpi)
        if facecolor is not None:
            setting.facecolor(facecolor)
        if edgecolor is not None:
            setting.edgecolor(edgecolor)
        if frameon is not None:
            setting.frameon(frameon)
        if tight_layout is not None:
            setting.tight_layout(tight_layout)
        if constrained_layout is not None:
            setting.constrained_layout(constrained_layout)
        return setting


class AxisSetting:
    def __init__(self):
        self._xlim = None
        self._ylim = None
        self._xticks = None
        self._yticks = None
        self._xticklabels = None
        self._yticklabels = None
        self._tick_params = None
        self._xscale = None  # 'linear', 'log', 'symlog', 'logit'
        self._yscale = None  # 'linear', 'log', 'symlog', 'logit'

    def xlim(self, xlim: tuple[float, float]):
        self._xlim = xlim
        return self

    def ylim(self, ylim: tuple[float, float]):
        self._ylim = ylim
        return self

    def xticks(self, xticks: list):
        self._xticks = xticks
        return self

    def yticks(self, yticks: list):
        self._yticks = yticks
        return self

    def xticklabels(self, xticklabels: list):
        self._xticklabels = xticklabels
        return self

    def yticklabels(self, yticklabels: list):
        self._yticklabels = yticklabels
        return self

    def tick_params(self, **kwargs):  # 支持matplotlib的tick_params参数
        self._tick_params = kwargs
        return self

    def xscale(self, xscale: str):  # 'linear', 'log', 'symlog', 'logit'
        self._xscale = xscale
        return self

    def yscale(self, yscale: str):  # 'linear', 'log', 'symlog', 'logit'
        self._yscale = yscale
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, xticks: list=None, yticks: list=None, xticklabels: list=None, yticklabels: list=None, tick_params: dict=None, xscale: str='linear', yscale: str='linear'):
        setting = AxisSetting()
        if xlim is not None:
            setting.xlim(xlim)
        if ylim is not None:
            setting.ylim(ylim)
        if xticks is not None:
            setting.xticks(xticks)
        if yticks is not None:
            setting.yticks(yticks)
        if xticklabels is not None:
            setting.xticklabels(xticklabels)
        if yticklabels is not None:
            setting.yticklabels(yticklabels)
        if tick_params is not None:
            setting.tick_params(**tick_params)
        if xscale is not None:
            setting.xscale(xscale)
        if yscale is not None:
            setting.yscale(yscale)
        return setting


class GridSetting:
    def __init__(self):
        self._visible = None
        self._which = None  # 'major', 'minor', 'both'
        self._axis = None   # 'both', 'x', 'y'
        self._color = None
        self._linestyle = None
        self._linewidth = None
        self._alpha = None

    def visible(self, visible: bool):
        self._visible = visible
        return self

    def which(self, which: str):  # 'major', 'minor', 'both'
        self._which = which
        return self

    def axis(self, axis: str):  # 'both', 'x', 'y'
        self._axis = axis
        return self

    def color(self, color: str):
        self._color = color
        return self

    def linestyle(self, linestyle: LINE_STYLE):
        self._linestyle = linestyle
        return self

    def linewidth(self, linewidth: float):
        self._linewidth = linewidth
        return self

    def alpha(self, alpha: float):
        self._alpha = alpha
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(visible: bool=True, which: str='major', axis: str='both', color: str='gray', linestyle: LINE_STYLE='-', linewidth: float=0.5, alpha: float=0.5):
        setting = GridSetting()
        if visible is not None:
            setting.visible(visible)
        if which is not None:
            setting.which(which)
        if axis is not None:
            setting.axis(axis)
        if color is not None:
            setting.color(color)
        if linestyle is not None:
            setting.linestyle(linestyle)
        if linewidth is not None:
            setting.linewidth(linewidth)
        if alpha is not None:
            setting.alpha(alpha)
        return setting


class AxisLabelSetting:
    def __init__(self):
        self._xlabel = None
        self._ylabel = None
        self._fontsize = None
        self._fontweight = None
        self._fontstyle = None
        self._color = None
        self._labelpad = None
        self._rotation = None

    def xlabel(self, xlabel: str):
        self._xlabel = xlabel
        return self

    def ylabel(self, ylabel: str):
        self._ylabel = ylabel
        return self

    def fontsize(self, fontsize: float):
        self._fontsize = fontsize
        return self

    def fontweight(self, fontweight: str):
        self._fontweight = fontweight
        return self

    def fontstyle(self, fontstyle: str):
        self._fontstyle = fontstyle
        return self

    def color(self, color: str):
        self._color = color
        return self

    def labelpad(self, labelpad: float):
        self._labelpad = labelpad
        return self

    def rotation(self, rotation: float):
        self._rotation = rotation
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(xlabel: str=None, ylabel: str=None, fontsize: float=12, fontweight: str='normal', fontstyle: str='normal', color: str='black', labelpad: float=4.0, rotation: float=0):
        setting = AxisLabelSetting()
        if xlabel is not None:
            setting.xlabel(xlabel)
        if ylabel is not None:
            setting.ylabel(ylabel)
        if fontsize is not None:
            setting.fontsize(fontsize)
        if fontweight is not None:
            setting.fontweight(fontweight)
        if fontstyle is not None:
            setting.fontstyle(fontstyle)
        if color is not None:
            setting.color(color)
        if labelpad is not None:
            setting.labelpad(labelpad)
        if rotation is not None:
            setting.rotation(rotation)
        return setting


class LegendSetting:
    def __init__(self):
        self._loc = None
        self._fontsize = None
        self._fontweight = None
        self._fontstyle = None
        self._fancybox = None
        self._shadow = None
        self._frameon = None
        self._facecolor = None
        self._edgecolor = None
        self._mode = None
        self._bbox_to_anchor = None
        self._ncol = None
        self._columnspacing = None
        self._handlelength = None
        self._handletextpad = None

    def loc(self, loc: str):  # 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        self._loc = loc
        return self

    def fontsize(self, fontsize: float):
        self._fontsize = fontsize
        return self

    def fontweight(self, fontweight: str):
        self._fontweight = fontweight
        return self

    def fontstyle(self, fontstyle: str):
        self._fontstyle = fontstyle
        return self

    def fancybox(self, fancybox: bool):
        self._fancybox = fancybox
        return self

    def shadow(self, shadow: bool):
        self._shadow = shadow
        return self

    def frameon(self, frameon: bool):
        self._frameon = frameon
        return self

    def facecolor(self, facecolor: str):
        self._facecolor = facecolor
        return self

    def edgecolor(self, edgecolor: str):
        self._edgecolor = edgecolor
        return self

    def mode(self, mode: str):  # None or "expand"
        self._mode = mode
        return self

    def bbox_to_anchor(self, bbox_to_anchor: tuple):
        self._bbox_to_anchor = bbox_to_anchor
        return self

    def ncol(self, ncol: int):
        self._ncol = ncol
        return self

    def columnspacing(self, columnspacing: float):
        self._columnspacing = columnspacing
        return self

    def handlelength(self, handlelength: float):
        self._handlelength = handlelength
        return self

    def handletextpad(self, handletextpad: float):
        self._handletextpad = handletextpad
        return self

    def build(self):
        return self.__dict__

    @staticmethod
    def default(loc: str='best', fontsize: float=10, fontweight: str='normal', fontstyle: str='normal', fancybox: bool=True, shadow: bool=False, frameon: bool=True, facecolor: str='white', edgecolor: str='black', mode: str=None, bbox_to_anchor: tuple=None, ncol: int=1, columnspacing: float=2.0, handlelength: float=2.0, handletextpad: float=0.8):
        setting = LegendSetting()
        if loc is not None:
            setting.loc(loc)
        if fontsize is not None:
            setting.fontsize(fontsize)
        if fontweight is not None:
            setting.fontweight(fontweight)
        if fontstyle is not None:
            setting.fontstyle(fontstyle)
        if fancybox is not None:
            setting.fancybox(fancybox)
        if shadow is not None:
            setting.shadow(shadow)
        if frameon is not None:
            setting.frameon(frameon)
        if facecolor is not None:
            setting.facecolor(facecolor)
        if edgecolor is not None:
            setting.edgecolor(edgecolor)
        if mode is not None:
            setting.mode(mode)
        if bbox_to_anchor is not None:
            setting.bbox_to_anchor(bbox_to_anchor)
        if ncol is not None:
            setting.ncol(ncol)
        if columnspacing is not None:
            setting.columnspacing(columnspacing)
        if handlelength is not None:
            setting.handlelength(handlelength)
        if handletextpad is not None:
            setting.handletextpad(handletextpad)
        return setting


class TitleSetting:
    def __init__(self):
        self._title = None
        self._fontsize = None
        self._fontweight = None
        self._fontstyle = None
        self._color = None
        self._loc = None
        self._pad = None

    def title(self, title: str):
        self._title = title
        return self

    def fontsize(self, fontsize: float):
        self._fontsize = fontsize
        return self

    def fontweight(self, fontweight: str):
        self._fontweight = fontweight
        return self

    def fontstyle(self, fontstyle: str):
        self._fontstyle = fontstyle
        return self

    def pad(self, pad: float):
        self._pad = pad
        return self

    def color(self, color: str):
        self._color = color
        return self

    def loc(self, loc: str):
        self._loc = loc
        return self

    def build(self):
        return self.__dict__
    
    @staticmethod
    def default(fontsize: float=10, fontweight: str='bold', fontstyle: str='normal', color: str='black', loc: str='center', pad: float=5.0):
        setting = TitleSetting()
        if fontsize is not None:
            setting.fontsize(fontsize)
        if fontweight is not None:
            setting.fontweight(fontweight)
        if fontstyle is not None:
            setting.fontstyle(fontstyle)
        if color is not None:
            setting.color(color)
        if loc is not None:
            setting.loc(loc)
        if pad is not None:
            setting.pad(pad)
        return setting

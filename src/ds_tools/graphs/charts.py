import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.font_manager import FontProperties
from datetime import datetime
from numpy import arange
import data_science_utils.config as cfg


FONT_TEXT = FontProperties(size=6)
TEXT_MARGIN = 0.05


def set_elements(ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    return ax


def set_locators(xvalues: list, ax: plt.Axes = None, rotation: bool = False):
    if isinstance(xvalues[0], datetime):
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
    elif isinstance(xvalues[0], str):
        ax.set_xticks(arange(len(xvalues)))
        if rotation:
            ax.set_xticklabels(xvalues, rotation='45', fontsize='small', ha='center')
        else:
            ax.set_xticklabels(xvalues, fontsize='small', ha='center')
    else:
        ax.set_xlim(xvalues[0], xvalues[-1])
        ax.set_xticks(xvalues)


def bar_chart(xvalues: list, yvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
              percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    set_locators(xvalues, ax=ax, rotation=rotation)
    ax.bar(xvalues, yvalues, edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR, tick_label=xvalues)
    for i in range(len(yvalues)):
        ax.text(i, yvalues[i] + TEXT_MARGIN, f'{yvalues[i]:.2f}', ha='center', fontproperties=FONT_TEXT)
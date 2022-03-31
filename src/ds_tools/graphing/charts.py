import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.font_manager import FontProperties
from datetime import datetime
from numpy import arange
from . import config as cfg

FONT_TEXT = FontProperties(size=6)
TEXT_MARGIN = 0.05

NR_COLUMNS: int = 3
HEIGHT: int = 4
WIDTH_PER_VARIABLE: int = 0.5


def choose_grid(nr):
    if nr < NR_COLUMNS:
        return 1, nr
    else:
        return (nr // NR_COLUMNS, NR_COLUMNS) if nr % NR_COLUMNS == 0 else (nr // NR_COLUMNS + 1, NR_COLUMNS)


def set_elements(ax: plt.Axes = None, title: str = '', x_label: str = '', y_label: str = '', percentage: bool = False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    return ax


def set_locators(x_values: list, ax: plt.Axes = None, rotation: bool = False):
    if isinstance(x_values[0], datetime):
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
    elif isinstance(x_values[0], str):
        ax.set_xticks(arange(len(x_values)))
        if rotation:
            ax.set_xticklabels(x_values, rotation='45', fontsize='small', ha='center')
        else:
            ax.set_xticklabels(x_values, fontsize='small', ha='center')
    else:
        ax.set_xlim(x_values[0], x_values[-1])
        ax.set_xticks(x_values)


def bar_chart(x_values: list, y_values: list, ax: plt.Axes = None, title: str = '', x_label: str = '',
              y_label: str = '',
              percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, x_label=x_label, y_label=y_label, percentage=percentage)
    set_locators(x_values, ax=ax, rotation=rotation)
    ax.bar(x_values, y_values, edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR, tick_label=x_values)
    for i in range(len(y_values)):
        ax.text(i, y_values[i] + TEXT_MARGIN, f'{y_values[i]:.2f}', ha='center', fontproperties=FONT_TEXT)


def multiple_bar_chart(x_values: list, y_values: dict, ax: plt.Axes = None, title: str = '', x_label: str = '',
                       y_label: str = '',
                       percentage: bool = False):
    ax = set_elements(ax=ax, title=title, x_label=x_label, y_label=y_label, percentage=percentage)
    ngroups = len(x_values)
    nseries = len(y_values)
    pos_group = arange(ngroups)
    width = 0.8 / nseries
    pos_center = pos_group + (nseries - 1) * width / 2
    ax.set_xticks(pos_center)
    ax.set_xticklabels(x_values)
    legend = []
    for i, metric in enumerate(y_values):
        ax.bar(pos_group, y_values[metric], width=width, edgecolor=cfg.LINE_COLOR, color=cfg.ACTIVE_COLORS[i])
        values = y_values[metric]
        legend.append(metric)
        for k in range(len(values)):
            ax.text(pos_group[k], values[k] + TEXT_MARGIN, f'{values[k]:.2f}', ha='center', fontproperties=FONT_TEXT)
        pos_group = pos_group + width
    ax.legend(legend, fontsize='x-small', title_fontsize='small')


def plot_line(x_values: list, y_values: list, ax: plt.Axes = None, title: str = '', x_label: str = '',
              y_label: str = '',
              percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, x_label=x_label, y_label=y_label, percentage=percentage)
    set_locators(x_values, ax=ax, rotation=rotation)
    ax.plot(x_values, y_values, c=cfg.LINE_COLOR, linewidth=0.5)


def calculated_rolling_mean_dev(df):
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()


def plot_forecasting(train, test, pred, ax=None, x_label: str = 'time', y_label: str = ''):
    plt.figure(figsize=(24, 10))
    if ax is None:
        ax = plt.gca()
    ax.plot(train, label='train', linewidth=0.5, color='blue')
    ax.plot(test, label='test', linewidth=0.5, color='green')
    ax.plot(pred.index, pred.values, label='predicted', color='yellow', linewidth=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend(['train', 'test', 'predicted'])
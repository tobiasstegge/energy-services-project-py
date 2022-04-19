from re import L
from matplotlib.pyplot import savefig, figure, subplots, title, plot, subplots, legend, subplot
from .utils import get_variable_types
from .graphing.charts import bar_chart, choose_grid, HEIGHT, multiple_bar_chart, plot_line
from seaborn import heatmap
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from numpy import cumsum
from statsmodels.tsa.stattools import adfuller


def show_dimensionality(df):
    nr_records = df.shape[0]
    nr_variables = df.shape[1]
    bar_chart(['Nr. of Variables,', 'Nr. of Variables'], [nr_variables, nr_records])


def show_variable_types(df):
    types = get_variable_types(df)
    counts = {}
    for tp in types.keys():
        counts[tp] = len(types[tp])
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')


def show_missing_values(df):
    figure()
    missing_values = {}
    for var in df:
        amount_missing = df[var].isna().sum()
        missing_values[var] = amount_missing
    bar_chart(list(missing_values.keys()), list(missing_values.values()), title='Nr of missing values per variable', rotation=True)


def show_distribution(df):
    if not numeric_vars:
        raise ValueError('There are no numeric variables.')
    numeric_vars = get_variable_types(df)['Numeric']
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s' % numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)


def show_outliers(df, file_path, file_name=''):
    NR_STDEV: int = 2
    numeric_vars = get_variable_types(df)['Numeric']
    if not numeric_vars:
        raise ValueError('There are no numeric variables.')
    outliers_iqr = []
    outliers_stdev = []
    summary = df.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary[var]['75%'] - summary[var]['25%'])
        outliers_iqr += [
            df[df[var] > summary[var]['75%'] + iqr].count()[var] +
            df[df[var] < summary[var]['25%'] - iqr].count()[var]]
        std = NR_STDEV * summary[var]['std']
        outliers_stdev += [
            df[df[var] > summary[var]['mean'] + std].count()[var] +
            df[df[var] < summary[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', x_label='variables',
                       y_label='nr outliers', percentage=False)
    savefig(f'{file_path}/outliers_{file_name}.png')


def show_histograms_numeric(df):
    numeric_vars = get_variable_types(df)['Numeric']
    if not numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(df[numeric_vars[n]].dropna().values, bins=100)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)


def show_histograms_symbolic(df):
    symbolic_vars = get_variable_types(df)['Symbolic']
    if not symbolic_vars:
        raise ValueError('There are no symbolic variables.')
    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = df[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n],
                  xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)


def show_sparsity(df):
    numeric_vars = get_variable_types(df)['Numeric']
    if not numeric_vars or len(numeric_vars) < 2:
        raise ValueError('There are no or not enough numeric variables.')

    rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i + 1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(df[var1], df[var2])


def show_heatmap(df):
    numeric_vars = get_variable_types(df)['Numeric']
    data_numeric = df[numeric_vars]
    figure(figsize=[16, 16])
    corr_mtx = abs(data_numeric.corr())
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation for Numeric Variables')



def plot_timeseries(df, columns, y_labels, start=None, end=None):
    fig, ax1 = subplots(figsize=(16, 6))
    if len(columns) == 1:
        if start and end:
            plot_line(x_values=df[start:end].index, y_values=df[columns[0]][start:end], y_label=y_labels[0])
        else:
            plot_line(x_values=df.index, y_values=df[columns[0]], x_label='Time', y_label=y_labels[0])
    else:
        if start and end:
            ax1.set_ylabel(y_labels[0], color='tab:red')
            ax1.plot(df[start:end].index, df[columns[0]][start:end], color='tab:red', linewidth=0.5)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_labels[1], color='tab:blue')
            ax2.plot(df[start:end].index, df[columns[1]][start:end], color='tab:blue', linewidth=0.5)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
        else:
            fig, ax1 = subplots()
            ax1.set_ylabel(y_labels[0], color='tab:red')
            ax1.plot(df.index, df[columns[0]], color='tab:red', linewidth=0.5)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_labels[1], color='tab:blue')
            ax2.plot(df.index, df[columns[1]], color='tab:blue', linewidth=0.5)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
    legend()


def scatter_timeseries(df, columns, y_labels, start=None, end=None):
    fig, ax1 = subplots(figsize=(16, 6))
    if len(columns) == 1:
        if start and end:
            plot_line(x_values=df[start:end].index, y_values=df[columns[0]][start:end], y_label=y_labels[0])
        else:
            plot_line(x_values=df.index, y_values=df[columns[0]], x_label='Time', y_label=y_labels[0])
    else:
        if start and end:
            ax1.set_ylabel(y_labels[0], color='tab:red')
            ax1.scatter(df[start:end].index, df[columns[0]][start:end], color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_labels[1], color='tab:blue')
            ax2.scatter(df[start:end].index, df[columns[1]][start:end], color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
        else:
            fig, ax1 = subplots()
            ax1.set_ylabel(y_labels[0], color='tab:red')
            ax1.scatter(df.index, df[columns[0]], color='tab:red', linewidth=0.5)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_labels[1], color='tab:blue')
            ax2.scatter(df.index, df[columns[1]], color='tab:blue', linewidth=0.5)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
    legend()


def plot_rolling_mean_dev(df, column, window, y_label='', start=None, end=None):
    figure(figsize=(24, 8))
    title("Rolling Mean and Standard Deviation")
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    if start and end:
        plot(df[start:end].index, df[column][start:end], label='original', linewidth=0.5, color='black')
        plot(df[start:end].index, rolling_mean[start:end], label='red', linewidth=0.5, color='blue')
        plot(df[start:end].index, rolling_std[start:end], label='std', linewidth=0.5, color='red')
    else:
        plot(df.index, df[column], label='original', linewidth=0.5, color='black')
        plot(df.index, rolling_mean, label='red', linewidth=0.5, color='blue')
        plot(df.index, rolling_std, label='std', linewidth=0.5, color='red')
    legend()


def plot_seasonal_decompose(df, column, period):
    figure(figsize=(24, 8))
    result = seasonal_decompose(df[column], model='additive', period=period)
    subplot(411)
    plot(df[column], label='Original', color='black', linewidth=0.5)
    legend(loc='upper left')
    subplot(412)
    plot(result.trend, label='Trend', color='black', linewidth=0.5)
    legend(loc='upper left')
    subplot(413)
    plot(result.seasonal, label='Seasonal', color='black', linewidth=0.5)
    legend(loc='upper left')
    subplot(414)
    plot(result.resid, label='Residual', color='black', linewidth=0.5)
    legend(loc='upper left')



def pca(df, n_components):
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(df)
    features = range(pca.n_components_)
    bar_chart(features, pca.explained_variance_ratio_, title='PCA Analysis', x_label='features', y_label='% variance explained')
    plot(cumsum(pca.explained_variance_ratio_), color='black', linewidth=0.5)
    return result


def adfuller_test(df):
    "if above 0.05, data is not stationary"
    adf, p_value, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
    return p_value
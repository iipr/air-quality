import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn import linear_model

class Plotter():

    def __init__(self, lw=1.75, # linewidth
                 title_fs=30, tick_fs=20, legend_fs=25, # fontsize
                ):
        self.title_fs = title_fs
        self.tick_fs = tick_fs
        self.legend_fs = legend_fs
        self.lw = lw

    def series(self, series_d, labels, dims=None, date_ticker=None,
                    int_ticker=False, tickers=None, style='-', grid=True,
                    xlim=None, ylim=None, scale=1, yscale='linear', out_path='tmp.png'):
        if not dims:
            dims = (len(series_d), 1)
        fig, axs = plt.subplots(*dims)
        axs = axs.ravel() if dims != (1, 1) else [axs]
        fig.set_size_inches(8 * (dims[1] if dims[1] > 1 else 2), 5 * (dims[0] if dims[0] > 1 else 2))
        for idx, (title, sub_d) in enumerate(series_d.items()):
            for s_name, series in sub_d.items():
                if date_ticker is not None:
                    axs[idx].plot(date_ticker[:, idx // dims[1]], series, style, linewidth=self.lw, label=s_name)
                elif tickers is not None:
                    axs[idx].plot(tickers, series, style, linewidth=self.lw, label=s_name)
                else:
                    axs[idx].plot(series, style, linewidth=self.lw, label=s_name)
            self.__format(axs[idx], title, labels, grid, tickers=tickers, int_ticker=int_ticker,
                               date_ticker=date_ticker, xlim=xlim, ylim=ylim, scale=scale, yscale=yscale)
        fig.tight_layout()
        # Save to file
        self.__save(fig, out_path)

    def scatters(self, series_d, labels, min_max=None, dims=None,
                      style='g.', grid=True, scale=1, out_path='tmp.png'):
        if not dims:
            dims = (len(series_d), 1)
        fig, axs = plt.subplots(*dims)
        axs = axs.ravel() if dims != (1, 1) else [axs]
        fig.set_size_inches(15, 5 * (dims[0] if dims[0] > 1 else 2))
        for idx, (title, sub_d) in enumerate(series_d.items()):
            (s1_name, s1), (s2_name, s2) = sub_d.items()
            axs[idx].plot(s1, s2, style, label='{} vs {}'.format(s1_name, s2_name))
            if min_max:
                # Create linear regression object
                regr = linear_model.LinearRegression()
                # Train the model using s2
                regr.fit(X=s2.reshape(-1, 1), y=s1)
                # Make predictions using the testing set
                line = regr.predict(s2.reshape(-1, 1))
                axs[idx].plot(s2, line, color='royalblue', label='LR of {} vs {}'.format(s1_name, s2_name))
                axs[idx].plot(min_max, min_max, color='r', label='y(x) = x')
            self.__format(axs[idx], title, labels, grid, min_max=min_max, scale=scale)
        if len(axs) >= len(series_d):
            axs[-1].set_axis_off()
        fig.tight_layout()
        # Save to file
        self.__save(fig, out_path)

    def boxplot(self, data, x_labels, labels, title,
                     flierprops={'markerfacecolor':'g', 'marker': 'o'},
                     grid=True, scale=1, yscale='linear', out_path='tmp.png'):
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        ax.boxplot(data, labels=x_labels, flierprops=flierprops)
        self.__format(ax, title, labels, grid,
                           legend=False, scale=scale, yscale=yscale)
        # Save to file
        self.__save(fig, out_path)

    def __format(self, ax, title, labels, grid, legend=True, tickers=None,
                 int_ticker=False, date_ticker=None, min_max=False,
                 loc='best', xlim=None, ylim=None, scale=1, yscale=None):
        ax.set_title(title, fontsize=int(self.title_fs * scale))
        ax.set_xlabel(labels[0], fontsize=int(self.title_fs * scale))
        ax.set_ylabel(labels[1], fontsize=int(self.title_fs * scale))
        ax.xaxis.set_tick_params(labelsize=int(self.tick_fs * scale))
        ax.yaxis.set_tick_params(labelsize=int(self.tick_fs * scale))
        if legend:
            ax.legend(loc=loc, fontsize=int(self.legend_fs * scale))
        if grid:
            ax.grid()
        if tickers:
            ax.xaxis.set_major_locator(mticker.FixedLocator(tickers))
        if int_ticker:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: int(x+1)))
        if date_ticker is not None:
            #ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        if min_max:
            ax.set_aspect('equal')
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if yscale:
            ax.set_yscale(yscale)

    def __save(self, fig, out_path):
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print('\n\tPlot saved to {}'.format(out_path))

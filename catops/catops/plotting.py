"""
Plotting is a suite of functions that perform specific data visualisation
operations on the catalog.
"""
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def quick_inspect_amplitudes(N: np.array,
                             E: np.array,
                             C: np.array,
                             s=1,
                             logx=False,
                             logy=False,
                             loglog=False,
                             cmap='viridis',
                             **kwargs
                             ) -> None:
    """This functions requires the """
    # computations needed later
    s_scale = s
    diff = N - E
    s = np.std(diff)  # std dv of histogram differences
    weights = np.ones_like(diff) / float(len(diff))  # weights for histogram
    # setup figures
    fig, axes = plt.subplots(1, 2, figsize=(7 * 2.5, 6))
    # direct plot
    h = axes[0].scatter(N, E, c=C, cmap=cmap, **kwargs)
    oneone = [np.min([N.min(), E.min()]) * 1.1,
              np.max([N.max(), E.max()]) * 1.1]
    axes[0].plot(oneone, oneone, 'k--', label="1:1")
    axes[0].plot(oneone, oneone + s * s_scale, 'r--',
                 label=f"+/- {s_scale}s: {(s_scale * s):.2f}"
                 )
    axes[0].plot(oneone, oneone - s * s_scale, 'r--')
    xlabel = "log10 p-p E-W [mm]"
    ylabel = "log10 p-p N-S [mm]"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.colorbar(h, ax=axes[0])
    # histogram of p-p differences
    out = axes[1].hist(diff, bins=30, weights=weights, edgecolor='black',
                       color='blue'
                       )
    axes[1].vlines(s * s_scale, 0, out[0].max(), color='r',
                   linestyles='dashed',
                   label=f"+/- {s_scale}s: {(s_scale * s):.2f}"
                   )
    axes[1].vlines(-s * s_scale, 0, out[0].max(),
                   color='r', linestyles='dashed'
                   )
    axes[1].legend()
    axes[1].set_xlabel("log10 p-p differences")


def magnitude_distance_plot(M: np.array,
                            Dist: np.array,
                            Dep: np.array,
                            A: np.array
                            ) -> None:
    """

    """
    hkwargs = dict(bottom=0.0, color='.8', edgecolor='k', rwidth=0.8,
                   weights=np.zeros_like(Dist) + 1. / len(Dist)
                   )
    hdepkwargs = dict(bottom=0.0, color='.8', edgecolor='k', rwidth=0.8,
                      weights=np.zeros_like(Dep) + 1. / len(Dep)
                      )
    # figure stuff
    fac = 2.6
    fig = plt.figure(constrained_layout=False, figsize=(7 * fac, 3 * fac))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48,
                           wspace=0.2, hspace=0.05
                           )
    ax1 = fig.add_subplot(gs1[:1, 0:-1])
    ax2 = fig.add_subplot(gs1[1:, :-1])

    ax3 = fig.add_subplot(gs1[1:, -1:])
    ax4 = fig.add_subplot(gs1[0, -1])
    # settings
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel("Frac. in bin")
    ax1.set_xlim(np.log10([1, 200]))
    # ax1.yaxis.set_ticks([0.05, 0.1, 0.15, 0.2])

    ax2.set_xscale('log')
    ax2.set_ylabel('Cat. Mag.')
    ax2.set_xlabel('Hypo. Dist. [km]')

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(
            np.maximum(-np.log10(y), 0)))).format(y)))
    ax2.set_xlim([1, 200])
    # ax2.xaxis.set_ticks([1, 10, 100, 1000])

    ax3.yaxis.set_visible(False)
    # ax3.xaxis.set_ticks([0.1, 0.2, 0.3])
    ax3.set_ylabel('Cat. Mag.')
    ax3.set_xlabel("Frac. in bin")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    ax4.set_xlabel("Depth [km]")
    ax4.xaxis.set_label_position("top")
    ax4.yaxis.tick_right()
    # ax4.yaxis.set_ticks([0.1, 0.2, 0.3])
    ax4.xaxis.tick_top()

    ax1.hist(np.log10(Dist), **hkwargs)
    ax3.hist(M, orientation='horizontal', **hkwargs)
    ax4.hist(Dep, **hdepkwargs)
    sout = ax2.scatter(Dist, M, c=A, lw=1, cmap='viridis', s=10)

    cbaxes = inset_axes(ax2, width="35%", height="3%", loc=2)
    cbar = fig.colorbar(sout, cax=cbaxes, orientation="horizontal")
    cbar.set_label(r"$\mathrm{log_{10}}(A[mm]$)", rotation=0,
                   fontsize=14, horizontalalignment='center')



def spatial_distribution_plot(Lon: np.array,
                              Lat: np.array,
                              Dep: np.array
                              ) -> None:

    hkwargs = dict(bottom=0.0, color='.8', edgecolor='k', rwidth=0.8,
                   weights=np.zeros_like(Lon) + 1. / len(Lon))

    # figure stuff
    fac = 2.75
    fig = plt.figure(constrained_layout=False, figsize=(7 * fac, 3 * fac))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48,
                           wspace=0.15, hspace=0.15)
    ax1 = fig.add_subplot(gs1[:1, 0:-1])
    ax2 = fig.add_subplot(gs1[1:, :-1])
    ax3 = fig.add_subplot(gs1[1:, -1:])
    ax4 = fig.add_subplot(gs1[0, -1])

    ax1.xaxis.set_visible(False)
    ax1.set_ylabel("Frac. in bin")
    # ax1.yaxis.set_ticks([0.05, 0.1, 0.15, 0.2])

    ax2.set_ylabel('Latitude [deg]')
    ax2.set_xlabel('Longitude [deg]')
    #     ax2.xaxis.set_ticks([1, 10, 100, 1000])

    ax3.yaxis.set_visible(False)
    # ax3.xaxis.set_ticks([0.1, 0.2, 0.3])
    ax3.set_ylabel('Cat. Mag.')
    ax3.set_xlabel("Frac. in bin")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    ax4.set_xlabel("Depth [km]")
    ax4.xaxis.set_label_position("top")
    ax4.yaxis.tick_right()
    # ax4.yaxis.set_ticks([0.1, 0.2, 0.3])
    ax4.xaxis.tick_top()

    ax1.hist(Lon, **hkwargs)
    ax3.hist(Lat, orientation='horizontal', **hkwargs)
    ax4.hist(Dep, **hkwargs)
    sout = ax2.scatter(Lon, Lat, c=Dep, lw=1, cmap='Greys_r', s=10)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    cbaxes = inset_axes(ax2, width="2.5%", height="55%", loc="lower left")
    cbar = fig.colorbar(sout, cax=cbaxes)
    cbar.set_label("Depth [km]", rotation=90)

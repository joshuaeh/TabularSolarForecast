#!/usr/bin/env python3
# Author: Joshua Hammond
# Common graphing and display parameters

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

class TickRedrawer(matplotlib.artist.Artist):
    """Artist to redraw ticks.
    To use, add the line `ax.add_artist(TickRedrawer())` when creating the plot."""

    __name__ = "ticks"

    zorder = 10

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer: matplotlib.backend_bases.RendererBase) -> None:
        """Draw the ticks."""
        if not self.get_visible():
            self.stale = False
            return

        renderer.open_group(self.__name__, gid=self.get_gid())

        for axis in (self.axes.xaxis, self.axes.yaxis):
            loc_min, loc_max = axis.get_view_interval()

            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_visible() and loc_min <= tick.get_loc() <= loc_max:
                    for artist in (tick.tick1line, tick.tick2line):
                        artist.draw(renderer)

        renderer.close_group(self.__name__)
        self.stale = False

# misc declarations
pd.options.display.max_rows = 300
pd.options.display.max_columns = 300
pd.plotting.register_matplotlib_converters()

# random seed
rng = np.random.default_rng(seed=42)

# plot style
# sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("colorblind"))
# sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# sns.set_palette("colorblind")
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_style({"xtick.bottom": True, "ytick.left": True})

plt.rc('font', family='sans-serif', size=9)
# plt.rc('font.size', 9)
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')

plt.rc("savefig", dpi=1_000, bbox="tight", pad_inches=0.01)
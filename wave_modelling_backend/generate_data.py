from typing import Any
import numpy as np
import mantis_core.utilities as manUtil
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple


plt.style.use("plotting.mantis_plotting")


def load_data_to_initial_model(file="input.csv"):
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    cij_array = np.empty((len(data), 6, 6))
    rho_array = np.empty(len(data))
    depth_array = np.empty(len(data))
    for i, d in enumerate(data):
        cij_array[i] = manUtil.VtoCij(Vp=d[1], Vs=d[2], Rho=d[3])
    rho_array = data[:, 3]
    depth_array = data[:, 0] / 1000.0
    result = {
        "top halfspace": {"Vp": data[0, 1], "Vs": data[0, 2], "Rho": data[0, 3]},
        "reflection zone": {"Cij_array": cij_array[1:-1], "Rho_array": rho_array[1:-1]},
        "bottom halfspace": {"Vp": data[-1, 1], "Vs": data[-1, 2], "Rho": data[-1, 3]},
        "Depth array": depth_array,
        "Vp array": data[:, 1],
        "Vs array": data[:, 2],
    }
    return result


class WigglePlot:
    epsilon = 0.001  # to avoid division by zero
    exag = {"small": 1, "normal": 1.5, "large": 4.0}
    layouts = {"tight": 0.6, "normal": 1, "loose": 2.0}

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_range: Tuple[float, float] | None = None,
        x_range: Tuple[float, float] | None = None,
        exaggeration: str = "small",
        layout: str = "tight",
    ):
        if x is None or y is None:
            raise ValueError("x and y must be provided")
        if len(x.shape) != 1:
            raise ValueError(f"x must be 1D array, but got {x.shape}")
        self.x = x
        if len(y.shape) != 2:
            raise ValueError(f"y must be 2D array, but got {y.shape}")
        self.y = (lambda x: (x) / (x.max() - x.min() + WigglePlot.epsilon))(
            y
        )  # scale to -1, 1
        self._l, self._m = y.shape
        self._exaggeration = exaggeration
        self._layout = layout
        self._y_range = y_range
        self._x_range = x_range
        self.fig = None
        self.ax = None

    @property
    def y_range(self):
        return self._y_range

    @y_range.setter
    def y_range(self, value: tuple[float, float]):
        if len(value) != 2:
            raise ValueError(f"y_range must be a tuple of length 2, but got {value}")
        if value[0] > value[1]:
            raise ValueError(
                f"y_range must be a tuple of increasing values, but got {value}"
            )
        self._y_range = value

    @property
    def x_range(self):
        return self._x_range

    @x_range.setter
    def x_range(self, value: tuple[float, float]):
        if len(value) != 2:
            raise ValueError(f"x_range must be a tuple of length 2, but got {value}")
        if value[0] > value[1]:
            raise ValueError(
                f"x_range must be a tuple of increasing values, but got {value}"
            )
        if value[0] < np.min(self.x) or value[1] > np.max(self.x):
            raise ValueError(
                f"x_range must be a tuple of values within the range of x, but got {value}"
            )
        self._x_range = value

    @property
    def exaggeration(self):
        return self._exaggeration

    @exaggeration.setter
    def exaggeration(self, value: str):
        if value not in WigglePlot.exag.keys():
            raise ValueError(
                f"Exaggeration must be one of {WigglePlot.exag.keys()}, but got {value}"
            )
        self._exaggeration = value

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value: str):
        if value not in WigglePlot.layouts.keys():
            raise ValueError(
                f"Layout must be one of {WigglePlot.layouts.keys()}, but got {value}"
            )
        self._layout = value

    def ax_return(self, ax):
        colors = plt.rcParams["axes.prop_cycle"]
        over_color = colors.by_key()["color"][1]
        under_color = colors.by_key()["color"][2]
        line_color = "#333333"
        # line_color = theme.ColourScheme.plot_backgrounds["dark"]
        line_widths = 0.5
        scale = WigglePlot.exag[self._exaggeration]
        # for pos, col in highlight.items():
        #     ax.axhline(pos, color=col, linewidth=2 * line_widths)

        if self._l > self._m:
            assert self.x.shape == (
                self._l,
            ), f"x must be 1D array of length {self._l}, but got {self.x.shape}"
            tick_positions = np.array([1 + i for i in range(self._m)])
            plotdata = [1 + i + scale * j for i, j in enumerate(self.y.T)]
            numplots = self._m
        else:
            assert self.x.shape == (
                self._m,
            ), f"x must be 1D array of length {self._m}, but got {self.x.shape}"
            tick_positions = np.array([1 + i for i in range(self._l)])
            plotdata = [1 + i + scale * j for i, j in enumerate(self.y)]
            numplots = self._l
        if self.x_range is not None:
            ax.set_ylim(self.x_range[0], self.x_range[1])
            normal_aspect = numplots / (self.x_range[1] - self.x_range[0])
        else:
            ax.set_ylim(np.min(self.x), np.max(self.x))
            normal_aspect = numplots / (np.max(self.x) - np.min(self.x))
        aspect_ratio = normal_aspect / WigglePlot.layouts[self._layout]
        segs = [np.column_stack([y, self.x]) for y in plotdata]
        ax.set_xlim(np.min(plotdata), np.max(plotdata))
        ax.set_aspect(aspect_ratio)
        ax.grid(True, which="both", axis="x", alpha=0.3)
        if self.y_range is not None:
            y_labels = np.round(
                np.linspace(self.y_range[0], self.y_range[1], len(tick_positions)), 1
            )
        else:
            y_labels = tick_positions

        layout_ticks = int(5 * WigglePlot.layouts[self._layout])
        if len(tick_positions) > layout_ticks:
            tick_space = len(tick_positions) // layout_ticks
        else:
            tick_space = 1
        ax.set_xticks(tick_positions[0::tick_space])
        ax.set_xticklabels([str(i) for i in y_labels[0::tick_space]])
        line_segments = LineCollection(
            segs,
            array=self.x,
            linestyles="solid",
            linewidths=line_widths,
            colors=line_color,
        )
        for i, pos in enumerate(tick_positions):
            ax.fill_betweenx(
                self.x,
                plotdata[i],
                pos,
                where=plotdata[i] > pos,
                facecolor=over_color,
                interpolate=True,
                alpha=0.95,
            )
            ax.fill_betweenx(
                self.x,
                plotdata[i],
                pos,
                where=plotdata[i] < pos,
                facecolor=under_color,
                interpolate=True,
                alpha=0.75,
            )

        ax.add_collection(line_segments)
        return ax

    def primitives(self, *, highlight: dict = {}):
        self.fig, self.ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"]
        over_color = colors.by_key()["color"][1]
        under_color = colors.by_key()["color"][2]
        line_color = theme.ColourScheme.plot_backgrounds["dark"]
        line_widths = 0.5
        scale = WigglePlot.exag[self._exaggeration]
        for pos, col in highlight.items():
            self.ax.axhline(pos, color=col, linewidth=2 * line_widths)

        if self._l > self._m:
            assert self.x.shape == (
                self._l,
            ), f"x must be 1D array of length {self._l}, but got {self.x.shape}"
            tick_positions = np.array([1 + i for i in range(self._m)])
            plotdata = [1 + i + scale * j for i, j in enumerate(self.y.T)]
            numplots = self._m
        else:
            assert self.x.shape == (
                self._m,
            ), f"x must be 1D array of length {self._m}, but got {self.x.shape}"
            tick_positions = np.array([1 + i for i in range(self._l)])
            plotdata = [1 + i + scale * j for i, j in enumerate(self.y)]
            numplots = self._l
        if self.x_range is not None:
            self.ax.set_ylim(self.x_range[0], self.x_range[1])
            normal_aspect = numplots / (self.x_range[1] - self.x_range[0])
        else:
            self.ax.set_ylim(np.min(self.x), np.max(self.x))
            normal_aspect = numplots / (np.max(self.x) - np.min(self.x))
        aspect_ratio = normal_aspect / WigglePlot.layouts[self._layout]
        segs = [np.column_stack([y, self.x]) for y in plotdata]
        self.ax.set_xlim(np.min(plotdata), np.max(plotdata))
        self.ax.set_aspect(aspect_ratio)
        self.ax.grid(True, which="both", axis="x", alpha=0.3)
        if self.y_range is not None:
            y_labels = np.round(
                np.linspace(self.y_range[0], self.y_range[1], len(tick_positions)), 1
            )
        else:
            y_labels = tick_positions

        layout_ticks = int(5 * WigglePlot.layouts[self._layout])
        if len(tick_positions) > layout_ticks:
            tick_space = len(tick_positions) // layout_ticks
        else:
            tick_space = 1
        self.ax.set_xticks(tick_positions[0::tick_space])
        self.ax.set_xticklabels([str(i) for i in y_labels[0::tick_space]])
        line_segments = LineCollection(
            segs,
            array=self.x,
            linestyles="solid",
            linewidths=line_widths,
            colors=line_color,
        )
        for i, pos in enumerate(tick_positions):
            self.ax.fill_betweenx(
                self.x,
                plotdata[i],
                pos,
                where=plotdata[i] > pos,
                facecolor=over_color,
                interpolate=True,
                alpha=0.95,
            )
            self.ax.fill_betweenx(
                self.x,
                plotdata[i],
                pos,
                where=plotdata[i] < pos,
                facecolor=under_color,
                interpolate=True,
                alpha=0.75,
            )

        self.ax.add_collection(line_segments)
        plt.gca().invert_yaxis()
        plt.show()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.fig, self.ax = plt.subplots()
        self.ax_return(self.ax, *args, **kwds)

        return self.fig, self.ax

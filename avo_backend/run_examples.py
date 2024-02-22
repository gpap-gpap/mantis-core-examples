from __future__ import annotations
from typing import Callable
import mantis_core._literature as mex
import mantis_core.interface as MANint
import mantis_core.utilities as MANut
from rich.console import Console
import matplotlib.pyplot as plt
import numpy as np


# plt.style.use("..plottin.mantis_plotting")
resolution: int = 128
s0 = np.linspace(0.0, 0.35, resolution)
labels = ["φ = 0", "φ = π/6", "φ = π/2", "φ = 2π/3"]


def create_SP_from_paper(
    paper: str, medium: str, relative_azimuth: float = 0.0
) -> MANint.SchoenbergProtazio:
    try:
        p = mex.Examples.presets[paper]
    except KeyError:
        print(
            f"paper must be a valid paper in the presets: {list(mex.Examples.presets.keys())}, got {paper} instead"
        )
    try:
        m = p[medium]
    except KeyError:
        print(
            f"medium must be a valid medium in the presets: {list(p.keys())}, got {medium} instead"
        )
    if m["symmetry"][1] == "Isotropic":
        cij = MANut.VtoCij(Vp=m["vp"], Vs=m["vs"], Rho=m["rho"])
        rho = m["rho"]
    else:
        cij = m["Cij"]
        rho = m["density"]
    if relative_azimuth != 0.0:
        cij = MANut.azimuthal_rotation(cij, relative_azimuth)
    return MANint.SchoenbergProtazio(Cij=cij, density=rho)


def run_example(
    paper: str, medium1: str, medium2: str, relative_azimuth=0.0
) -> tuple[Callable, str]:
    assert paper in mex.Examples.presets.keys(), "paper must be a valid preset"
    assert (
        medium1 in mex.Examples.presets[paper].keys()
        and medium2 in mex.Examples.presets[paper].keys()
    ), "media must be defined"
    _spTop = create_SP_from_paper(paper, medium1)
    _spBot = create_SP_from_paper(paper, medium2, relative_azimuth)
    instance = MANint.ReflectionTransmissionMatrix.sp_init(spUp=_spTop, spDown=_spBot)
    description = (
        f"Paper: {paper}, \nTop halfspace: {medium1}, \nBottom halfspace: {medium2}"
    )
    return (
        lambda horiz: instance(horizontal_slowness=horiz),
        description,
    )


def plotting(
    data: np.ndarray = np.zeros(shape=(1, resolution)),
    labels: list[str] = ["default"],
    title: str = "default",
    y_range: tuple[float, float] = (-1.0, 1.0),
):
    assert data.shape == (
        len(labels),
        resolution,
    ), f"data must be a 2D array with shape ({len(labels)}, {resolution}), got {data.shape} instead"
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.2)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.grid(visible=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(s0.min(), s0.max())
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel("radial slowness (s/km)", fontsize=10)
    ax.set_ylabel("real part of reflection/transmission coefficient", fontsize=10)
    colours = ["darkblue", "black", "red", "green", "orange", "purple"]
    for dat in enumerate(data):
        ax.plot(s0, dat[1], color=colours[dat[0]])
    ax.legend(labels)
    return plt.show()

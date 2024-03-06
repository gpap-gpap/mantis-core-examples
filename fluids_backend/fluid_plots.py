import matplotlib.pyplot as plt
import numpy as np
from mantis_core.rock_physics.fluid_presets import presets as manPResets
import mantis_core.rock_physics.fluid as manFL

plt.style.use("plotting.mantis_plotting")


def initialise_constant_plot(*, constant: str = "Temperature"):
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    if constant not in ["Temperature", "Pressure"]:
        raise ValueError("Constant not found")
    if constant == "Temperature":
        for axis in ax:
            axis.set_xlabel("Pressure (MPa)")
    else:
        for axis in ax:
            axis.set_xlabel("Temperature (C)")
    return fig, ax


def finalise_constant_plot(fig, ax, constant: str = "Temperature"):
    fig.tight_layout()
    if constant == "Temperature":
        fig.suptitle("Fluid Properties by Pressure")
    else:
        fig.suptitle("Fluid Properties by Temperature")
    plt.show()
    plt.close()


def plot_fluid_by_pressure(*, ax, fluid_name, temperature):
    fluid_names = ["CarbonDioxide", "Water", "Methane", "Hydrogen"]
    fluid_properties = {
        "Density": "Density (g/cm^3)",
        "Modulus": "Bulk Modulus (GPa)",
        "Viscosity": "Viscosity (Pa.s)",
    }
    if fluid_name not in fluid_names:
        raise ValueError("Fluid not found")
    for i, prop in enumerate(fluid_properties):
        ax[i].set_ylabel(fluid_properties[prop])
        manPResets.ReadPresets(fluid_name=fluid_name).data[prop].sel(
            Temperature=temperature, method="nearest"
        ).plot(ax=ax[i], label=f"{fluid_name} at {temperature}C")
        ax[i].set_title("")


def plot_fluid_by_temperature(*, ax, fluid_name, pressure):
    fluid_names = ["CarbonDioxide", "Water", "Methane", "Hydrogen"]
    fluid_properties = {
        "Density": "Density (g/cm^3)",
        "Modulus": "Bulk Modulus (GPa)",
        "Viscosity": "Viscosity (Pa.s)",
    }
    if fluid_name not in fluid_names:
        raise ValueError("Fluid not found")
    for i, prop in enumerate(fluid_properties):
        ax[i].set_ylabel(fluid_properties[prop])
        manPResets.ReadPresets(fluid_name=fluid_name).data[prop].sel(
            Pressure=pressure, method="nearest"
        ).plot(ax=ax[i], label=f"{fluid_name} at {pressure}MPa")
        ax[i].set_title("")


def initialise_saturation_plot():
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    for axis in ax:
        axis.set_xlabel("Water Saturation")
    return fig, ax


def finalise_saturation_plot(fig, ax):
    fig.tight_layout()
    plt.show()
    plt.close()


def fluid_mix_data(f: manFL.FluidMix, patch_parameter: float = 1.0):

    saturation = np.linspace(0, 1, 100)
    viscosity_data = np.empty(len(saturation))
    density_data = np.empty(len(saturation))
    modulus_data = np.empty(len(saturation))
    relative_permeability_data_1 = np.empty(len(saturation))
    relative_permeability_data_2 = np.empty(len(saturation))

    for i, sw in enumerate(saturation):
        f.saturation = sw
        f.patch_q = patch_parameter
        viscosity_data[i] = f.viscosity
        density_data[i] = f.density
        modulus_data[i] = f.modulus
        relative_permeability_data_1[i], relative_permeability_data_2[i] = (
            f.effectivePermeability
        )

    return {
        "saturation": saturation,
        "viscosity": viscosity_data,
        "density": density_data,
        "modulus": modulus_data,
        "relative_permeability_1": relative_permeability_data_1,
        "relative_permeability_2": relative_permeability_data_2,
    }


def fluid_mix_plot(ax, f: manFL.FluidMix, title: str = ""):
    fluid_properties = {
        "Density": "Density (g/cm^3)",
        "Modulus": "Bulk Modulus (GPa)",
        "Viscosity": "Viscosity (Pa.s)",
        "Relative Permeability": "Relative Permeability",
    }
    patch_parameter = {
        "uniform": 1.0,
        "intermediate": np.sqrt(f.min_q),
        "patchy": f.min_q,
    }
    properties = ["density", "modulus", "viscosity"]
    data_uniform = fluid_mix_data(f, patch_parameter["uniform"])
    data_intermediate = fluid_mix_data(f, patch_parameter["intermediate"])
    data_patchy = fluid_mix_data(f, patch_parameter["patchy"])
    for i, axis in enumerate(ax):
        for j, data in enumerate([data_uniform, data_intermediate, data_patchy]):
            axis.plot(data["saturation"], data[properties[i]])

        axis.set_ylabel(properties[i])


# def fluid_mix_plot(ax, f: manFL.FluidMix, title: str = ""):
#     fluid_properties = {
#         "Density": "Density (g/cm^3)",
#         "Modulus": "Bulk Modulus (GPa)",
#         "Viscosity": "Viscosity (Pa.s)",
#     }
#     saturation = np.linspace(0, 1, 100)
#     patch_parameter = {
#         "uniform": 1.0,
#         "intermediate": np.sqrt(f.min_q),
#         "patchy": f.min_q,
#     }

#     def fluid(sw: float, patch_parameter: float = 1.0):
#         f.saturation = sw
#         f.patch_q = patch_parameter
#         return f.density, f.modulus, f.viscosity

#     for axis in ax:
#         axis.set_xlabel("Water Saturation")
#     mod_visc_dens = np.empty((3, len(saturation)))
#     for sw in saturation:
#         plots["uniform"] = np.array([fluid(sw, 1)[0] for sw in saturation])
#         plots["intermediate"] = np.array([fluid(sw, 0.5)[0] for sw in saturation])
#         plots["patchy"] = np.array([fluid(sw, 0.2)[0] for sw in saturation])

#     plots = {}
#     plots["uniform"] = np.array([fluid(sw, 1)[0] for sw in saturation])
#     plots["intermediate"] = np.array([fluid(sw, 0.5)[0] for sw in saturation])
#     plots["patchy"] = np.array([fluid(sw, 0.2)[0] for sw in saturation])
#     ax.set_ylabel("Fluid Modulus")
#     ax.set_title(title)
#     for key, value in plots.items():
#         ax.plot(saturation, value, label=key)
#     ax.legend()

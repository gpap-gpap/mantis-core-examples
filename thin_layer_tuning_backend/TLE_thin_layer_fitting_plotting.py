from __future__ import annotations
from typing import Any
import mantis_core.rock_physics as manRP
import mantis_core.rock_physics.fluid as manFL
import mantis_core.utilities as manUT
import mantis_core.interface as manINT
from mantis_core.rock_physics import density as manDEN
from mantis_wave_modelling import model_building as manMB
from mantis_wave_modelling import wave_modelling as manWM
from mantis_wave_modelling import wavelet as manWLET
import scipy.fft as fft
import scipy.optimize as opt
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = [8.0, 6.0]
mpl.rcParams["figure.dpi"] = 80
mpl.rcParams["savefig.dpi"] = 300
plt.style.use("plotting.mantis_plotting")

plt.rc("lines", linewidth=2)

import scipy.fft as fft


class WaveSimulator:
    def __init__(
        self,
        *,
        n_samples: int = 2**10,
        dt: float = 0.001,
    ):
        self._wavelet = None
        self.wave_modeller = manWM.WaveModeller(nSamples=n_samples, dt=dt)

    def create_wavelet(self, **kwargs):
        try:
            self._wavelet = manWLET.wavelet(identifier="ricker", **kwargs)
        except:
            raise ValueError(f"wavelet parameters not found in {kwargs}")

    @property
    def wavelet(self):
        if self._wavelet is None:
            self.create_wavelet(central_frequency=39)
        return self._wavelet

    def convolve_freq_domain(self, *, spectrum: np.ndarray):
        normalization = (
            2.0 * np.pi / self.wave_modeller.nSamples / self.wave_modeller.dt
        )
        result = normalization * fft.irfft(spectrum, norm="forward")
        return result


class GetField:
    conditions = {
        "sleipner": {"temperature": 35, "pressure": 9},
        "endurance": {"temperature": 57, "pressure": 15},
        "smeaheia": {"temperature": 40, "pressure": 9},
        "nl": {"temperature": 60, "pressure": 26},
    }
    in_situ_values = {
        "sleipner": {
            # 1: {"Vp": 2.15, "Vs": 1.3, "Rho": 2.15},
            1: {"Vp": 2.15, "Vs": 0.86, "Rho": 2.15},
            2: {"Vp": 2.05, "Vs": 0.7, "Rho": 2.05},
            3: {"Vp": 2.05, "Vs": 0.7, "Rho": 2.05},
        },
        "endurance": {
            1: {"Vp": 3.82, "Vs": 2.06, "Rho": 2.56},
            2: {"Vp": 3.89, "Vs": 2.18, "Rho": 2.31},
            3: {"Vp": 3.89, "Vs": 2.18, "Rho": 2.31},
        },
        "smeaheia": {
            1: {"Vp": 2.6, "Vs": 1.15, "Rho": 2.38},
            2: {"Vp": 2.72, "Vs": 1.3, "Rho": 2.23},
            3: {"Vp": 2.72, "Vs": 1.3, "Rho": 2.23},
        },
        "nl": {
            1: {"Vp": 3.9, "Vs": 1.9, "Rho": 2.55},
            2: {"Vp": 3.6, "Vs": 1.9, "Rho": 2.25},
            3: {"Vp": 3.6, "Vs": 1.9, "Rho": 2.25},
        },
        # "nl": {
        #     1: {"Vp": 3.9, "Vs": 1.9, "Rho": 2.55},
        #     2: {"Vp": 3.4, "Vs": 1.0, "Rho": 2.5},
        #     3: {"Vp": 3.8, "Vs": 1.0, "Rho": 2.5},
        # },
    }
    model_values = {
        "sleipner": {
            1: {"parameters": {}},
            2: {
                "parameters": {
                    # "Km": 37.5,
                    "Km": 37.5,
                    "Phi": 0.34,
                    "Brie_e": 4.5,
                    "permeability": 1e-12,
                },
            },
            3: {"parameters": {}},
        },
        "endurance": {
            1: {"parameters": {}},
            2: {
                "parameters": {
                    "Km": 37.5,
                    "Phi": 0.22,
                    "Brie_e": 4.5,
                    "permeability": 1e-13,
                },
            },
            3: {"parameters": {}},
        },
        "smeaheia": {
            1: {
                "parameters": {},
            },
            2: {
                "parameters": {
                    "Km": 32,
                    "Phi": 0.28,
                    "Brie_e": 4.5,
                    "permeability": 1e-12,
                },
            },
            3: {
                "parameters": {},
            },
        },
        "nl": {
            1: {"parameters": {}},
            2: {
                "parameters": {
                    "Km": 37.5,
                    "Phi": 0.22,
                    "Brie_e": 4.5,
                    "permeability": 1e-12,
                },
            },
            3: {"parameters": {}},
        },
    }

    def __init__(self, field: str):
        self.field = None
        self.fluid_mix = None
        if field in self.conditions.keys():
            self.field = field
            self.fluid_mix = self._create_water_co2_mix2()
            q = (
                GetField.model_values[self.field][2]["parameters"]["Brie_e"]
                ** (3 / 2.0)
                * self.fluid_mix.fluid2.modulus
                / self.fluid_mix.fluid1.modulus
            )
            self.fluid_mix.patch_q = q
        else:
            raise ValueError(f"field must be one of {self.conditions.keys()}")

    def _create_water_co2_mix2(self):
        if self.field == "sleipner":
            modulus = 0.07
            density = 0.7
        elif self.field == "smeaheia":
            modulus = 0.0458
            density = 0.63
        elif self.field == "endurance":
            modulus = 0.0767
            density = 0.65
        elif self.field == "nl":
            modulus = 0.2
            density = 0.79
        else:
            raise ValueError("conditions not found")
        fluid1 = manFL.Fluid.from_presets(
            name="Water", **GetField.conditions[self.field]
        )
        fluid_temp = manFL.Fluid.from_presets(
            name="CarbonDioxide", **GetField.conditions[self.field]
        )
        fluid2 = manFL.Fluid(
            name="CarbonDioxide",
            modulus=modulus,
            viscosity=fluid_temp.viscosity,
            density=density,
        )
        return manFL.FluidMix(fluid1=fluid1, fluid2=fluid2)


class ThinLayerModels:
    model_names = ["white", "crm1d", "gassmann"]
    initial_parameters = {"bubble_radius": 1e-6, "patch_size": 1e-6}

    def __init__(self, field: str):
        self.field = GetField(field=field)
        self.wave_simulator = WaveSimulator()
        self._models = {}
        self.data_vs_saturation = None
        self.saturation_axis = np.linspace(0, 1, 100)
        self.top_model = manRP.models(
            identifier="generic", **GetField.in_situ_values[self.field.field][1]
        )
        self.bot_model = manRP.models(
            identifier="generic", **GetField.in_situ_values[self.field.field][3]
        )
        self.topRho = GetField.in_situ_values[self.field.field][1]["Rho"]
        self.bottomRho = GetField.in_situ_values[self.field.field][3]["Rho"]

    def _create_models(self):
        in_situ = GetField.in_situ_values[self.field.field][2]
        fluid = self.field.fluid_mix
        parameters = GetField.model_values[self.field.field][2]["parameters"]
        parameters["fluid"] = fluid
        parameters = dict(parameters, **in_situ)
        gassmann = manRP.models(identifier="gassmann", **parameters)
        parameters = dict(parameters, **ThinLayerModels.initial_parameters)
        parameters["permeability"] = GetField.model_values[self.field.field][2][
            "parameters"
        ]["permeability"]
        white = manRP.models(identifier="white", **parameters)
        caspari = manRP.models(identifier="crm1d", **parameters)
        self._models["gassmann"] = gassmann
        self._models["white"] = white
        self._models["crm1d"] = caspari

    @property
    def models(self):
        if self._models == {}:
            self._create_models()

        return self._models

    def reim(self, a: np.cdouble | float) -> np.cdouble:
        """
        Ensures the root with negative complex part is chosen in the complex square root.

        Args:
            z (complex): any complex number z  = a + i b

        Returns:
            complex: the number with positive real part and negative imaginary part
        """

        return np.abs(np.real(a)) - 1j * np.abs(np.imag(a))

    def saturation_data(
        self,
        *,
        bubble_radius: float = 1e-6,
        patch_size: float = 1e-6,
        log_freq: None | float = None,
    ):
        if log_freq is None:
            log_freq = np.log10(self.wave_simulator.wavelet.central_frequency)
        self.models["white"].bubble_radius = bubble_radius
        self.models["crm1d"].patch_size = patch_size
        self.data_vs_saturation = {}
        self.data_vs_saturation["saturation"] = self.saturation_axis
        gman = np.empty(shape=(len(self.saturation_axis)), dtype=complex)
        cas = gman.copy()
        whi = gman.copy()
        for i, s in enumerate(self.saturation_axis):
            self.field.fluid_mix.saturation = s
            gman[i] = self.models["gassmann"].Cij()[0, 0]
            whi[i] = self.models["white"].Cij(omega=log_freq)[0, 0]
            cas[i] = self.models["crm1d"].Cij(omega=log_freq)[0, 0]
        self.data_vs_saturation["gassmann"] = gman
        self.data_vs_saturation["white"] = whi
        self.data_vs_saturation["crm1d"] = cas

    # print(rho1())
    def normal_incidence_traces(self, *, saturation: float = 0.8):
        self.field.fluid_mix.saturation = saturation

        def ni_RC(Z1, Z2):
            return (Z2 - Z1) / (Z2 + Z1)

        Z0 = np.sqrt(self.top_model.Cij()[0, 0] * self.topRho)
        Z2 = np.sqrt(self.bot_model.Cij()[0, 0] * self.bottomRho)
        r0 = ni_RC(Z0, Z2)
        wlet = self.wave_simulator.wavelet.frequency_domain(
            freq=self.wave_simulator.wave_modeller.freq_axis
        )

        half_spectrum = self.wave_simulator.wave_modeller.nSamples // 2
        self.waveforms = {}
        self.waveforms["time"] = (
            self.wave_simulator.wave_modeller.time_axis[:-1]
            - half_spectrum * self.wave_simulator.wave_modeller.dt
        )
        # ------ before injection -----
        p_i = self.wave_simulator.convolve_freq_domain(spectrum=r0 * wlet)
        self.waveforms["pre-injection"] = p_i

        # ------ co2 rho-----
        rho1 = manDEN.Density(
            fluid=self.field.fluid_mix,
            Phi=self.models["gassmann"].Phi,
            # Phi=model_inputs[play][2]["parameters"]["Phi"],
            Rho_d=2.65,
        )
        for key, model in self.models.items():
            temp = model
            try:
                temp.fluid.saturation = saturation
            except AttributeError:
                pass
            Z1 = self.reim(
                np.sqrt(
                    temp.Cij(
                        omega=np.log10(self.wave_simulator.wavelet.central_frequency)
                    )[0, 0]
                    * rho1()
                )
            )
            r1 = ni_RC(Z0, Z1)
            r2 = ni_RC(Z1, Z2)

            w1 = self.wave_simulator.convolve_freq_domain(spectrum=r1 * wlet)
            w2 = self.wave_simulator.convolve_freq_domain(spectrum=r2 * wlet)
            self.waveforms[key] = {
                "top-co2": np.roll(w1, half_spectrum),
                "co2-water": np.roll(w2, half_spectrum),
            }

    def freq_data(self, *, saturation: float = 0.8):
        self.data_vs_freq = {}

        self.data_vs_freq["log_frequency"] = np.linspace(0, 4, 100)
        for key, model in self.models.items():
            temp = model
            try:
                temp.fluid.saturation = saturation
            except AttributeError:
                pass
            self.data_vs_freq[key] = np.array(
                [temp.Cij(omega=o)[0, 0] for o in self.data_vs_freq["log_frequency"]]
            )

    def tuning_data(self, *, saturation: float = 0.8):
        self.tuning = {}
        self.tuning["samples"] = np.arange(1, 23)
        half_spectrum = self.wave_simulator.wave_modeller.nSamples // 2
        if self.waveforms is None:
            self.normal_incidence_traces(saturation=saturation)

        for key, model in self.models.items():
            amplitude = np.empty(len(self.tuning["samples"]), dtype=float)
            for j, temp_thick in enumerate(self.tuning["samples"]):
                td = self.waveforms[key]["top-co2"] + np.roll(
                    self.waveforms[key]["co2-water"], temp_thick
                )
                max_pos = np.argmax(td[half_spectrum:])
                min_pos = np.argmin(td)
                amplitude[j] = td[max_pos] - td[min_pos]
                self.tuning[key] = amplitude

    def slsAVO(self, *, saturation: float = 0.8, Q: float = 10, thickness: int = 10):
        max_angle = 45
        d_angle = 3
        temporal_thickness = thickness
        angles = np.pi * np.arange(0, max_angle, d_angle) / 180.0
        wlet = self.wave_simulator.wavelet.frequency_domain(
            freq=self.wave_simulator.wave_modeller.freq_axis
        )
        slownesses = np.array(
            [
                manUT.incidence_angle_to_slowness(
                    incidence_angle=a,
                    vp=np.sqrt(self.top_model.Cij()[0, 0] / self.topRho),
                )
                for a in angles
            ]
        )
        # print(slownesses)
        self.field.fluid_mix.saturation = saturation
        rho1 = manDEN.Density(
            fluid=self.field.fluid_mix,
            Phi=self.models["gassmann"].Phi,
            Rho_d=2.65,
        )
        vp = np.sqrt(self.models["gassmann"].Cij()[0, 0] / rho1())
        self.vp = vp
        vs = np.sqrt(self.models["gassmann"].Cij()[3, 3] / rho1())
        self.vs = vs
        rho = rho1()
        self.rho = rho
        sls_model = manRP.models(
            identifier="sls",
            Vp=vp,
            Vs=vs,
            Rho=rho,
            Q_sls=Q,
            Log_omega_ref=np.log10(self.wave_simulator.wavelet.central_frequency),
        )
        self.sls = {}
        half_spectrum = self.wave_simulator.wave_modeller.nSamples // 2

        elasticR0 = manINT.ReflectionTransmissionMatrix(
            cijUp=self.top_model.Cij(),
            cijDown=self.models["gassmann"].Cij(),
            rhoUp=self.topRho,
            rhoDown=rho1(),
        )

        elasticR1 = manINT.ReflectionTransmissionMatrix(
            cijUp=self.models["gassmann"].Cij(),
            cijDown=self.bot_model.Cij(),
            rhoUp=rho1(),
            rhoDown=self.bottomRho,
        )
        self.sls["freq"] = np.linspace(0, 4, 100)
        self.sls["spectrum"] = np.array(
            [sls_model.Cij(omega=f) for f in self.sls["freq"]]
        )
        self.sls["angles"] = angles

        self.sls["elastic"] = {}
        self.sls["viscoelastic"] = {}
        for i, sl in enumerate(slownesses):
            w0 = self.wave_simulator.convolve_freq_domain(
                spectrum=elasticR0(horizontal_slowness=[sl, 0])[0][0, 0] * wlet
            )

            w1 = self.wave_simulator.convolve_freq_domain(
                spectrum=elasticR1(horizontal_slowness=[sl, 0])[0][0, 0] * wlet
            )

            self.sls["elastic"][i] = np.roll(w0, half_spectrum) + np.roll(
                w1, half_spectrum + temporal_thickness
            )

        log_f = np.log10(self.wave_simulator.wave_modeller.freq_axis)
        r0f = np.empty(
            (len(self.wave_simulator.wave_modeller.freq_axis), len(slownesses)),
            dtype=complex,
        )
        r1f = r0f.copy()
        favo = np.empty(
            (len(slownesses), self.wave_simulator.wave_modeller.nSamples), dtype=float
        )
        cij_freq = np.empty((len(log_f), 6, 6), dtype=complex)
        for i, o in enumerate(log_f):
            cij = sls_model.Cij(omega=o)
            cij_freq[i] = cij
            r0 = manINT.ReflectionTransmissionMatrix(
                cijUp=self.top_model.Cij(),
                cijDown=cij,
                rhoUp=self.topRho,
                rhoDown=rho1(),
            )

            r1 = manINT.ReflectionTransmissionMatrix(
                cijUp=cij,
                cijDown=self.bot_model.Cij(),
                rhoUp=rho1(),
                rhoDown=self.bottomRho,
            )
            for j, sl in enumerate(slownesses):
                r0f[i, j] = r0(horizontal_slowness=[sl, 0])[0][0, 0]
                r1f[i, j] = r1(horizontal_slowness=[sl, 0])[0][0, 0]
        self.r0f = r0f
        self.r1f = r1f
        for i, sl in enumerate(slownesses):
            w0 = self.wave_simulator.convolve_freq_domain(spectrum=r0f[:, i] * wlet)

            w1 = self.wave_simulator.convolve_freq_domain(spectrum=r1f[:, i] * wlet)

            self.sls["viscoelastic"][i] = np.roll(w0, half_spectrum) + np.roll(
                w1, half_spectrum + temporal_thickness
            )


class Plots:
    def __init__(self):
        pass

    def reim(self, a: np.cdouble | float) -> np.cdouble:
        """
        Ensures the root with negative complex part is chosen in the complex square root.

        Args:
            z (complex): any complex number z  = a + i b

        Returns:
            complex: the number with positive real part and negative imaginary part
        """

        return np.abs(np.real(a)) - 1j * np.abs(np.imag(a))

    def plot_saturation_data(self, data: dict, ax: plt.Axes, labels: dict):
        if "saturation" in data.keys():
            for key, value in data.items():
                if key != "saturation":
                    ax.plot(data["saturation"], value.real, label=labels[key])
            ax.set_xlabel("Saturation")
            ax.set_ylabel("P-wave modulus")
            ax.legend()
        else:
            raise ValueError("saturation data not found")

    def plot_tuning_data(self, data: dict, ax: plt.Axes, labels: dict):
        if "samples" in data.keys():
            for key, value in data.items():
                if key != "samples":
                    ax.plot(data["samples"], value, label=labels[key])
            ax.set_xlabel("Temporal thickness(ms)")
            ax.set_ylabel("Amplitude")
            ax.legend()
        else:
            raise ValueError("tuning data not found")

    def plot_freq_data(self, data: dict, ax: plt.Axes, labels: dict):
        if "log_frequency" in data.keys():
            for key, value in data.items():
                if key != "log_frequency":
                    ax.plot(data["log_frequency"], value.real, label=labels[key])
            ax.set_xlabel("log(Frequency)")
            ax.set_ylabel("P-wave modulus")
            ax.legend()
        else:
            raise ValueError("frequency data not found")

    def plot_att_data(self, data: dict, ax: plt.Axes, labels: dict):
        if "log_frequency" in data.keys():
            for key, value in data.items():
                v = self.reim(value)
                if key != "log_frequency":
                    ax.plot(
                        data["log_frequency"],
                        -v.imag / v.real,
                        label=labels[key],
                    )
            ax.set_xlabel("log(Frequency)")
            ax.set_ylabel("P-wave attenuation")
            ax.legend()
        else:
            raise ValueError("frequency data not found")


class SLSfits:
    def __init__(self):
        pass

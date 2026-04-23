# Advanced Computational Architectures for Photovoltaic Simulation: A Comprehensive Technical Treatise on the pvlib-python Ecosystem

The evolution of solar energy simulation has moved decisively from fragmented, proprietary algorithms toward open-source, community-validated computational frameworks. At the epicenter of this transition is pvlib-python, a robust software library designed to provide a standardized, transparent, and interoperable suite of functions for modeling the performance of photovoltaic systems.[^1] Originally conceived in 2013 as a translation of the Sandia National Laboratories' PVLIB for MATLAB, the Python ecosystem has matured into a comprehensive toolbox supported by global research institutions and industry leaders.[^2] The core mission of the library is to provide benchmark implementations of the scientific models that underpin solar resource assessment and system energy yield prediction, ensuring that the modeling chain remains rigorous, reproducible, and adaptable to emerging technologies.[^1]

The architectural design of pvlib-python is predicated on two primary modeling philosophies: procedural and object-oriented. This flexibility allows researchers to either invoke specific physical functions for granular analysis or employ high-level class structures to automate complex simulation workflows.[^3] For engineers tasked with replacing legacy scripts—often referred to as a `physics_model.py`—pvlib offers a validated path that eliminates the need to reinvent fundamental models while providing a quantifiable boost in simulation accuracy through its diverse library of peer-reviewed algorithms.[^5]

---

## Architectural Paradigm: Functional Blocks and Class Orchestration

The fundamental organization of pvlib-python is built upon specialized modules that handle distinct stages of the photovoltaic modeling chain. These modules range from data ingestion and astronomical calculations to atmospheric modeling, electrical performance, and loss assessment.[^3]

### Procedural versus Object-Oriented Interfaces

The procedural interface consists of standalone functions that perform specific tasks, such as calculating the solar zenith angle or estimating the diffuse fraction of irradiance.[^3] This approach is ideal for developers who wish to integrate specific pvlib components into existing software architectures without adopting the entire class hierarchy. Conversely, the object-oriented interface provides a set of classes—`Location`, `PVSystem`, `Array`, and `ModelChain`—that encapsulate system state and automate the progression of data through the modeling steps.[^3]

The `Location` class is used to represent the geographical context of a simulation, storing attributes such as latitude, longitude, altitude, and time zone.[^3] It provides convenient methods for solar position and clear-sky irradiance that are localized to these parameters.[^12] The `PVSystem` class represents the hardware configuration, including the arrangement of modules (series/parallel), inverter specifications, and mounting geometry (fixed or tracking).[^11] Within this structure, the `Array` class further refines the modeling by allowing for multiple orientations or different module types within a single system.[^11]

The `ModelChain` serves as the high-level orchestrator, linking the `Location` and `PVSystem` objects. Its primary function is to manage the flow of extrinsic data, such as weather time series, through the intrinsic models defined in the system attributes.[^10] This orchestration is crucial for maintaining a standardized modeling workflow and ensuring that the interdependencies between physical phenomena—such as the impact of temperature on electrical efficiency—are correctly handled.[^4]

| Computational Entity | Module/Class | Primary Functionality |
|---|---|---|
| Data Ingestion | `pvlib.iotools` | Parsing TMY, EPW, and real-time API data[^5] |
| Solar Position | `pvlib.solarposition` | Determining zenith and azimuth angles[^12] |
| Irradiance Models | `pvlib.irradiance` | Decomposition, transposition, and reflection[^18] |
| Clear Sky | `pvlib.clearsky` | Estimating irradiance under cloudless conditions[^13] |
| Thermal Dynamics | `pvlib.temperature` | Cell and module temperature modeling[^25] |
| Electrical Output | `pvlib.pvsystem` | DC and AC power calculations[^11] |
| System Orchestration | `pvlib.modelchain` | End-to-end simulation management[^10] |

---

## Data Acquisition and Metadata Standardization

The first step in any photovoltaic simulation is the acquisition and preparation of weather and location data. The `pvlib.iotools` module provides an extensive suite of functions for reading standardized file formats and accessing remote databases, which ensures that input data is consistent and correctly localized.[^2]

### Standardized File Formats and Remote Data Access

Modern solar modeling relies on Typical Meteorological Year (TMY) datasets or real-time satellite-derived irradiance. The library supports the ingestion of TMY2 and TMY3 formats, as well as EnergyPlus Weather (EPW) files, which are commonly used in building performance simulations.[^5] Furthermore, specialized functions exist for reading data from international networks like the Baseline Surface Radiation Network (BSRN) and the Solar Radiation Monitoring Laboratory (SRML).[^7]

Remote data access is facilitated through wrappers for APIs such as NREL's National Solar Radiation Database (NSRDB), PVGIS, and CAMS McClear.[^2] These tools allow researchers to retrieve long-term historical data or high-resolution forecasts without manual file management.[^30] The integration of these tools into the simulation environment is a critical advantage over custom physics models, as it provides built-in handling for unit conversions and metadata extraction, reducing the likelihood of manual data entry errors.[^5]

### The Critical Role of Time Zone Localization

A significant insight gleaned from professional modeling experience is the frequency of errors related to time zone localization. Solar position algorithms rely on timestamps being localized to the correct time zone to synchronize the astronomical position with the reported irradiance data.[^5] A telltale sign of improper localization is a temporal shift between the peak of the solar elevation and the peak of clear-sky global horizontal irradiance (GHI).[^5] The `pvlib.iotools` functions typically return pandas DataFrames with localized DatetimeIndexes, which mitigates these risks by ensuring that the temporal context is preserved throughout the modeling chain.[^12]

---

## Astronomical and Geometric Foundations

The precision of a PV system simulation is fundamentally constrained by the accuracy of the solar position calculation. This calculation determines the angle at which sunlight strikes the modules and is a primary input for shading, transposition, and airmass models.[^19]

### The Solar Position Algorithm (SPA) Hierarchy

The `pvlib.solarposition` module provides several algorithms that represent different trade-offs between computational speed and physical accuracy.[^12] The primary interface is the `get_solarposition` function, which acts as a convenience wrapper for these calculators.[^12]

1. **NREL SPA (`nrel_numpy`/`nrel_c`):** The NREL Solar Position Algorithm (SPA) is the most rigorous implementation available, widely considered the gold standard for accuracy in solar position modeling.[^19] It provides results for zenith and azimuth angles with uncertainties as low as $\pm 0.0003°$. The Python implementation (`nrel_numpy`) offers high portability, while the C-wrapper (`nrel_c`) can be utilized for extreme performance requirements, although it is no more accurate than the numpy-based version.[^31]

2. **Numba Acceleration (`nrel_numba`):** For high-resolution time series simulation where speed is paramount, the `nrel_numba` method compiles the SPA code into machine language at runtime. This significantly reduces execution time while maintaining the high precision of the original algorithm.[^19]

3. **PyEphem and Ephemeris:** For general-purpose applications where millidegree precision is not required, the library provides the `pyephem` and internal `ephemeris` models.[^12] These models are faster than SPA and are often sufficient for energy yield predictions where the uncertainty in irradiance data exceeds the geometric error of the solar position.[^32]

### Atmospheric Refraction and Geometric Correction

One of the nuances in solar position modeling is the correction for atmospheric refraction. Refraction causes the Sun to appear higher in the sky than its true geometric position, an effect that is most pronounced near the horizon.[^19] The library distinguishes between the "true" zenith (geometric) and the "apparent" zenith (refraction-corrected).[^19]

The accuracy of the refraction correction depends on atmospheric conditions, specifically pressure and temperature. Research indicates that while using instantaneous weather data for these corrections is technically justified, the difference between using time-dependent data and annual average data is minimal.[^32] For the sake of consistency in PV simulations, annual average values are often preferred as they prevent artificial noise in the irradiance modeling chain.[^32]

| Parameter | Impact on Angular Accuracy | Recommended Source |
|---|---|---|
| Latitude/Longitude | Primary geometric drivers | Site survey or GPS[^19] |
| Time/Date | Temporal synchronization | Localized DatetimeIndex[^5] |
| Pressure | Influences refraction correction | Measured or altitude-derived[^19] |
| Temperature | Minor impact on refraction | Measured or annual average[^23] |

### Orbital Mechanics and Extraterrestrial Irradiance

A second-order insight regarding geometric accuracy concerns the Sun-Earth distance. The Earth's orbit is elliptical, with an eccentricity of approximately 0.017. This variance causes the extraterrestrial irradiance ($G_{ext}$) to fluctuate by about $\pm 3.4\%$ over the course of a year.[^34] While simple circular models exist, pvlib provides sophisticated functions like `get_extra_radiation` and `nrel_earthsun_distance` that account for the ecliptic orbit.[^12] Neglecting these orbital mechanics can introduce systematic errors in clear-sky modeling and sensor calibration.[^34]

---

## Clear-Sky Irradiance and Atmospheric Characterization

Clear-sky models predict the solar radiation that would reach the surface if there were no clouds. These models serve as a vital benchmark for detecting system anomalies and for modeling the potential energy yield of a site.[^23]

### Comparison of Theoretical Frameworks

The `pvlib.clearsky` module implements several models that vary in their input requirements and physical complexity.[^23] Simple models rely primarily on solar geometry, while complex models incorporate atmospheric transmittances for aerosols, water vapor, and ozone.[^23]

1. **Ineichen and Perez Model:** This is one of the most flexible models in the library, providing GHI, DNI, and DHI estimates.[^23] It is driven by the Linke Turbidity ($T_L$), which describes the attenuation of the atmosphere relative to a dry and clean one. The function `lookup_linke_turbidity` allows users to retrieve monthly climatological values for any location globally, ensuring that the model is accurately parameterized even in the absence of site-specific measurements.[^23]

2. **Bird Simple Model:** Developed by Bird and Hulstrom, this is a broadband model that calculates the transmittances of individual atmospheric constituents.[^24] In various validation studies, specifically those conducted in high-insolation regions like South Africa, the Bird model has been shown to outperform simpler models across diverse climatic regions.[^37]

3. **Simplified Solis Model:** A computationally efficient version of the Solis radiative transfer model, this function is particularly useful when accurate data for precipitable water ($P_{wat}$) and aerosols are available.[^23] Research has shown that using local meteorological data with the Solis model can provide a low-cost yet accurate tool for detecting soiling on irradiance sensors.[^36]

4. **Haurwitz and Berger-Duffie:** These are simple models that require only the solar zenith angle.[^23] While they are less accurate than the complex models, they provide a quick estimate of GHI when atmospheric parameters are unknown.[^23]

| Model | Outputs | Primary Driver | Complexity |
|---|---|---|---|
| Ineichen | GHI, DNI, DHI | Linke Turbidity ($T_L$) | Medium[^23] |
| Bird | GHI, DNI, DHI | Airmass, Aerosols | High[^23] |
| Solis | GHI, DNI, DHI | $P_{wat}$, Elevation | High[^23] |
| Haurwitz | GHI only | Zenith Angle | Low[^23] |

### The Role of Turbidity and Airmass

The performance of any clear-sky model is heavily dependent on the atmospheric inputs. The `pvlib.atmosphere` and `pvlib.clearsky` modules provide tools to estimate these values when they are not measured. For example, the `kasten96_lt` function calculates Linke Turbidity from precipitable water and aerosol optical depth (AOD).[^24] Similarly, `bird_hulstrom80_aod_bb` provides a mechanism to approximate broadband AOD from discrete measurements.[^24]

Relative airmass—the path length of sunlight through the atmosphere relative to the zenith—is another critical parameter. The library provides several airmass models, including Kasten-Young (1989) and Kasten-Czeplak (1980), which account for the curvature of the Earth and the vertical profile of the atmosphere.[^7]

---

## Irradiance Decomposition and Transposition Mechanics

The core of a PV performance model is the transformation of global horizontal irradiance (GHI) into plane-of-array (POA) irradiance. This process typically involves two stages: decomposition (separating GHI into direct and diffuse components) and transposition (calculating the irradiance on the tilted surface).[^18]

### Decomposition: The GHI to DNI/DHI Transition

In many weather datasets, only GHI is recorded. To accurately model a tilted PV array, the direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI) must be estimated. The `pvlib.irradiance` module contains several validated decomposition models.[^18]

- **DISC Model:** An empirical model that relates the clearness index ($k_t$) to DNI based on airmass and time of year.[^18]
- **Erbs Model:** A widely used model for estimating the diffuse fraction ($DHI/GHI$) based on the clearness index.[^18]
- **DIRINT and DIRINDEX:** These models represent more sophisticated attempts to capture the impact of temporal variability and local climate on the diffuse fraction.[^18]

### Transposition: Calculating POA Irradiance

The total irradiance incident on a tilted module is the sum of three components: the beam (direct) component, the sky diffuse component, and the ground-reflected component.[^22]

$$I_{tot} = I_{beam} + I_{sky\_diffuse} + I_{ground\_reflected}$$

The `get_total_irradiance` function is the primary entry point for this calculation. While the beam and ground components are largely geometric, the sky diffuse component is notoriously difficult to model because the sky is not uniformly bright.[^18]

1. **Isotropic Model:** Assumes the sky diffuse irradiance is uniform across the entire sky dome. While simple, it typically underestimates the energy on clear days by neglecting circumsolar brightening.[^22]

2. **Hay-Davies and Reindl Models:** These models introduce a circumsolar brightening factor, which recognizes that the sky is brighter near the solar disk. They are considered an improvement over the isotropic model for most general-purpose simulations.[^22]

3. **Perez Model:** The most technically rigorous transposition model, it divides the sky into eight sectors to account for circumsolar brightening and horizon brightening.[^21] It is highly sensitive to the circumsolar and horizon coefficients and requires extraterrestrial radiation as an input.[^22]

4. **King and Klucher:** Alternative models that offer different empirical fits for specific climatic conditions, such as overcast or hazy skies.[^18]

Validation studies have shown that the choice of transposition model can significantly affect the predicted yield, particularly for high-latitude or highly tilted systems.[^6] The Perez model, when properly parameterized, is generally regarded as the most accurate for these scenarios.[^22]

---

## Thermal Dynamics and Operating Temperature Modeling

Photovoltaic module efficiency decreases as temperature increases. Therefore, the ability to predict cell temperature from ambient conditions (irradiance, air temperature, and wind speed) is a critical component of a performance model.[^25]

### Steady-State Heat Balance Models

The `pvlib.temperature` module provides several empirical models that assume the module is in thermal equilibrium with its environment.[^26]

- **Faiman Model:** A common model used in international standards (IEC 61853), it uses a heat loss factor that is linearly dependent on wind speed.[^25]
- **PVsyst Temperature Model:** Similar to the Faiman model but adds a term for the electrical efficiency of the module. This accounts for the fact that energy converted to electricity is not available to heat the module.[^41]
- **SAPM Temperature Model:** Developed by Sandia, this model uses exponential fits for wind speed and is categorized by racking type (e.g., open rack vs. close-mounted).[^14]

### Recent Advances in Parameter Translation

A key emerging theme in the thermal modeling community is the equivalence of these different models. Research has demonstrated that the Faiman, PVsyst, SAPM, and SAM NOCT models have identical or very similar characteristics and differ primarily in their parameterization.[^47] This insight has led to the development of translation equations that allow users to convert parameters from one model to another (e.g., converting SAPM coefficients to PVsyst U-values), enhancing the interoperability of simulations across different software platforms.[^47]

### Specialized Modeling: Floating PV (FPV)

Floating photovoltaic systems represent a unique thermal modeling challenge due to the cooling effect of the water body and the different wind patterns over water. The documentation highlights that for FPV systems, the mounting structure and water footprint significantly influence the heat loss coefficients.[^41]

| FPV System Configuration | Uc (W/m²K) | Uv (W/m²K⋅s/m) | Reference Source |
|---|---|---|---|
| Open structure, 2-axis tracking | 24.4 – 57.0 | 0.0 – 6.5 | Netherlands[^49] |
| Closed structure, Large footprint | 25.2 – 37.0 | 0.0 – 3.7 | Netherlands[^49] |
| Closed structure, Singapore | 18.9 – 34.8 | 0.8 – 8.9 | Singapore[^49] |
| In contact with water | 71.0 | 0.0 | Norway[^49] |

For systems in direct thermal contact with the water, the cooling effectiveness is largely dictated by water temperature rather than air temperature, leading to significantly higher heat loss coefficients ($U_c$).[^41]

---

## DC Performance and Electrical Conversion

The conversion of absorbed irradiance and temperature into DC power is the final step in the modeling of the PV array itself. pvlib supports several electrical models, ranging from simple empirical equations to complex physical representations.[^11]

### The Single-Diode Model (SDM) Architecture

The single-diode model is the most popular means of simulating the electrical output of a PV module. It represents the cell as a current source in parallel with a diode and a shunt resistance, all in series with a series resistance.[^50] The `pvlib.pvsystem` module provides functions for solving the single-diode equation (SDE) using different analytical and numerical methods.[^50]

1. **Lambert W-Function:** An analytical solution that is computationally efficient but requires specific mathematical libraries.[^50]
2. **Bishop's Algorithm:** A robust numerical solver that is less sensitive to initial conditions and can handle reverse-bias modeling for shading analysis.[^50]

Different versions of the single-diode model use different auxiliary equations to determine the parameters (light current $I_L$, saturation current $I_o$, etc.) at operating conditions:

- **CEC Model:** Uses a large database of module parameters (SAM).[^21]
- **De Soto Model:** A classic five-parameter model that is well-documented and widely used for silicon modules.[^6]
- **PVsyst Model:** Includes specialized terms for thin-film modules and recombination currents.[^44]

### Empirical Alternatives: SAPM and PVWatts

For users who do not have the five or six parameters required for a single-diode model, empirical models provide a reliable alternative.[^27]

- **Sandia Array Performance Model (SAPM):** A highly accurate empirical model that uses performance coefficients derived from outdoor testing. It accounts for spectral effects and non-linear efficiency changes.[^14]
- **PVWatts DC Model:** The simplest DC model in the library, it uses only a nameplate power ($P_{dc0}$) and a power temperature coefficient ($\gamma_{pdc}$).[^27] Recent updates include an optional $k$ factor to improve accuracy at low irradiance levels.[^52]

| DC Model | Input Requirements | Primary Advantage |
|---|---|---|
| Single Diode | 5-6 parameters ($R_s, R_{sh}$, etc.) | Physical accuracy, I-V curve generation[^50] |
| SAPM | Coefficients ($I_{sc0}, V_{oc0}$, etc.) | High empirical accuracy, spectral sensitive[^14] |
| PVWatts | $P_{dc0}, \gamma_{pdc}$ | Robustness, minimal data requirement[^52] |

---

## High-Level Orchestration: The ModelChain Class

The `ModelChain` class is the primary tool for replacing a legacy `physics_model.py`. It provides a standardized, high-level interface that automates the transition from weather data to system output.[^10]

### Automated Model Inference

One of the most significant advantages of using the `ModelChain` is its ability to "infer" the appropriate models based on the attributes of the associated `PVSystem`.[^4] If a user provides a `PVSystem` with parameters retrieved from the CEC database, the `ModelChain` will automatically select the CEC single-diode model, the corresponding temperature model, and an appropriate inverter model.[^4]

This inference logic extends to several categories:

- **DC Model:** Inferred from module parameters (`sapm`, `desoto`, `cec`, `pvsyst`, or `pvwatts`).[^4]
- **AC Model:** Inferred from inverter parameters (`sandia`, `adr`, or `pvwatts`).[^16]
- **Temperature Model:** Inferred from system configuration and module type.[^16]
- **Optical Losses:** Inferred from the presence of AOI or spectral parameters.[^16]

### Execution Workflows and Result Management

The `ModelChain` can be executed using various `run_model` methods depending on the data available. The standard `run_model(weather)` takes a GHI/DNI/DHI time series, while `run_model_from_poa(data)` allows the user to bypass the decomposition and transposition steps if they already possess POA measurements.[^10]

The results of the simulation are stored in a `ModelChainResult` object, which provides a single point of access for all intermediate and final modeling outputs.[^10]

| Attribute | Description | Units |
|---|---|---|
| `solar_position` | Zenith, azimuth, and elevation | Degrees[^9] |
| `total_irrad` | POA global, direct, and diffuse | $W/m^2$[^16] |
| `cell_temperature` | Operating temperature of the cells | $°C$[^16] |
| `effective_irrad` | Irradiance after spectral and AOI losses | $W/m^2$[^16] |
| `dc` | DC power output of the array | Watts[^16] |
| `ac` | AC power output after the inverter | Watts[^14] |

---

## Validation, Benchmarking, and Comparative Accuracy

For practitioners transitioning from proprietary software to pvlib, the question of accuracy is paramount. Extensive validation against industry standards like PVSyst and SAM has confirmed the credibility of pvlib's simulations.[^6]

### Comparison with Industry Standard Tools

Studies comparing pvlib to PVSyst for sites in Southern Africa found that the two platforms showed high coherence across all modeling steps.[^6] In one evaluation, the difference in plane-of-array (POA) irradiance calculations was less than 1% when using identical transposition models.[^6] While minor differences occur in the sunrise/sunset definitions—with PVSyst often reporting sunrise about an hour earlier—these do not significantly impact the annual energy yield as the irradiance at those times is near zero.[^6]

Further research comparing simulation models (PVLib, PVSyst, SAM, Helioscope) against measured data at the Utrecht Photovoltaic Outdoor Test facility concluded that pvlib's combinations of SAPM and physical models were among the most accurate on both a macro-level (annual yield) and a micro-level (data point aggregation).[^53] Notably, PVSyst and SAM were found to generally underestimate electricity yields compared to the measured benchmarks, whereas pvlib allowed for finer tuning of the model chain to match local conditions.[^53]

### Sensitivity and Error Propagation

Accuracy in PV modeling is often a function of data resolution and model complexity.[^35] For instance, complex clear-sky models (Bird, Solis) consistently outperform simple ones, provided that high-quality atmospheric data (AOD, $P_{wat}$) is available.[^35] However, in regions where these data points are not measured, using simpler models or satellite-derived turbidity can lead to systematic biases.[^36]

The library's modularity allows users to mitigate these errors by selecting models that match their data quality. If high-resolution wind data is missing, a simpler temperature model may be more robust than a complex one like Prilliman, which requires regular sampling and no data gaps.[^5]

---

## Strategic Implementation: Replacing physics_model.py

To satisfy the objective of replacing a custom physics model with pvlib entirely, a researcher should follow a structured integration plan that leverages the library's built-in automation features.

### Step 1: Mapping Existing Functions to pvlib Modules

A typical custom physics model includes functions for airmass, solar angle, and power coefficients. These should be mapped to the corresponding pvlib modules. Replacing a manual secant-based airmass calculation with `pvlib.atmosphere.get_relative_airmass` ensures that the model correctly handles the atmosphere near the horizon, where simple trigonometric models often fail.[^31]

### Step 2: Defining the Physical System

Instead of hard-coding module efficiencies and temperature coefficients, the user should utilize the `PVSystem` class and the `retrieve_sam` function to load validated parameters for specific modules and inverters.[^11] This allows for the use of the Single-Diode or SAPM models, which capture the non-linear behavior of the system much better than simple efficiency-multiplier models.[^14]

### Step 3: Integrating the ModelChain

The `ModelChain` should be adopted as the primary simulation engine. By defining the system as a `ModelChain(system, location)`, the user benefits from automatic spectral and angle-of-incidence corrections that are often overlooked in home-grown models.[^10] This "physics-complete" approach provides a significant boost in simulation credibility for professional reporting.

### Step 4: Verification through Visualization

Finally, the transition should be validated by comparing the results of the new pvlib workflow against the legacy model. The powerful plotting capabilities of the pandas and matplotlib libraries, which are natively integrated with pvlib, allow for the visual identification of inconsistencies, such as non-zero irradiance when the sun is below the horizon.[^5]

In summary, pvlib-python represents a transformative tool for solar energy researchers. By providing an open, reliable, and interoperable implementation of PV system models, it empowers developers to build sophisticated modeling processes without the burden of reinventing established physics.[^1] Whether modeling a standard residential array or a complex floating bifacial system, the library's depth and technical rigor ensure that the results are indistinguishable from—and often superior to—the output of industry-standard proprietary tools.[^6]

---

## Works Cited

[^1]: pvlib python — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/
[^2]: GitHub - pvlib/pvlib-python: A set of documented functions for simulating the performance of photovoltaic energy systems., accessed April 22, 2026, https://github.com/pvlib/pvlib-python
[^3]: Package Overview — pvlib-python 0.3.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.3.0/package_overview.html
[^4]: ModelChain — pvlib-python 0.5.1+0.g72a7144.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.5.1/modelchain.html
[^5]: Frequently Asked Questions — pvlib python 0.9.4 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.4/user_guide/faq.html
[^6]: (PDF) PHOTOVOLTAIC SYSTEM MODELLING USING PVLIB-PYTHON - ResearchGate, accessed April 22, 2026, https://www.researchgate.net/publication/313249264_PHOTOVOLTAIC_SYSTEM_MODELLING_USING_PVLIB-PYTHON
[^7]: Overview: module code — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/_modules/index.html
[^8]: Index — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/genindex.html
[^9]: Source code for pvlib.modelchain, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.3.2/_modules/pvlib/modelchain.html
[^10]: ModelChain — pvlib python 0.11.2 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.11.2/user_guide/modelchain.html
[^11]: PVSystem — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/pvsystem.html
[^12]: pvlib.solarposition.get_solarposition, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.solarposition.get_solarposition.html
[^13]: PV Modeling — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/index.html
[^14]: pvlib.pvsystem.PVSystem — pvlib-python 0.4.2+0.g04b7a82.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.pvsystem.PVSystem.html
[^15]: PVSystem. pvsyst_celltemp - pvlib python - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.pvsystem.PVSystem.pvsyst_celltemp.html
[^16]: ModelChain — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/modelchain.html
[^17]: pvlib.modelchain.ModelChain — pvlib python 0.15.0 documentation - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.html
[^18]: Irradiance — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/index.html
[^19]: pvlib.solarposition.get_solarposition — pvlib-python 0.4.2+0.g04b7a82.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.solarposition.get_solarposition.html
[^20]: Solar Position - Solar Resource Assessment in Python, accessed April 22, 2026, https://assessingsolar.org/notebooks/solar_position.html
[^21]: pvlib.location.Location.get_solarposition — pvlib python 0.11.1 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.11.1/reference/generated/pvlib.location.Location.get_solarposition.html
[^22]: pvlib.irradiance.get_total_irradiance — pvlib python 0.15.0 documentation - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html
[^23]: Clear sky — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/clearsky.html
[^24]: Clear sky — pvlib python 0.11.1 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.11.1/user_guide/clearsky.html
[^25]: pvlib.temperature.faiman — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html
[^26]: PV temperature models — pvlib python 0.15.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html
[^27]: PVSystem. - pvwatts_dc - pvlib python - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.PVSystem.pvwatts_dc.html
[^28]: pvlib.pvsystem.pvwatts_dc — pvlib python 0.9.0+0.g518cc35.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.pvsystem.pvwatts_dc.html
[^29]: pvlib.solarposition.get_solarposition — pvlib python 0.10.4 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.10.4/reference/generated/pvlib.solarposition.get_solarposition.html
[^30]: A Comparison of PV Power Forecasts Using PVLib-Python - Zenodo, accessed April 22, 2026, https://zenodo.org/records/1400857/files/Hol17%20A%20Comparison%20of%20PV%20Power%20Forecasts%20Using%20PVLib-Python.pdf?download=1
[^31]: pvlib-python/pvlib/solarposition.py at main - GitHub, accessed April 22, 2026, https://github.com/pvlib/pvlib-python/blob/master/pvlib/solarposition.py
[^32]: Inconsistent default settings for _prep_inputs_solar_pos in prepare_inputs and prepare_inputs_from_poa · Issue #1065 · pvlib/pvlib-python - GitHub, accessed April 22, 2026, https://github.com/pvlib/pvlib-python/issues/1065
[^33]: Modules — pvlib-python 0.3.0 documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.3.0/modules.html
[^34]: Which solar geometry should be assumed by pvlib? - Stack Overflow, accessed April 22, 2026, https://stackoverflow.com/questions/73305806/which-solar-geometry-should-be-assumed-by-pvlib
[^35]: VALIDATING CLEAR-SKY IRRADIANCE MODELS IN FIVE SOUTH AFRICAN LOCATIONS, accessed April 22, 2026, https://www.sasec.org.za/papers2019/6.pdf
[^36]: Evaluating the Accuracy of Various Irradiance Models in Detecting Soiling of Irradiance Sensors: Preprint - Publications, accessed April 22, 2026, https://docs.nrel.gov/docs/fy20osti/75156.pdf
[^37]: The Performance Assessment of Six Global Horizontal Irradiance Clear Sky Models in Six Climatological Regions in South Africa - MDPI, accessed April 22, 2026, https://www.mdpi.com/1996-1073/14/9/2583
[^38]: Comparison of eight clear sky broadband models against 16 independent data banks | Request PDF - ResearchGate, accessed April 22, 2026, https://www.researchgate.net/publication/222694906_Comparison_of_eight_clear_sky_broadband_models_against_16_independent_data_banks
[^39]: pvlib.irradiance.get_total_irradiance — pvlib python 0.9.0+0.g518cc35.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.irradiance.get_total_irradiance.html
[^40]: pvlib.pvsystem.pvwatts_dc — pvlib python 0.11.0 documentation - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.11.0/reference/generated/pvlib.pvsystem.pvwatts_dc.html
[^41]: Temperature modeling for floating PV - pvlib python - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/latest/gallery/floating-pv/plot_floating_pv_cell_temperature.html
[^42]: PVsyst Cell Temperature Model, accessed April 22, 2026, https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/pvsyst-cell-temperature-model/
[^43]: Source code for pvlib.temperature, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.9.0/_modules/pvlib/temperature.html
[^44]: pvlib.temperature.pvsyst_cell - pvlib python - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.pvsyst_cell.html
[^45]: pvlib.temperature.pvsyst_cell — pvlib-python 0.7.0+0.ge2a8f31.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.7.0/generated/pvlib.temperature.pvsyst_cell.html
[^47]: PV Module Operating Temperature Model Equivalence and Parameter Translation, accessed April 22, 2026, https://www.researchgate.net/publication/378856749_PV_Module_Operating_Temperature_Model_Equivalence_and_Parameter_Translation
[^48]: Temperature model parameter translation · Issue #1442 · pvlib/pvlib-python - GitHub, accessed April 22, 2026, https://github.com/pvlib/pvlib-python/issues/1442
[^49]: Temperature modeling for floating PV - pvlib python - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.13.1/gallery/floating-pv/plot_floating_pv_cell_temperature.html
[^50]: pvlib-python/docs/sphinx/source/user_guide/modeling_topics/singlediode.rst at main - GitHub, accessed April 22, 2026, https://github.com/pvlib/pvlib-python/blob/main/docs/sphinx/source/user_guide/modeling_topics/singlediode.rst
[^51]: PVSystem — pvlib-python 0.6.0+0.g15c00ef.dirty documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/v0.6.0/pvsystem.html
[^52]: pvlib.pvsystem.pvwatts_dc — pvlib python 0.15.1.dev24+g44cb79fbc documentation, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.pvsystem.pvwatts_dc.html
[^53]: A comparative study of PV simulation and machine learning models on a macrolevel and microlevel - Utrecht University Student Theses Repository Home, accessed April 22, 2026, https://studenttheses.uu.nl/bitstream/handle/20.500.12932/35556/Master%27s%20Thesis%20A_ZUIKER.pdf?sequence=1&isAllowed=y
[^54]: pvlib.temperature.prilliman — pvlib python 0.15.0 documentation - Read the Docs, accessed April 22, 2026, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.prilliman.html

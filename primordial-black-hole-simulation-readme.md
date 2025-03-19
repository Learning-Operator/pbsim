# 🌌 Primordial Black Hole Simulation

![GitHub stars](https://img.shields.io/github/stars/yourusername/primordial-black-hole-simulation?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/primordial-black-hole-simulation?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/primordial-black-hole-simulation?style=social)

> A computational framework for investigating the formation and survival probability of primordial black holes in the early universe.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Theoretical Framework](#theoretical-framework)
- [Numerical Methods](#numerical-methods)
- [N-body Simulation Architecture](#n-body-simulation-architecture)
- [Current Development Status](#current-development-status)
- [Technical Implementation](#technical-implementation)
- [Usage Instructions](#usage-instructions)
- [Results and Analysis](#results-and-analysis)
- [References](#references)
- [License](#license)
- [Contact](#contact)

## 🔭 Project Overview

This repository contains computational simulations investigating the existence probability of primordial black holes (PBHs) in our universe. Originally presented at the Rorke Adams Science Fair, this project explores the formation conditions and survival probability of PBHs from the early universe to present day through detailed cosmological modeling and N-body simulations.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Simulation+Visualization" alt="Simulation Visualization" width="80%">
</p>

## 🌠 Theoretical Framework

### Cosmological Assumptions

- **Flat Universe Model**: Justified by Planck Collaboration data showing scalar curvature parameters near zero:
  - Least constrained parameter: -0.058 +0.046/-0.051
  - Most constrained parameter: 0.0005 +/-0.0038/-0.0040

- **Parameter Sources**: TT, TE, EE+lowE+lensing+BAO constrained parameters from Planck 281 Collaboration

- **Cosmological Constant**: Omitted from Friedmann equations due to negligible contribution at early universe scales

### Modified Friedmann Equation Implementation

- First Friedmann equation modified to accommodate flat universe assumption
- Density parameters rescaled appropriately through simulation timeframe
- Initial approach used constant critical density (resulted in errors)
- Corrected approach scales density directly and recalculates parameters afterward

## 🧮 Numerical Methods

### Integration Technique

- **Runge-Kutta 4th Order Integration**: Implemented without adaptive timestep
- Future improvement: Implement adaptive timestep to improve accuracy and reduce computational artifacts

### Simulation Validation

Generated and validated three key relationships:
1. Scale factor variation over time
2. Rate of change of scale factor over time
3. Hubble parameter evolution across simulation timeframe

## 🔬 N-body Simulation Architecture

### Toy Simulation Development

- Basic particles with positional tracking
- Gravitational interaction implementation
- Co-moving coordinate system integration

### Co-moving Coordinate System

- Implemented relationship: peculiar distance = scale factor × distance
- Equations of motion rewritten to accommodate co-moving framework
- Currently in testing/validation phase

### Distribution Models

| Model Type | Description | Status |
|------------|-------------|--------|
| **Uniform Distribution** | Fixed particle count with uniform spatial distribution | Implemented |
| **Random Gaussian Field** | Models early universe fluctuations with Gaussian probability density functions | Refining |
| **Power Spectrum Distribution** | Fluctuation amplitude varies with wave number (k) | Implemented |
| **Scale-Invariant Spectrum** | Constant fluctuation amplitude as function of k | Implemented |

### Particle Classification

- **Radiation Particles**: Implemented with "effective mass" properties
- **Mass Particles**: Standard gravitational mass properties
- Both particle types interact according to modified equations of motion

## 📊 Current Development Status

### Completed Features ✅

- Friedmann equation integration with cosmological parameters
- Basic N-body simulation with gravitational interactions
- Co-moving coordinate system implementation
- Multiple distribution models for initial conditions

### In Progress 🔄

- Refining random Gaussian field implementation
- Testing co-moving coordinate system accuracy
- Optimizing integration performance

### Planned Features 📝

- Analytical calculations for collapse conditions
- Perturbation tracking system
- Improved distribution accuracy based on cosmological models
- Quantitative analysis of PBH formation probability

## 💻 Technical Implementation

```python
# Example of Runge-Kutta 4th order integration implementation
def rk4_step(y, t, dt, derivatives):
    k1 = derivatives(y, t)
    k2 = derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(y + dt * k3, t + dt)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
```

- **Integration Method**: Runge-Kutta 4th order without adaptive timestep
- **Particle Interaction**: Direct N-body calculation with gravitational forces
- **Coordinate System**: Co-moving coordinates with proper scaling relationships
- **Distribution Generation**: Gaussian field implementation with power spectrum considerations

## 📝 Usage Instructions

### Installation

```bash
git clone https://github.com/yourusername/primordial-black-hole-simulation.git
cd primordial-black-hole-simulation
pip install -r requirements.txt
```

### Running the Simulation

```bash
python simulate.py --particles 1000 --distribution gaussian --timespan 10e9 --timestep 1e5
```

### Output Analysis

```bash
python analyze_results.py --input simulation_output.dat --plot-scale-factor --plot-hubble
```

## 📈 Results and Analysis

The simulation outputs currently include:

<p align="center">
  <img src="https://via.placeholder.com/400x300?text=Scale+Factor+Evolution" alt="Scale Factor Evolution" width="45%">
  &nbsp;&nbsp;
  <img src="https://via.placeholder.com/400x300?text=Hubble+Parameter" alt="Hubble Parameter" width="45%">
</p>

1. Scale factor evolution visualizations
2. Rate of change of scale factor over time
3. Hubble parameter evolution
4. Particle distribution snapshots at various simulation epochs

Future analysis will include:
- Statistical analysis of density fluctuations
- PBH formation rate calculations
- Mass spectrum predictions
- Survival probability to present day

## 📚 References

- Planck Collaboration. (Year). Planck 281: Cosmological Parameters.
- [Additional references for cosmological models]
- [References for numerical methods]
- [References for PBH formation theory]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/primordial-black-hole-simulation](https://github.com/yourusername/primordial-black-hole-simulation)

---

<p align="center">
  <sub>Created with ❤️ for the advancement of cosmological science</sub>
</p>

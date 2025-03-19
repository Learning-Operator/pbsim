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
- [Contact](#contact)

## 🔭 Project Overview

This repository contains computational simulations investigating the existence probability of primordial black holes (PBHs) in our universe. This project explores the formation conditions and survival probability of PBHs from the early universe to present day through a cosmological modeling and N-body simulations.

<p align="center">

  ![n_body_fixed_5](https://github.com/user-attachments/assets/98e1f186-3c22-456f-8c7a-26920ba469af)


## 🌠 Theoretical Framework

### Cosmological Assumptions

- **Flat Universe Model**: Justified by Planck Collaboration data showing scalar curvature parameters near zero:
  - Least constrained parameter: -0.058 +0.046/-0.051
  - Most constrained parameter: 0.0005 +/-0.0038/-0.0040

- **Parameter Sources**: TT, TE, EE+lowE+lensing+BAO constrained parameters from Planck 2018 Collaboration

- **Cosmological Constant**: Omitted from Friedmann equations due to negligible contribution at early universe scales

### Modified Friedmann Equation Implementation

- First Friedmann equation modified to accommodate flat universe assumption
- Density parameters rescaled appropriately through simulation timeframe
- Initial approach used constant critical density (resulted in errors), however, modified approach to scale density, and then recalculate density parameter.

## 🧮 Numerical Methods

### Integration Technique

- **Runge-Kutta 4th Order Integration**: Implemented without adaptive timestep
- **Future improvement**: Implement adaptive timestep to improve accuracy and reduce computational artifacts

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

### Distributions Modeled

| Distribution | Description | Status |
|------------|-------------|--------|
| **Uniform Distribution** | Fixed particle count with uniform spatial distribution | Refining |
| **Random Gaussian Field** | Models early universe fluctuations with Gaussian probability density functions | Refining |
| **Power Spectrum Distribution** | Fluctuation amplitude varies with wave number (k) | Refining |
| **Scale-Invariant Spectrum** | Constant fluctuation amplitude as function of k | Refining |

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

- Refining random Gaussian field implementation and other distribution
- Testing co-moving coordinate system accuracy
- Optimizing integration performance through adaptive timesteps

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
git clone https://github.com/Learning-Operator/PBH_PROJ.git
```


## 📈 Results and Analysis


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

Escrivà, A., & Romano, A. E. (2021). Effects of the shape of curvature peaks on the size of primordial black holes. 
Journal of Cosmology and Astroparticle Physics, 2021(05), 066. https://doi.org/10.1088/1475-7516/2021/05/066
Musco, I., De Luca, V., Franciolini, G., & Riotto, A. (2021). Threshold for primordial black holes. II. A simple 
analytic prescription. Physical Review D, 103(6), 063538. https://doi.org/10.1103/PhysRevD.103.063538
Carr, B., & Kuhnel, F. (2021). Primordial black holes as dark matter candidates. SciPost Physics Lecture Notes,
48. https://doi.org/10.48550/arXiv.2110.02821

Musco, I., Miller, J. C., & Rezzolla, L. (2005). Computations of primordial black-hole formation. Classical and 
Quantum Gravity, 22(7), 14051424. https://doi.org/10.1088/0264-9381/22/7/013
The picture of our universe: A view from modern cosmology - D. Reid et al. (n.d.). Ned.ipac.caltech.edu. 
https://ned.ipac.caltech.edu/level5/Sept02/Reid/Reid5_2.html

Escrivà, A. (2022, January 21). PBH formation from spherically symmetric hydrodynamical perturbations: A 
Review. MDPI. https://www.mdpi.com/2218-1997/8/2/66

Planck Collaboration, Aghanim, N., Akrami, Y., Ashdown, M., Aumont, J., Baccigalupi, C., Ballardini, M., 
Banday, A. J., Barreiro, R. B., Bartolo, N., Basak, S., Battye, R., Benabed, K., Bernard, J.-P. ., Bersanelli, M., 
Bielewicz, P., Bock, J. J., Bond, J. R., Borrill, J., & Bouchet, F. R. (2019). Planck 2018 results. VI. Cosmological 
parameters. ArXiv:1807.06209 [Astro-Ph]. https://arxiv.org/abs/1807.06209

International Centre for Theoretical Sciences. (2020, September 21). Essential Cosmological Perturbation Theory 
by David Wands. YouTube. https://www.youtube.com/watch?v=WrGv9nkTaRQ

Musco, I., De Luca, V., Franciolini, G., & Riotto, A. (2021). Threshold for primordial black holes. II. A simple analytic 
prescription. Physical Review D, 103(6), 063538. https://doi.org/10.1103/PhysRevD.103.063538

## 📬 Contact

Tanay Vajhala - [Tanayvajhala@gmail.com](mailto:Tanayvajhala@gmail.com)

---

<p align="center">
  <sub>Created with ❤️ for the advancement of cosmological science</sub>
</p>

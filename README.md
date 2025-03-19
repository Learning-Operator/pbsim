Primordial Black Hole Simulation
Project Overview
This repository contains computational simulations investigating the existence probability of primordial black holes (PBHs) in our universe. Originally presented at the Rorke Adams Science Fair, this project explores the formation conditions and survival probability of PBHs from the early universe to present day through detailed cosmological modeling and N-body simulations.
Theoretical Framework
Cosmological Assumptions

Flat Universe Model: Justified by Planck Collaboration data showing scalar curvature parameters near zero:

Least constrained parameter: -0.058 +0.046/-0.051
Most constrained parameter: 0.0005 +/-0.0038/-0.0040


Parameter Sources: TT, TE, EE+lowE+lensing+BAO constrained parameters from Planck 281 Collaboration
Cosmological Constant: Omitted from Friedmann equations due to negligible contribution at early universe scales

Modified Friedmann Equation Implementation

First Friedmann equation modified to accommodate flat universe assumption
Density parameters rescaled appropriately through simulation timeframe
Initial approach used constant critical density (resulted in errors)
Corrected approach scales density directly and recalculates parameters afterward

Numerical Methods
Integration Technique

Runge-Kutta 4th Order Integration: Implemented without adaptive timestep
Future improvement: Implement adaptive timestep to improve accuracy and reduce computational artifacts

Simulation Validation
Generated and validated three key relationships:

Scale factor variation over time
Rate of change of scale factor over time
Hubble parameter evolution across simulation timeframe

N-body Simulation Architecture
Toy Simulation Development

Basic particles with positional tracking
Gravitational interaction implementation
Co-moving coordinate system integration

Co-moving Coordinate System

Implemented relationship: peculiar distance = scale factor × distance
Equations of motion rewritten to accommodate co-moving framework
Currently in testing/validation phase

Distribution Models

Uniform Distribution: Fixed particle count with uniform spatial distribution
Random Gaussian Field: Models early universe fluctuations

Implements Gaussian probability density functions for spatial variables
Currently being refined for improved accuracy


Power Spectrum Distribution: Fluctuation amplitude varies with wave number (k)
Scale-Invariant Spectrum: Constant fluctuation amplitude as function of k

Particle Classification

Radiation Particles: Implemented with "effective mass" properties
Mass Particles: Standard gravitational mass properties
Both particle types interact according to modified equations of motion

Current Development Status
Completed Features

Friedmann equation integration with cosmological parameters
Basic N-body simulation with gravitational interactions
Co-moving coordinate system implementation
Multiple distribution models for initial conditions

In Progress

Refining random Gaussian field implementation
Testing co-moving coordinate system accuracy
Optimizing integration performance

Planned Features

Analytical calculations for collapse conditions
Perturbation tracking system
Improved distribution accuracy based on cosmological models
Quantitative analysis of PBH formation probability

Technical Implementation Details

Integration method: Runge-Kutta 4th order without adaptive timestep
Particle interaction: Direct N-body calculation with gravitational forces
Coordinate system: Co-moving coordinates with proper scaling relationships
Distribution generation: Gaussian field implementation with power spectrum considerations

---
title: 'ParallelKDE.jl: A Package for Highly Parallel Kernel Density Estimation'
tags:
  - Julia
  - kernel density estimation
  - parallel algorithms
  - Python
authors:
  - name:
      given: Christian
      surname: Sustay Martinez
    orcid: 0009-0006-1282-4850
    affiliation: 1
  - name:
      given: Patrick K.
      surname: Quoika
    orcid:
    affiliation: 1
  - name:
      given: Martin
      surname: Zacharias
    orcid:
    affiliation: 1
affiliations:
  - index: 1
    name: Center for Functional Protein Assemblies (CPA), Technical University of Munich, Germany
date: 09.10.2025
bibliography: paper.bib
---

# Summary

Kernel density estimation (KDE) is a valuable tool in exploratory analysis, simulation, and probabilistic modeling across the sciences. However, its susceptibility to the curse of dimensionality can make routine KDE prohibitively slow on large, high-dimensional datasets.
`ParallelKDE.jl` provides a fast, open-source KDE toolkit in Julia [@bezanson_julia_2017] that provides extensible infrastructure to support a variety of methods with serial, multi-threaded and GPU implementations. The package includes (i) a highly parallel implementation of established KDE rules of thumb as well as (ii) an implementation of GradePro, a new algorithm described in [**cite?**], engineered here for serial, threaded, and CUDA backends. A lightweight Python wrapper, `ParallelKDEpy` exposes the same functionality to Python workflows. Our gaol is to make robust density estimation practical for large sample sizes of modeate dimensionality through careful parallelization of, mainly, grid-based KDE estimators. For background on KDE and bandwidth selection, see @silverman_density_1986, @scott_multivariate_1992, @wand_kernel_1995 and @jones_brief_1996.

# Statement of need

Many researchers rely on Python implementations (e.g., SciPy's `gaussian_kde` [@virtanen_scipy_2020; @scipy_developers_scipystatsgaussian_kde_2025] with automatic bandwidth via rules-of-thumb; scikit-learn's `KernelDensity` [@pedregosa_scikit-learn_2011; @scikit-learn_developers_kerneldensity_2025] which also supports rules-of-thumb; statsmodels' [@seabold_statsmodels_2010] univariate FFT KDE and multivariate estimators with cross-validated bandwidths [@statsmodels_developers_statsmodelsnonparametrickernel_densitykdemultivariate_2024]; KDEpy's [@odland_tommyodkdepy_2018] FFT KDE with rules-of-thumb and plug-in rules [@sheather_reliable_1991]). While mature and widely used, these tools are CPU-centric and can become slow as data grow in size and dimension, especially when many evaluations are needed in downstream tasks.

`ParallelKDE.jl` addresses this gap with:

1. a unified device interface so that the estimations can be executed on serial, threads (via Julia's `Threads.@threads`), or GPU (e.g. via `CUDA.jl`) without changing user code;
2. parallelized calculation of FFT-based KDE with rules-of-thumb and GradePro;
3. a reproducible benchmark comparing against SciPy, scikit-learn, statsmodels and KDEpy under matched estimation tasks.

Our benchmarks (\autoref{fig:benchmark_samples} and \autoref{fig:benchmark_grids}) demonstrate that moving denisty estimation to parallel execution or parallel hardware produces marked acceleration compared with standard CPU-based libraries, in particular for high-resolution grids or large datasets. We reported accuracy and fine-tuning features of GradePro in [**cite?**]. Additional accuracy and other feature comparison of this and a diverse set of methods is in progress. Here we focus on the engineering choices enabling fast implementations.

![Benchmark of common KDE packages and their estimators along with methods in ParallelKDE with respect to sample size. Relative mean times are reported where a value of 1.0 indicates that the estimator was the fastest for a sample size. The estimations were performed for 100 to 100000 samples in 1D with a grid of 500 points; and for 1000 to 1000000 samples in 2D with a grid of 100 points per dimension.\label{fig:benchmark_samples}](./benchmark_samples.pdf)

![Benchmark of common KDE packages and their estimators along with methods in ParallelKDE with respect to grid size. Relative mean times are reported where a value of 1.0 indicates that the estimator was the fastest for a grid size. The estimations were performed for 100 to 2500 grid points in 1D with 10000 samples; and for 33 to 300 grid points per dimension with 100000 samples in 2D.\label{fig:benchmark_grids}](./benchmark_grid.pdf)

# Functionality

**Design and architecture.** `ParallelKDE.jl` is organized around a minimal device interface that isolates parallelism from algorithmic logic. We ship three backends:

- Serial CPU for portability and determinism;
- Threaded CPU using Julia's built-in multithreading;
- CUDA GPU via `CUDA.jl` [@besard_effective_2019; @besard_rapid_2019].

**Implemented methods.** The package includes (a) parallel rules-of-thumb KDE, namely Scott and Silverman, for rapid baselines and (b) GradePro, engineered to perform independent grid-point-wise density estimation using FFT and reducing necessary memory by reusing it throughout the routines without damaging performance. The serial version serves as a reference; the threaded version parallelizes routines applied over the bootstrap re-samples required by the method; the CUDA routines provide parallelization at each estimated grid-point. Each implemented method naturally exposes all its parameters up to the user and can place convenient defaults when reasonable. \autoref{fig:flowchart} shows a flowchart of the implemented algorithm.

![Flowchart of the parallelizable point-wise density estimation algorithm. \label{fig:flowchart}](./parallelkde_flowchart.pdf)

**Reproducibility and testing.** We include timing typed benchmarks to illustrate the speed gains obtained from parallelizable KDE algorithms.

**Python wrapper.** The companion package `ParallelKDEpy` provides, with the help of `PythonCall.jl` [@rowley_pythoncalljl_2022], Python bindings to `ParallelKDE.jl`, offering identical estimators and device selection for seamless integration in Python-based workflows.

# Acknowledgements

funding...

# References

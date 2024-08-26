# DexRT: A GPU-Accelerated Non-LTE Radiative Transfer Code Based on Radiance Cascades


|   |   |   |   |
|---|---|---|---|
| __Maintainer__ | Chris Osborne | __Institution__ | University of Glasgow  |
| __License__ | ![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue) | | |


Description
-----------

DexRT is a modern, in-development, non-LTE predominantly solar radiative transfer code intended for studying the formation of atomic spectral lines in multidimensional atmospheres with complex structure, such as solar prominences.
Currently 2D (with additional third invariant axis) models are supported.
The calculation of the radiation field through the atmosphere is performed by a novel technique known as [Radiance Cascades](https://github.com/Raikiri/RadianceCascadesPaper), avoiding the ray effects that plague short characteristics solvers on problems that have complex mixed optically-thin and thick structure.

Atomic data is provided in the [CRTAF](https://github.com/Goobley/CommonRTAtomicFormat) format, a nascent potential-standard for interoperation between non-LTE codes.

![COCOPLOT of J across Ly beta in a prominence model](Images/cocoplot_lyb_j.png)

📖 Documentation
----------------

Documentation is thin on the ground currently, please see the paper (when released), doxygen strings in some files, and the tests.
Please ask if things are unclear.

⬇ Installation
--------------

Docker container specifications are available in `.devcontainer`. The most tested option is nvhpc, but the code, via [YAKL](https://github.com/mrnorman/YAKL) should be adaptable to all major GPU vendors.
Feel free to get in touch if trying to run this on different hardware.

🤝 Contributing
---------------

We would love for you to get involved.
Please report any bugs encountered on the GitHub issue tracker.

If you are interested in adding a feature, please reach out, even if to say "I'm not sure if this is a good idea"!
We can then determine the best way to go about the feature, and whether it's already on the core roadmap.

We require all contributors to abide by the [code of conduct](CODE_OF_CONDUCT.md).


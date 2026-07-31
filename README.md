# DA-HA-SAC: Dual-Actuator History-Augmented Reinforcement Learning

This repository accompanies the manuscript **“Dual-Actuator History-Augmented Reinforcement Learning for Separation Suppression and Loss Reduction in a Compressor Cascade.”** The study develops a dual-actuator history-augmented soft actor–critic (**DA-HA-SAC**) framework for closed-loop control of suction-surface separation in a two-dimensional NACA 65 compressor cascade.

Two independently controlled actuators are located at 15% and 60% chord and can operate in either blowing or suction mode. The controller uses history-stacked pressure measurements, downstream total-pressure-loss information, and previous actions to account for partial observability and delayed aerodynamic responses. A delay-aware, stage-dependent reward combines reverse-flow suppression, total-pressure-loss reduction, actuation efficiency, and command smoothness. Parallel CFD environments are used to train a single policy at incidence angles of 8°, 11°, and 12°.

The learned policy suppresses suction-surface separation and narrows the downstream high-loss region at all three incidence angles. The peak pitchwise total-pressure-loss coefficient is reduced by more than 30%. Comparisons with mass-flow-matched constant-jet actuation show that the RL policy achieves higher actuation efficiency at all three incidences and improves both loss reduction and efficiency under the more strongly separated conditions. Flow-field analysis further reveals a two-stage mechanism: incidence-dependent transient restructuring through coordinated suction and blowing, followed by predominantly blowing-based quasi-steady maintenance.

## Flow-control visualisation

[![Flow-control animation](videos/11i.gif)](https://github.com/willian-yz/DA-HA-SAC/)

The animated preview links to the full automatically playing video hosted with GitHub Pages.

## Citation

Please cite the associated manuscript when using the methods, data, or visualisations provided in this repository.

## Contact

**Yizhou Luo**  
School of Energy Science and Engineering, Harbin Institute of Technology

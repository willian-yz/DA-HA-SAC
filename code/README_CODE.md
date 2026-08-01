# Public DA-HA-SAC Code Structure

This directory contains a reduced public representation of the Fluent–SAC
software architecture.

## Files

- `framework.py`: reusable Fluent/Gymnasium/SAC framework.
- `case.py`: case-specific configuration and main entry point.

## Requirements

- ansys-fluent-core
- gymnasium
- numpy
- stable-baselines3


## Included workflow

The public code retains the main sequence:

```text
SAC action
   ↓
physical actuator commands
   ↓
Fluent boundary-condition update
   ↓
unsteady CFD advancement
   ↓
pressure observation and aerodynamic metrics
   ↓
stage-dependent reward
   ↓
history-augmented next state
   ↓
SAC update
```

It also retains:

- dual-actuator control;
- Gymnasium environment structure;
- history stacking;
- parallel Fluent environments;
- SAC training and deterministic testing;
- basic history recording.

## Omitted before publication

The following implementations are intentionally represented by private hooks:

- pressure-probe extraction and normalisation;
- reverse-flow-region calculation;
- total-pressure-loss and actuation-efficiency calculations;
- response-delay alignment;
- stage-dependent reward equations and weights;
- case-specific baselines;
- exact SAC hyperparameters;
- trained models and replay buffers.

The repository therefore documents the implementation architecture without
providing a complete reproduction package.

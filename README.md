# cosmos-spatio
Affective-Physics: Spatio-Temporal Interaction States in Human–Robot Collaboration

Robots should not guess how humans feel. They should predict how humans are likely to react.

Executive Summary:

  Traditional human–robot interaction systems rely on single-frame emotion detection, an approach that breaks down in dynamic, real-world environments. This project leverages NVIDIA Cosmos to shift human–robot interaction from reactive perception to predictive interaction modeling. By analyzing spatio-temporal human motion dynamics, including acceleration patterns, postural shifts, and violations of personal space, the system infers interaction statesrather than assigning emotion labels. These states enable a robot to autonomously execute conservative and interpretable behaviors such as slowing motion, pausing tasks, or increasing interpersonal distance. The result is safer, more fluid collaboration that preserves psychological safety while maintaining operational efficiency in shared human–robot workspaces.

Problem Statement

Most emotion-aware HRI systems:

Depend on static facial expressions

Ignore body dynamics and context

React only after discomfort becomes explicit

In real environments, emotion is expressed through motion, posture, and spatial behavior over time. Static perception fails precisely when safety and trust matter most.

Core Insight

Emotion is not a label.
It is a spatio-temporal physical signal.

By modeling how humans move, hesitate, approach, or retreat over time, robots can infer interaction risk early and respond conservatively before unsafe or uncomfortable interactions occur.

Interaction State Taxonomy (What the Robot Predicts)

Instead of “happy / angry / sad”:

Comfort / Engagement

Uncertainty / Confusion

Discomfort / Stress

Escalation Risk (rapid approach, abrupt motion, crowding)

These states are actionable, interpretable, and ethically defensible.

System Overview

Inputs

Short video clips of human–robot interaction

Scene context and motion over time

Cosmos World Modeling

Temporal scene understanding

Prediction of near-future physical behavior

Contextual reasoning over motion, posture, and proximity

Inference Layer

Observable cues:

Motion acceleration and jerk

Postural rigidity or relaxation

Lean-in vs lean-away

Distance changes and speed of approach

Hesitation or start–stop behavior

Robot Policy (Simple and Transparent)

Interaction State	Robot Action
Comfort	Proceed normally
Confusion	Pause and clarify
Discomfort	Slow movement, increase distance
Escalation Risk	Stop, retreat, request human confirmation
Demonstration Scenarios

Demo 1: Healthcare Assistive Robot

Human approaches rapidly, posture tightens

Cosmos predicts increasing discomfort

Robot slows and increases personal space before contact

Demo 2: Collaborative Manufacturing

Human hesitates and repeatedly reorients

System infers confusion

Robot pauses and signals next step

Demo 3: Service Robot

Smooth approach, stable distance

Engagement inferred

Robot continues task uninterrupted

Why Cosmos Is Essential?

Static perception cannot model emotion as it unfolds.
Cosmos enables temporal world modeling, context-aware reasoning, and predictive foresight, which are necessary to interpret affective signals embedded in physical behavior.

Evaluation Plan

Technical Metrics

Agreement with human-labeled interaction states

Early-warning time vs baseline (seconds gained)

False stop / unnecessary intervention rate

Behavioral Metrics

Minimum distance maintained during discomfort

Smoothness of robot motion

Reduction in abrupt robot stops

Ethical Guardrails (Explicit and Important)

No identity recognition

No medical or psychological diagnosis

No biometric storage

Interaction states are inferred from observable motion only

Robot behaviors are conservative and reversible

Human agency is preserved at all times

Notebook Structure 

Setup and environment

Baseline static emotion detection

Cosmos temporal reasoning pipeline

Interaction state inference

Robot response policy

Evaluation and failure cases

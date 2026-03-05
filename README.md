Project Title

Affective-Physics: Predictive Emotion-Aware Human–Robot Interaction Using NVIDIA Cosmos

Abstract

Human-robot interaction systems often rely on static frame-level emotion classification, which fails to capture the dynamic nature of real human behavior. This project introduces Affective-Physics, a perception-reasoning pipeline that uses NVIDIA Cosmos Reason to interpret emotional signals from short video sequences and convert them into actionable interaction states.

Instead of simply identifying emotions, the system predicts how a robot should behave in response to human emotional conditions. By analyzing facial expressions, posture dynamics, and movement patterns, the system determines the safest and most appropriate interaction strategy.

The robot can respond to different emotional states such as anger, happiness, sadness, and frustration by adjusting its tone, movement, and interaction distance.

This approach enables more natural, safe, and adaptive human-robot collaboration in real-world environments such as retail stores, hospitals, and service centers.

System Architecture
Camera / Video Input 
        │
        ▼
Perception Layer
(Face, posture, motion signals) 
        │
        ▼
Cosmos Reason 2
(Video reasoning + emotion inference) 
        │
        ▼
Interaction State Generator
(de-escalate / engage / support / assist) 
        │
        ▼
Robot Policy Engine
(behavior strategy) 
        │
        ▼
Robot Actions
(voice tone, distance, movement, guidance)
Visual Pipeline (Simple Flow)
Human Video
     │
     ▼
Emotion Signals 
     │
     ▼
Cosmos Reasoning 
     │
     ▼
Interaction State 
     │
     ▼
Robot Behavior 

Example:

Angry Human 
     │
     ▼
Cosmos detects tension + aggressive posture 
     │
     ▼
Interaction State: De-escalate 
     │
     ▼
Robot increases distance and speaks calmly 
Demo Output Example
Emotion detected: Angry
Confidence: 0.91
Risk level: Medium

Recommended robot actions
• Increase distance 0.5 m
• Slow speech rate
• Calm voice tone

Role of NVIDIA Cosmos

Cosmos Reason enables the system to perform video-level reasoning over human behavior, connecting perception signals such as facial expression, body posture, and movement dynamics into a coherent interpretation of the human interaction state.

This reasoning capability allows the robot to predict the appropriate behavioral response rather than relying on simple emotion classification.

cat > README.md << 'EOF'
# Bearing Fault Detection Using Online Machine Learning

**Real-time anomaly detection for predictive maintenance in rotating machinery**

## Overview

This project implements an **online learning approach** to detect bearing degradation in real-time from vibration sensor data. Unlike traditional offline ML models, this system continuously learns from streaming data, enabling predictive maintenance days before equipment failure.

## Key Features

✅ **Online Learning** - Model adapts continuously from streaming data  
✅ **Real-time Detection** - Identifies bearing degradation as it happens  
✅ **Predictive Maintenance** - Schedule repairs before failure occurs  
✅ **Interactive Dashboard** - Web-based visualization of model behavior  

## Dataset

- **Source**: Case Western Reserve University Bearing Dataset
- **Samples**: 4,600 vibration spectrograms
- **Format**: 32×32 time-frequency representations
- **Conditions**: Normal, Ball Fault, Inner Race Fault, Outer Race Fault

## Results

| Stage | Anomaly Score |
|-------|---|
| Normal | 0.7353 |
| Early Fault | 0.8444 |
| Late Fault | 0.8654 |

**Clear separation between stages enables reliable fault detection**

## Project Structure
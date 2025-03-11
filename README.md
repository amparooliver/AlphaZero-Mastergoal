# 🏆 MasterGoal AlphaZero Implementation

> *This work is dedicated to the memory of Alberto Bogliaccini, the creator of MasterGoal.*

Welcome to the official repository for the **MasterGoal AlphaZero implementation**! This project adapts the powerful AlphaZero algorithm to the unique soccer-inspired game of MasterGoal, creating an AI that can learn and master the game through self-play.

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Getting Started](#-getting-started)
  - [Docker Setup (Recommended)](#docker-setup-recommended)
  - [Virtual Environment Setup](#virtual-environment-setup)
- [Training Your AI](#-training-your-ai)
- [Playing Against Your AI](#-playing-against-your-ai)
- [Acknowledgments](#-acknowledgments)

## 🎮 Introduction
MasterGoal combines strategic thinking with soccer mechanics, creating a unique board game experience. This project implements the cutting-edge AlphaZero algorithm to train an AI that can learn and master the intricacies of MasterGoal through self-play and reinforcement learning.

## 🚀 Getting Started
Choose your preferred method to set up the project:

### Docker Setup (Recommended)
Containerized for ease of use across all systems! 🐳

#### Prerequisites
* Docker installed on your system

#### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amparooliver/Mastergoal-AlphaZero.git
   cd Mastergoal-AlphaZero
   ```

2. **Build the Docker Image:**
   ```bash
   docker build -t mastergoal-alphazero:latest .
   ```

#### Running Scripts
To run any Python script in this project:
```bash
docker run mastergoal-alphazero:latest python <script_name.py>
```

**⚽ Quick Examples:**
* Start training:
  ```bash
  docker run mastergoal-alphazero:latest python main.py
  ```
* Play against the AI:
  ```bash
  docker run mastergoal-alphazero:latest python human_vs_ai.py
  ```

### Virtual Environment Setup
For those who prefer a traditional setup. 🛠️

#### Prerequisites
* Python 3.9

#### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amparooliver/Mastergoal-AlphaZero.git
   cd Mastergoal-AlphaZero
   ```

2. **Create a Virtual Environment:**
   ```bash
   # Linux
   sudo apt install python3.9-venv
   python3.9 -m venv myenv
   source myenv/bin/activate
   
   # Windows
   python3.9 -m venv myenv
   myenv\Scripts\activate
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Training Your AI
Start the AlphaZero training process with:
```bash
python main.py
```

This initiates the complete pipeline:
1. Self-play to generate training data
2. Neural network training
3. Model evaluation

For detailed configuration options, check the comments in `main.py` and `NNet.py`.

## 🎯 Playing Against Your AI
Two options to test your trained AI:

### 1. AI vs Random Player Evaluation
Measure how well your AI performs against random moves:

* **Setup:** Edit `compare_to_random.py` to point to your trained model
* **Run:**
  ```bash
  python compare_to_random.py
  ```

### 2. Human vs AI Challenge
Challenge your AI to a match:

* **Setup:** Edit `human_vs_ai.py` to point to your trained model
* **Run:**
  ```bash
  python human_vs_ai.py
  ```
* Follow the on-screen instructions to make your moves

## 👏 Acknowledgments
This project stands on the shoulders of these amazing works:

* **Dougyd92:** [AlphaZero General for DuckChess](https://github.com/dougyd92/alpha-zero-general-duckchess)
* **Surag Nair:** [AlphaZero General](https://github.com/suragnair/alpha-zero-general)

Special thanks to both developers for open-sourcing their implementations, which made this adaptation possible! 🙏

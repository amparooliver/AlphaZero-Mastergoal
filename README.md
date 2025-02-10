
# MasterGoal AlphaZero Implementation

This repository contains the implementation of **AlphaZero** for the custom game **MasterGoal**, inspired by soccer mechanics. The project uses a generalized version of AlphaZero, adapted to work with the unique rules and dynamics of MasterGoal.

This work is dedicated to the memory of Alberto Bogliaccini, the creator of MasterGoal.
## Acknowledgments

This project is built upon the incredible work of the following creators:  
- **Dougyd92**: [AlphaZero General for DuckChess](https://github.com/dougyd92/alpha-zero-general-duckchess).  
- **Surag Nair**: [AlphaZero General](https://github.com/suragnair/alpha-zero-general), which served as the base framework for this adaptation.  

Special thanks to both for open-sourcing their work, which made this adaptation possible.

---

## Installation

To get started, follow these steps:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/amparooliver/AlphaMastergoal.git
   cd AlphaMastergoal
   ```

2. **Create a Virtual Environment**  
   It’s recommended to use a virtual environment to manage dependencies. Ensure you have `Python 3.9` installed to avoid compatibility issues:  
   ```bash
   sudo apt install python3.9-venv
   python3.9 -m venv myenv
   source myenv/bin/activate  # Linux/macOS
   myenv\Scripts\activate     # Windows
   ```

3. **Install Requirements**  
   All necessary dependencies are listed in `requirements.txt`. Install them with:  
   ```bash
   pip install -r requirements.txt
   ```
---

## Training a Model

To train the AlphaZero model for MasterGoal, simply run:  
```bash
python main.py
```  
This script starts the self-play, training, and evaluation pipeline. For more details, refer to the comments in `main.py`.

---
## Playing Against a Trained Model

There are two ways to interact with a trained model:

1. **Compare Against a Random Player**  
   Use `compare_to_random.py` to evaluate how the trained model performs against random moves.  

   - **Edit the File**: Before running, ensure you specify the path to your trained model (this is indicated in the file).  
   - **Run the Script**:  
     ```bash
     python .\compare_to_random.py
     ```

2. **Play as a Human Against the AI**  
   Use `human_vs_ai.py` to play against the trained model.  

   - **Edit the File**: Specify the path to your trained model as instructed in the file.  
   - **Run the Script**:  
     ```bash
     python .\human_vs_ai.py
     ```  
   During execution, the script will provide detailed instructions for how to play.

---
## About MasterGoal (Temporary)

MasterGoal is a soccer-inspired strategy game. The goal is to implement the best possible game-playing AI using reinforcement learning principles from AlphaZero.  
- The game board is 15x11, with unique mechanics for ball handling and player movement.  
- The rules and dynamics are specifically designed for this project, creating a new and engaging challenge.




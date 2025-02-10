from flask import Flask, render_template, request, jsonify
import human_vs_ai_WEB
import threading
from queue import Queue

app = Flask(__name__)

# Thread-safe queue to handle click data
click_queue = Queue()
# Global variable to store the board state
current_board = None

def run_game():
    try:
        human_vs_ai_WEB.main(click_queue)  # Start the game logic
    except Exception as e:
        print(f"Error running the game: {e}")

@app.route("/")
def game():
    return render_template("start_game.html")

@app.route("/game")
def start_page():
    return render_template("game.html")

@app.route("/start_game", methods=["POST"])
def start_game():
    try:
        # Start the game using the human_vs_ai_web python file
        #human_vs_ai_WEB.main()
        # Start the game logic in a separate thread
        game_thread = threading.Thread(target=run_game)
        game_thread.start()

        # Redirect to the game page (update this to match your game route)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/handle_click", methods=["POST"])
def handle_click():
    """
    Endpoint to receive click data from the frontend.
    """
    data = request.get_json()
    click_queue.put((data['row'], data['col']))  # Add click data to the queue
    return jsonify({"success": True})

@app.route("/update_board", methods=["POST"])
def update_board():
    """
    Endpoint to receive the current board state from the backend game logic.
    """
    global current_board
    data = request.get_json()
    current_board = data.get("board")  # Update the global board state
    return jsonify({"success": True})

@app.route("/get_board", methods=["GET"])
def get_board():
    """
    Endpoint to fetch the current board state for the frontend.
    """
    global current_board
    if current_board is None:
        return jsonify({"success": False, "message": "No board state available"})
    return jsonify({"success": True, "board": current_board})

if __name__ == "__main__":
    app.run(debug=True)
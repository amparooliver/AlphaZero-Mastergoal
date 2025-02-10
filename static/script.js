const tablero = document.getElementById('tablero');
const turnoElement = document.getElementById('turno');
const marcadorElement = document.getElementById('marcador');

const filas = 15;
const columnas = 11;

function crearTablero() {
    tablero.innerHTML = ''; // Clear previous board layout

    for (let i = 0; i < filas; i++) {
        for (let j = 0; j < columnas; j++) {
            const casilla = document.createElement('div');
            casilla.className = 'casilla';
            casilla.id = `casilla-${i}-${j}`;

            if (esCasillaNegra(i, j)) casilla.classList.add('negro');
            else if (esArco(i, j)) casilla.classList.add('arco');
            else if (esAreaChica(i, j)) casilla.classList.add('area-chica');
            else if (esAreaGrande(i, j)) casilla.classList.add('area-grande');
            else if (esCorner(i, j)) casilla.classList.add('corner');

            // Attach click event listener
            casilla.addEventListener('click', () => handleClick(i, j));
            tablero.appendChild(casilla);
        }
    }

    // Add static yellow dots
    const yellowDots = [
        [1, 0], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 10],
        [13, 0], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 10],
    ];
    yellowDots.forEach(([fila, columna]) => addYellowDot(fila, columna));
}

function esCasillaNegra(i, j) {
    return (i === 0 && (j <= 2 || j >= 8)) || (i === 14 && (j <= 2 || j >= 8));
}

function esArco(i, j) {
    return (i === 0 || i === 14) && (j >= 3 && j <= 7);
}

function esAreaChica(i, j) {
    return ((i === 1 || i === 2 || i === 12 || i === 13) && (j >= 2 && j <= 8));
}

function esAreaGrande(i, j) {
    return ((i >= 1 && i <= 4) || (i >= 10 && i <= 13)) && (j >= 1 && j <= 9);
}

function esCorner(i, j) {
    return (i === 1 && (j === 0 || j === 10)) || (i === 13 && (j === 0 || j === 10));
}

function addYellowDot(fila, columna) {
    const casilla = document.getElementById(`casilla-${fila}-${columna}`);
    if (casilla) {
        casilla.classList.add('yellow-dot');
    }
}

function updatePieces(board) {
    // Clear existing pieces
    const allCells = document.querySelectorAll('.casilla');
    allCells.forEach(cell => {
        cell.innerHTML = ''; // Remove any player/ball elements
    });

    // Place pieces and ball dynamically
    board.forEach((row, i) => {
        row.forEach((cell, j) => {
            const casilla = document.getElementById(`casilla-${i}-${j}`);
            if (cell === 1) {
                const jugadorBlanco = document.createElement('div');
                jugadorBlanco.className = 'jugador-blanco';
                casilla.appendChild(jugadorBlanco);
            } else if (cell === -1) {
                const jugadorRojo = document.createElement('div');
                jugadorRojo.className = 'jugador-rojo';
                casilla.appendChild(jugadorRojo);
            } else if (cell === 2 || cell === -2) {
                const pelota = document.createElement('div');
                pelota.className = 'pelota';
                casilla.appendChild(pelota);
            }
        });
    });
}

function fetchBoardUpdates() {
    fetch("/get_board")
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePieces(data.board); // Update only the pieces
            } else {
                console.error("No board state available:", data.message);
            }
        })
        .catch(err => console.error("Error fetching board updates:", err));
}

function handleClick(row, col) {
    console.log("CLICK DETECTED");
    console.log("Row: " + row + " Col: " + col);

    fetch("/handle_click", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row: row, col: col })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Click data sent successfully");
        } else {
            console.error("Failed to send click data");
        }
    })
    .catch(err => console.error("Error:", err));
}

// Initialize the static board layout
crearTablero();

// Poll for board updates every second
setInterval(fetchBoardUpdates, 1000);

# Chess AI Testing Tool

This Python program allows you to test your chess AI against various opponents using parallel processing. It uses the `python-chess` library and implements several evaluation techniques, including material evaluation and positional evaluation, with a minimax algorithm for move decision-making. The tool is designed for stress-testing an AI's performance by playing multiple games in parallel, collecting game results, and exporting them in CSV and PGN formats.

## Features

- **Opponent Types**: Test your AI against various types of opponents:
  - **Random**: Random move selection.
  - **Material**: Opponent only considers material count (ignores positional advantage).
  - **Basic**: Uses a basic evaluation of the board, considering both material and position.
  
- **Parallel Processing**: Run multiple games simultaneously using Python's `multiprocessing` library for faster testing.

- **Evaluation Functions**: The AI uses a combination of:
  - **Positional Evaluation**: Evaluates piece values and their positions using piece-square tables.
  - **Material Evaluation**: Evaluates only the material balance on the board.
  
- **PGN and CSV Export**: Export game results in PGN format for later analysis or to track game history, and CSV format for statistical analysis.

## Installation

To use this tool, you'll need to install the required dependencies. You can install them using `pip`:

```bash
pip install python-chess
```

## Usage

The script provides a command-line interface to run a test suite of games. You can adjust the number of games, opponent types, and search depth.

### Example Usage

You can run the script directly or customize the parameters. Here's an example of a quick test with 1000 games vs a Material AI opponent:

```python
if __name__ == "__main__":
    print("Chess AI Testing Tool")
    print("=====================\n")
    
    # Run 1000 games against a Material opponent
    results = run_test_suite(
        num_games=1000,
        opponent_type=OpponentType.MATERIAL,
        your_color='both',  # Alternate between white and black
        depth=4,            # Minimax search depth
        num_workers=8       # Number of parallel workers (default: CPU count)
    )
    
    # Export results
    export_to_csv(results, "quick_test_results.csv")
    export_to_pgn(results, "quick_test_games.pgn")
```

### Running a Test Suite

To run a test suite with different configurations:

```python
results = run_test_suite(
    num_games=100,             # Number of games to run
    opponent_type=OpponentType.BASIC,  # Opponent type (RANDOM, MATERIAL, BASIC)
    your_color='both',         # Play both white and black
    depth=3,                   # Minimax depth (1-5)
    num_workers=4              # Number of workers for parallel processing
)
```

### Exporting Results

After running the test suite, you can export the results to CSV and PGN formats:

```python
export_to_csv(results, "test_results.csv")
export_to_pgn(results, "test_games.pgn")
```

## Arguments

- `num_games`: The number of games to play.
- `opponent_type`: Type of opponent (`RANDOM`, `MATERIAL`, `BASIC`).
- `your_color`: Can be `'white'`, `'black'`, or `'both'` (alternates between colors).
- `depth`: Search depth for the minimax algorithm (1-5).
- `num_workers`: Number of parallel workers (default is the number of available CPU cores).

## File Outputs

- **CSV**: Exports game results to a CSV file with the following columns:
  - `game_number`: The game number.
  - `result`: 'win', 'loss', or 'draw'.
  - `your_color`: The color you played (`'white'` or `'black'`).
  - `opponent_type`: Type of opponent (`RANDOM`, `MATERIAL`, `BASIC`).
  - `move_count`: The number of moves played.
  - `moves`: The sequence of moves played in the game.

- **PGN**: Exports the games to a PGN file, including:
  - Event name: "AI Testing Match".
  - Round: The game number.
  - Players: AI vs opponent (depending on color).
  - Result: 1-0, 0-1, or 1/2-1/2 (draw).

## Example Output

The output of the test will include statistics such as the number of wins, losses, and draws, along with performance metrics (total time, games per second, average game time).

```
==============================
Starting test suite:
  Games: 100
  Opponent: material
  Your color: both
  Search depth: 4
  CPU cores: 8
==============================

Testing Complete!
==============================
Results:
  Wins:      50 (50.0%)
  Losses:    40 (40.0%)
  Draws:     10 (10.0%)
  Total:     100

Performance:
  Total time:   120.5s
  Games/sec:    0.83
  Avg game:     1.21s
==============================
```

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, submit issues, or create pull requests. We welcome any contributions, whether bug fixes, improvements, or new features!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

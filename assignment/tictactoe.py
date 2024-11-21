import sys
import random

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for i in range(3):
        print(','.join(board[i*3:(i+1)*3]))

# Function to check if there's a winner
def check_winner(board, player):
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

# Minimax algorithm to find the best move for O
def minimax(board, is_maximizing):
    if check_winner(board, 'x'):
        return -1  # Player X wins
    if check_winner(board, 'o'):
        return 1   # Player O wins
    if '-' not in board:
        return 0   # Draw

    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == '-':
                board[i] = 'o' # Simulating a move
                score = minimax(board, False)
                board[i] = '-' # Undoing the move
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == '-':
                board[i] = 'x'
                score = minimax(board, True)
                board[i] = '-'
                best_score = min(score, best_score)
        return best_score

# Find the best move for Player O
def best_move(board):
    best_score = -float('inf')
    move = -1
    for i in range(9):
        if board[i] == '-':
            board[i] = 'o'
            score = minimax(board, False)
            board[i] = '-'
            if score > best_score:
                best_score = score
                move = i
    return move

# Main function to run the game
def main():
    # Check if a seed is provided as a command-line argument
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        # If no seed is provided, ask the user for input or use a default seed
        seed = int(input("Enter seed value (or leave empty for default): "))

    random.seed(seed)


    # Initialize the board
    board = ['-'] * 9

    # Alternate turns between X and O
    for turn in range(9):
        if turn % 2 == 0:  # Player X (random)
            available_moves = [i for i, x in enumerate(board) if x == '-']
            move = random.choice(available_moves)
            board[move] = 'x'
        else:  # Player O (Minimax)
            move = best_move(board)
            board[move] = 'o'

        # Print the board after each move
        print_board(board)
        print()  # Blank line between moves

        # Check if there's a winner
        if check_winner(board, 'x'):
            print("Player X wins!")
            break
        if check_winner(board, 'o'):
            print("Player O wins!")
            break

    if '-' not in board:
        print("It's a draw!")

# Run the main function
if __name__ == "__main__":
    main()

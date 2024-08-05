from collections import deque

class TakGame:
    def __init__(self, size,ai,depth):
        self.size = size
        self.papan = [[[] for _ in range(size)] for _ in range(size)]
        self.player_turn = 'A'
        self.first_move_made = {'A': False, 'B': False}  # Track if the first move is made
        self.stones = {'A': 0, 'B': 0}  # Tracks the number of stones each player has
        self.capstones = {'A': 0, 'B': 0}  # Tracks the number of capstones each player has
        self.depth = depth
        self.ai = ai

        # Initialize the number of stones and capstones based on the papan size
        if size == 3:
            self.stones['A'] = self.stones['B'] = 10
            self.capstones['A'] = self.capstones['B'] = 0
        elif size == 4:
            self.stones['A'] = self.stones['B'] = 15
            self.capstones['A'] = self.capstones['B'] = 0
        elif size == 5:
            self.stones['A'] = self.stones['B'] = 21
            self.capstones['A'] = self.capstones['B'] = 1
        elif size == 6:
            self.stones['A'] = self.stones['B'] = 30
            self.capstones['A'] = self.capstones['B'] = 1
        elif size == 8:
            self.stones['A'] = self.stones['B'] = 50
            self.capstones['A'] = self.capstones['B'] = 2

    def display_papan(self):
        for row in self.papan:
            print([''.join(cell) if cell else '.' for cell in row])
        print()

    def place_piece(self, x, y, piece_type,temp,player):
        valid_piece_types = ['F', 'C', 'W']  # F for Flat, C for Capstone, W for Wall
        if piece_type not in valid_piece_types:
            print(f"Invalid piece type. Valid types are {valid_piece_types}.")
            return False

        # Check if there are enough stones or capstones before placing
        if piece_type == 'F' or piece_type == 'W':
            if self.stones[player] == 0:
                print("No more stones left to place.")
                return False
            else:
                if temp :
                  self.stones[player] -= 1
        elif piece_type == 'C':
            if self.capstones[player] == 0:
                print("No more capstones left to place.")
                return False
            else:
                if temp:
                  self.capstones[player] -= 1

        if not self.first_move_made[player]:
            # If it's the first move, place the opponent's piece
            opponent =  'B' if player == 'A' else 'A'
            if not self.papan[x][y]:
                self.papan[x][y].append(opponent + piece_type)
                if temp:
                  self.first_move_made[player] = True
                  self.player_turn = 'B' if player == 'A' else 'A'
                return True
            else:
                return False
        else:
            # Normal piece placement for subsequent moves
            if not self.papan[x][y]:
                self.papan[x][y].append(player + piece_type)
                if temp:
                  self.player_turn = 'B' if player == 'A' else 'A'
                return True
            else:
                return False

    def move_stack(self, from_x, from_y, direction, temp_num_pieces,temp,player):
        if isinstance(temp_num_pieces, tuple):
          num_pieces = sum(temp_num_pieces)
        else:
          num_pieces = temp_num_pieces
        if not (0 <= from_x < self.size and 0 <= from_y < self.size) or not self.papan[from_x][from_y]:
          return False

        # Check if the top piece belongs to the current player
        if not self.papan[from_x][from_y][-1].startswith(player):
            return False

        if num_pieces > len(self.papan[from_x][from_y]) or num_pieces > self.size:
            return False

        top_piece = self.papan[from_x][from_y][-1]
        is_capstone = 'C' in top_piece

        # Capstones can't be moved as part of a larger stack
        # if num_pieces > 1 and is_capstone:
        #     return False

        # Calculate path

        # Prepare the stack to be moved
        stack_queue = deque(self.papan[from_x][from_y][-num_pieces:])
        del self.papan[from_x][from_y][-num_pieces:]

        current_x, current_y = from_x, from_y
        moved_pieces = 0
        index = 0
        while moved_pieces < num_pieces and stack_queue:
            # Calculate the next square based on direction
            if direction == 'l':
                current_y -= 1
                if current_y < 0:
                  current_y+=1
            elif direction == 'r':
                current_y += 1
                if current_y == self.size:
                  current_y-=1
            elif direction == 't':
                current_x -= 1
                if current_x < 0:
                  current_x+=1
            elif direction == 'b':
                current_x += 1
                if current_x == self.size:
                  current_x-=1

            # Check boundaries
            if not (0 <= current_x < self.size and 0 <= current_y < self.size) or \
               (self.papan[current_x][current_y] and ('W' in self.papan[current_x][current_y][-1] or 'C' in self.papan[current_x][current_y][-1])):
                    # Check if the top of the stack is a capstone
              is_top_piece_capstone = stack_queue[-1] == player + 'C'
              if is_top_piece_capstone and num_pieces-moved_pieces == 1 and 'W' in self.papan[current_x][current_y][-1]:
                        # Flatten a standing stone if the top piece of the stack is a capstone
                self.papan[current_x][current_y][-1] = self.papan[current_x][current_y][-1][0] + 'F'
                piece = stack_queue.popleft()
                self.papan[current_x][current_y].append(piece)
                moved_pieces += 1
              else:
                      # Place all remaining pieces on the current square
                if direction == 'l':
                  current_y += 1
                elif direction == 'r':
                  current_y -= 1
                elif direction == 't':
                  current_x += 1
                elif direction == 'b':
                  current_x -= 1
                self.papan[current_x][current_y].extend(stack_queue)
                break

          # If no wall/capstone, ask for number of pieces to drop
            else:
              if temp and player != self.ai:
                drop_count = int(input(f"Enter number of pieces to drop on square ({current_x}, {current_y}): "))
              else:
                drop_count = temp_num_pieces[index]
                index+=1
              drop_count = min(drop_count, num_pieces - moved_pieces, len(stack_queue))
              moved_pieces += drop_count

              # Move the specified number of pieces
              for _ in range(drop_count):
                  piece = stack_queue.popleft()
                  self.papan[current_x][current_y].append(piece)

              # self.display_papan()
        if temp :
          self.player_turn = 'B' if player == 'A' else 'A'
        return True

    def is_road(self, player):
        def bfs(start_edge, direction):
            queue = deque(start_edge)
            visited = [[False for _ in range(self.size)] for _ in range(self.size)]
            for x, y in start_edge:
                visited[x][y] = True

            while queue:
                x, y = queue.popleft()
                if (direction == 'horizontal' and x == self.size - 1) or (direction == 'vertical' and y == self.size - 1):
                    return True

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx][ny]:
                        if self.papan[ny][nx]:
                            top_piece = self.papan[ny][nx][-1]
                            if top_piece.startswith(player) and top_piece.endswith(('F', 'C')):
                                visited[nx][ny] = True
                                queue.append((nx, ny))
            return False

        horizontal_start_edges = [(0, y) for y in range(self.size) if self.papan[y][0] and self.papan[y][0][-1].startswith(player) and self.papan[y][0][-1].endswith(('F', 'C'))]
        vertical_start_edges = [(x, 0) for x in range(self.size) if self.papan[0][x] and self.papan[0][x][-1].startswith(player) and self.papan[0][x][-1].endswith(('F', 'C'))]

        has_horizontal_road = bfs(horizontal_start_edges, 'horizontal')
        has_vertical_road = bfs(vertical_start_edges, 'vertical')

        return has_horizontal_road or has_vertical_road

    def check_winner(self):
        # Check for both players
        if self.is_road('A'):
            return 'A'
        elif self.is_road('B'):
            return 'B'
        empty_places = 0
        ownership = {'A': 0, 'B': 0}
        for row in self.papan:
            for cell in row:
                if not cell:  # If the cell is empty
                    empty_places += 1
                else:
                    top_piece = cell[-1]
                    if 'W' not in top_piece:  # Exclude walls
                      if top_piece.startswith('A'):
                          ownership['A'] += 1
                      elif top_piece.startswith('B'):
                          ownership['B'] += 1
        # If no empty places, the game ends and we determine the winner by ownership
        if empty_places == 0:
            if ownership['A'] > ownership['B']:
                return 'A'
            elif ownership['B'] > ownership['A']:
                return 'B'
            else:
                return 'Draw'  # It's a draw if both have the same number of owned places

        # If the game is not over by road or ownership, there is no winner yet
        return None


    def evaluate_papan(self):
        score = 0
        # self.display_papan()
        score += self.evaluate_road_threat()
        score += self.evaluate_control_and_piece_count()
        score += self.evaluate_capstone_positions()
        # print('B' if(self.player_turn=='A') else 'A')
        # print(self.is_road('A'))
        if self.is_road('B' if(self.player_turn=='A') else 'A'):
          score -= float('inf')
          # self.display_papan()
        if self.is_road(self.player_turn):
          score += float('inf')
          # self.display_papan()
        return score

    def evaluate_road_threat(self):
        threat_score = 0
        # for row in range(self.size):
        #     for col in range(self.size):
        #         cell = self.papan[row][col]
        #         if cell and cell[-1].startswith(self.player_turn):  # Check the top piece
        threat_score += self.count_horizontal_road(self.player_turn)
        threat_score += self.count_vertical_road(self.player_turn)
        threat_score -= self.count_horizontal_road('B' if(self.player_turn=='A') else 'A')
        threat_score -= self.count_vertical_road('B' if(self.player_turn=='A') else 'A')

        return threat_score

    def count_horizontal_road(self, player):
        # max_size = self.size
        def bfs_horizontal(start_edge, direction):
            queue = deque(start_edge)
            ctr = 0
            visited = [[False for _ in range(self.size)] for _ in range(self.size)]
            for x, y in start_edge:
                visited[x][y] = True
                ctr+=1

            while queue:
                x, y = queue.popleft()
                if (direction == 'horizontal' and x == self.size - 1):
                    return ctr

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx][ny]:
                        if self.papan[ny][nx]:
                            top_piece = self.papan[ny][nx][-1]
                            if top_piece.startswith(player) and top_piece.endswith(('F', 'C')):
                                visited[nx][ny] = True
                                ctr+=1
                                queue.append((nx, ny))
            return 0
        horizontal_start_edges = [(0, y) for y in range(self.size) if self.papan[y][0] and self.papan[y][0][-1].startswith(player) and self.papan[y][0][-1].endswith(('F', 'C'))]
        road_length = bfs_horizontal(horizontal_start_edges, 'horizontal')
        return road_length if road_length >= self.size else 0

    def count_vertical_road(self, player):
        def bfs_vertical(start_edge, direction):
            queue = deque(start_edge)
            ctr = 0
            visited = [[False for _ in range(self.size)] for _ in range(self.size)]
            for x, y in start_edge:
                visited[x][y] = True
                ctr+=1

            while queue:
                x, y = queue.popleft()
                if (direction == 'vertical' and x == self.size - 1):
                    return ctr

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx][ny]:
                        if self.papan[ny][nx]:
                            top_piece = self.papan[ny][nx][-1]
                            if top_piece.startswith(player) and top_piece.endswith(('F', 'C')):
                                visited[nx][ny] = True
                                ctr+=1
                                queue.append((nx, ny))
            return 0
        horizontal_start_edges = [(0, y) for y in range(self.size) if self.papan[y][0] and self.papan[y][0][-1].startswith(player) and self.papan[y][0][-1].endswith(('F', 'C'))]
        road_length = bfs_vertical(horizontal_start_edges, 'vertical')

        return road_length if road_length >= self.size else 0

    def check_horizontal_road(self, row, col, player):
        # max_size = self.size
        # road_length = 0

        # for c in range(max_size):
        #     if self.is_road_piece(row, c, player):
        #         road_length += 1
        #     else:
        #         break


        # return road_length if road_length >= max_size else 0
        max_size = self.size
        road_length = 0


        # Check to the right
        for c in range(col, max_size):
            if self.is_road_piece(row, c, player):
                road_length += 1
            else:
                break


        # Check to the left, not including the starting cell
        for c in range(col - 1, -1, -1):
            if self.is_road_piece(row, c, player):
                road_length += 1
            else:
                break


        return road_length



    def check_vertical_road(self, row, col, player):
        # max_size = self.size
        # road_length = 0


        # for r in range(max_size):
        #     if self.is_road_piece(r, col, player):
        #         road_length += 1
        #     else:
        #         break


        # return road_length if road_length >= max_size else 0
        max_size = self.size
        road_length = 0


        # Check downwards
        for r in range(row, max_size):
            if self.is_road_piece(r, col, player):
                road_length += 1
            else:
                break


        # Check upwards, not including the starting cell
        for r in range(row - 1, -1, -1):
            if self.is_road_piece(r, col, player):
                road_length += 1
            else:
                break


        return road_length



    def is_road_piece(self, row, col, player):
        if not self.papan[row][col]:
            return False  # Empty cell
        top_piece = self.papan[row][col][-1]
        return top_piece.startswith(player) and 'W' not in top_piece

    def evaluate_capstone_positions(self):
        player1_score = 0
        player2_score = 0

        for row in range(self.size):
            for col in range(self.size):
                cell = self.papan[row][col]
                if cell:
                    top_piece = cell[-1]
                    piece_type, player = top_piece[1], top_piece[0]
                    if piece_type == 'C':
                        score = self.assess_capstone_position(row, col, player)
                        if player == 'A':
                            player1_score += score
                        elif player == 'B':
                            player2_score += score

        return player1_score - player2_score

    def assess_capstone_position(self, row, col, player):
        score = 0
        score += self.check_capstone_mobility(row, col)
        score += self.check_road_contribution(row, col, player)
        return score

    def check_capstone_mobility(self, row, col):
        mobility_score = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Directions: right, left, down, up

        for d_row, d_col in directions:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                if not self.papan[new_row][new_col] or self.is_flattenable_standing_stone(new_row, new_col):
                    mobility_score += 1

        return mobility_score

    def is_flattenable_standing_stone(self, row, col):

        cell = self.papan[row][col]
        if cell:
            top_piece = cell[-1]
            return 'W' in top_piece  # Check if the top piece is a wall stone
        return False

    def check_road_contribution(self, row, col, player):
        contribution_score = 0

        # Check if the capstone is part of a potential road for the player
        contribution_score += self.check_potential_road_contribution(row, col, player)

        # Check if the capstone is blocking an opponent's potential road
        opponent = 'B' if player == 'A' else 'A'
        contribution_score += self.check_blocking_opponent_road(row, col, opponent)

        return contribution_score

    def check_potential_road_contribution(self, row, col, player):
        # Check the contribution of the capstone in potential road formations
        potential_road_score = 0
        potential_road_score += self.check_horizontal_road(row, col, player)
        potential_road_score += self.check_vertical_road(row, col, player)
        return potential_road_score

    def check_blocking_opponent_road(self, row, col, opponent):
        # Check if the capstone is positioned to block an opponent's potential road
        blocking_score = 0
        blocking_score += self.check_horizontal_road(row, col, opponent)
        blocking_score += self.check_vertical_road(row, col, opponent)
        return blocking_score

    def evaluate_control_and_piece_count(self):
        player1_score, player2_score = 0, 0
        piece_values = {'F': 1, 'W': 0.5, 'C': 2}  # Assuming 'F' for flat, 'W' for standing, 'C' for capstone

        for row in self.papan:
            for cell in row:
                if cell:
                    piece_type, player = cell[-1][1], cell[-1][0]
                    if player == 'A':
                        player1_score += piece_values[piece_type]
                    elif player == 'B':
                        player2_score += piece_values[piece_type]

        return player1_score - player2_score

    def choose_best_move(self):
      best_score = float('-inf')
      best_move = None
      print(self.player_turn)
      # print(self.generate_legal_moves())
      for move in self.generate_legal_moves(self.player_turn):
          # print(move)
          affected_cells = self.apply_move(move,False,self.player_turn)

          score = self.minimax(self.depth, float('-inf'), float('inf'), False)

          self.revert_move(move,affected_cells)
          # print(move,score,best_score)
          if score > best_score:
              best_score = score
              best_move = move
      if best_move == None:
        best_move = self.generate_legal_moves(self.player_turn)[0]
      print(best_move)
      return best_move

    def is_papan_full(self):
        for row in self.papan:
            for cell in row:
                if not cell:  # If the cell is an empty list, it's an empty space.
                    return False  # papan is not full
        return True  # No empty spaces found, papan is full

    def is_terminal(self):
        winner = self.check_winner()
        if winner:
            return True, winner
        if self.is_papan_full():  # Game ends if the papan is full
            return True, None
        return False, None

    def minimax(self, depth, alpha, beta, maximizing_player):
        terminal, winner = self.is_terminal()
        if depth == 0 or terminal:
            return self.evaluate_papan()

        if maximizing_player:
            max_eval = float('-inf')
            # print(self.generate_legal_moves())
            # print(depth)
            for move in self.generate_legal_moves(self.player_turn):
                affected_cells = self.apply_move(move,False,self.player_turn)
                # self.apply_move(move,False)
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.revert_move(move,affected_cells)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.generate_legal_moves('B' if(self.player_turn=='A') else 'A'):
                affected_cells = self.apply_move(move, False,'B' if(self.player_turn=='A') else 'A')
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.revert_move(move,affected_cells)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def apply_move(self, move, temp,player):
        # print(f"Applying move: {move}")
        move_type = move[0]
        affected_cells = None
        if move_type == 'place':
            _, x, y, piece_type = move
            if temp:
              self.place_piece(x, y, piece_type, True,player)
            else:
              self.place_piece(x, y, piece_type, False,player)
        elif move_type == 'move_stack':
            _, from_x, from_y, direction, drop_sequence = move
            if temp:
              self.move_stack(from_x, from_y, direction, drop_sequence,True,player)
            else:
              affected_cells = self.calculate_affected_cells(from_x, from_y, direction, drop_sequence)
              self.move_stack(from_x, from_y, direction, drop_sequence,False,player)
        return affected_cells

    def revert_move(self, move,affected_cells):
        # print(f"Revert move: {move}")
        move_type = move[0]
        if move_type == 'place':
            _, x, y, _ = move
            self.papan[x][y].pop()

        elif move_type == 'move_stack':
            _, from_x, from_y, direction, drop_sequence = move
            # Calculate the affected cells and revert their states
            # print(affected_cells)
            self.restore_stack(affected_cells)
            # del self.affected_cells_before_move
            # affected_cells = self.calculate_affected_cells(from_x, from_y, direction, drop_sequence)
            # print(affected_cells)
            # self.restore_stack(affected_cells)
            # self.display_papan()

    def calculate_affected_cells(self, from_x, from_y, direction, drop_sequence):
        affected_cells = []
        current_x, current_y = from_x, from_y

        # Include the original cell's state
        affected_cells.append(((from_x, from_y), list(self.papan[from_x][from_y])))

        # Handle drop_sequence as a single number or a tuple
        if isinstance(drop_sequence, int):
            drop_sequence = [drop_sequence] * self.size

        for drop_count in drop_sequence:
            if drop_count == 0:
                continue

            # Move to the next cell based on direction
            if direction == 'l':
                current_y -= 1
            elif direction == 'r':
                current_y += 1
            elif direction == 't':
                current_x -= 1
            elif direction == 'b':
                current_x += 1

            # Check boundaries
            if not (0 <= current_x < self.size and 0 <= current_y < self.size):
                break

            affected_cells.append(((current_x, current_y), list(self.papan[current_x][current_y])))

        return affected_cells

    def restore_stack(self, affected_cells):
        for cell_info in reversed(affected_cells):
            coordinates, previous_state = cell_info
            x, y = coordinates
            self.papan[x][y] = list(previous_state)

    def generate_legal_moves(self,player):
        legal_moves = []

        # Generate place moves for each cell if the cell is empty
        for x in range(self.size):
            for y in range(self.size):
                if not self.papan[x][y]:  # If the cell is empty
                    if self.stones[player] > 0:  # If the player has stones left
                        legal_moves.append(('place', x, y, 'F'))  # Place a flat stone
                        legal_moves.append(('place', x, y, 'W'))  # Place a flat stone
                    if self.capstones[player] > 0:  # If the player has capstones left
                        legal_moves.append(('place', x, y, 'C'))  # Place a capstone

        # Generate move stack actions
        for x in range(self.size):
            for y in range(self.size):
                stack = self.papan[x][y]
                if stack and stack[-1].startswith(player):  # If the cell has a stack that belongs to the current player
                    # Determine the maximum number of pieces that can be moved
                    max_height = min(len(stack), self.size)
                    for height in range(1, max_height + 1):
                        # Generate all distributions for moving 'height' number of pieces
                        for drop_sequence in self.generate_drop_sequences(height):
                            for direction in ['l', 'r', 't', 'b']:
                              valid = True
                              if direction == 'l':
                                if y-1<0:
                                  valid = False
                                if y-1 >=0 and self.papan[x][y-1] and ('W' in self.papan[x][y-1][-1] or 'C' in self.papan[x][y-1][-1]) and 'C' not in stack:
                                  valid = False
                              elif direction == 'r':
                                if y+1 > self.size:
                                  valid = False
                                if y+1<self.size and self.papan[x][y+1] and ('W' in self.papan[x][y+1][-1] or 'C' in self.papan[x][y+1][-1]) and 'C' not in stack:
                                  valid = False
                              elif direction == 't':
                                if x-1 < 0:
                                  valid = False
                                if x-1>=0 and self.papan[x-1][y] and ('W' in self.papan[x-1][y][-1] or 'C' in self.papan[x-1][y][-1]) and 'C' not in stack:
                                  valid = False
                              elif direction == 'b':
                                if x+1 > self.size:
                                  valid = False
                                if x+1 < self.size and self.papan[x+1][y] and ('W' in self.papan[x+1][y][-1] or 'C' in self.papan[x+1][y][-1]) and 'C' not in stack:
                                  valid = False
                              if valid:
                                legal_moves.append(('move_stack', x, y, direction, drop_sequence))

        return legal_moves

    def generate_drop_sequences(self, height):
        # Base case: if height is 1, there's only one way to drop the pieces
        if height == 1:
            return [(1,)]

        drop_sequences = []
        for first_drop in range(1, height + 1):
            remaining_height = height - first_drop
            if remaining_height == 0:
                # If there's nothing left, it's a valid sequence
                drop_sequences.append((first_drop,))
            else:
                # Otherwise, get all sequences for the remaining height and append
                for subsequent_drops in self.generate_drop_sequences(remaining_height):
                    drop_sequences.append((first_drop,) + subsequent_drops)

        return drop_sequences

# Game Loop
game = TakGame(int(input("Size (3,4,5,6,8) : ")),input("AI Turn ( Player 1 (A) Player 2 (B) ) : ").upper(),int(input("AI Depth : "))-1)
game.display_papan()

while True:
    if game.player_turn==game.ai:
      best_move = game.choose_best_move()
      print("AI")
      game.apply_move(best_move,True,game.player_turn)
      game.display_papan()
    else:
      print(f"Player {game.player_turn}'s turn")
      action = input("Place (p) or move (m) a piece? ").lower()

      if action == 'p':
          x = int(input("Enter row to place: "))
          y = int(input("Enter column to place: "))
          piece_type = input("Enter piece type (F for Flat, C for Capstone, W for Wall): ").upper()
          if game.place_piece(x, y, piece_type, True,game.player_turn):
              game.display_papan()
          else:
              print("Invalid placement, try again.")
      elif action == 'm':
          from_x = int(input("Enter row to move from: "))
          from_y = int(input("Enter column to move from: "))
          direction = input("Enter direction to move (left (l), right (r), top (t), bottom (b)): ").lower()
          num_pieces = int(input("Enter number of pieces to move: "))
          if game.move_stack(from_x, from_y, direction, num_pieces,True,game.player_turn):
              game.display_papan()
          else:
              print("Invalid move, try again.")
    winner = game.check_winner()
    if winner:
      if winner == 'Draw':
        print(f"Draw")
      else:
        print(f"Player {winner} wins!")
      break
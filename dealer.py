import time
import pandas as pd
import threading as t

class Dealer(t.Thread):
    def __init__(self, games: int):
        super(Dealer, self).__init__(name='dealer')
        self.games = games
        self.turns = ['A', 'B', 'C', 'D', 'E']
        self.suite_cycle = ['S', 'D', 'C', 'H']
        self.card_cycle = [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]
        self.game_table = pd.DataFrame({
            'TRUMP SUITE': [self.suite_cycle[i % 4] for i in range(self.games)],
            'CARDS': [self.card_cycle[i % len(self.card_cycle)] for i in range(self.games)],})
        score_table = pd.DataFrame({player: [None for i in range(self.games)] for player in self.turns})
        self.game_table = pd.concat([self.game_table, score_table], axis=1)

    def cycle_players(self, previous_turn, game):
        turn = previous_turn
        for _ in range(game):
            turn = previous_turn[1:] + [previous_turn[0]]
        return turn

    def run(self):
        for game in range(self.game_table.shape[0]):
            cards = self.game_table['CARDS'][game]
            trump_suite = self.game_table['TRUMP SUITE'][game]
            self.turns = self.cycle_players(self.turns, game)
            print('TURNS:', self.turns)
            print('DEAL THE CARDS....')
            time.sleep(10)

    def start_game(self):
        self.start()

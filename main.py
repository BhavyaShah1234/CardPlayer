import time
import eyes as e
import brain as b
import camera as c
import pandas as pd
import logging as l

def create_game_table(games, suite_cycle, card_cycle, turns):
    game_table = pd.DataFrame({
        'TRUMP_SUITE': [suite_cycle[i % 4] for i in range(games)],
        'CARDS': [card_cycle[i % len(card_cycle)] for i in range(games)],
        })
    score_table = pd.DataFrame({player: [None for i in range(games)] for player in turns})
    game_table = pd.concat([game_table, score_table], axis=1)
    return game_table

def cycle_players(previous_turn, game):
    turn = previous_turn
    for _ in range(game):
        turn = previous_turn[1:] + [previous_turn[0]]
    return turn

def get_new_text_id(texts):
    return max(list(texts.keys())) + 1

def add_text(texts, text_id, text, position, camera):
    texts[text_id] = {'text': text, 'position': position}
    camera.update_text(texts)
    return texts

def remove_text(texts, text_id, camera):
    texts.pop(text_id)
    camera.update_text(texts)
    return texts

def decide_hands(trump_suite, total_cards, player_turns, expected_hands, our_cards, model):
    prompt = f"We are playing cards, specifically 'judgement' where one has to score exactly the number of hands they predict at the start of the game. The game currently has trump suite as {trump_suite} and {total_cards} cards per player. If the number of players times cards per player doesn't add to 52 then some lower cards have been burried after dealing. The players in the game are {player_turns} and play in the same order. The number of expected hands are given by {expected_hands}. If the keys in dictionary are less than players that means that some of the players' turn is after ours. The cards we have are {our_cards}. How many hands can we make with these cards?"
    # hands = model.decide_hands(prompt)
    hands = 3
    return hands

if __name__ == '__main__':
    WEIGHT_PATH = r'weights\best.pt'
    CAMERA_ID = 1
    IMG_H, IMG_W = 720, 1280
    GAMES = 1
    DEAL_TIME = 5
    PLAYER_TURNS = ['A', 'B', 'C', 'COMPUTER']
    SUITE_CYCLE = ['S', 'D', 'C', 'H']
    CARD_CYCLE = [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    LOGGER = l.getLogger(__name__)
    l.basicConfig(filename='system.log', filemode='w', level=l.DEBUG, format='[%(levelname)s] %(filename)s---->%(message)s')
    GAME_TABLE = create_game_table(GAMES, SUITE_CYCLE, CARD_CYCLE, PLAYER_TURNS)
    print(GAME_TABLE)
    LOGGER.info(f'\n\n{GAME_TABLE}')
    YOLO_MODEL = e.Detector(WEIGHT_PATH)
    LLM_MODEL = b.Brain()
    print('MODELS LOADED')
    # LOGGER.info('MODELS LOADED')
    CAMERA = c.Camera(CAMERA_ID, IMG_H, IMG_W, LOGGER)
    CAMERA.start_capturing()
    print('CAMERA ON')
    # LOGGER.info('CAMERA ON')
    #######################################################################################################
    TEXTS = {}
    #######################################################################################################
    for game in range(GAME_TABLE.shape[0]):
        # LOGGER.info(f'GAME {game}#########################################')
        TEXTS = add_text(TEXTS, 0, f'GAME: {game}', 'game', CAMERA)

        total_cards = GAME_TABLE['CARDS'][game]
        # LOGGER.info(f'TOTAL CARDS: {total_cards}')
        TEXTS = add_text(TEXTS, 1, f'TOTAL CARDS: {total_cards}', 'total', CAMERA)

        trump_suite = GAME_TABLE['TRUMP_SUITE'][game]
        # LOGGER.info(f'TRUMP SUITE: {trump_suite}')
        TEXTS = add_text(TEXTS, 2, f'TRUMP SUITE: {trump_suite}', 'trump', CAMERA)

        PLAYER_TURNS = cycle_players(PLAYER_TURNS, game)
        TEXTS = add_text(TEXTS, 3, f'TURNS: {PLAYER_TURNS}', 'turn', CAMERA)
        # LOGGER.info(f'PLAYER TURNS: {PLAYER_TURNS}')

        print('DEFAULT TEXT ADDED ON SCREEN')
        LOGGER.info(f'DEFAULT TEXT ADDED ON SCREEN')
        ###################################################################################################
        TEXTS = add_text(TEXTS, 4, f'DEAL THE CARDS IN {DEAL_TIME} SECONDS', 'deal', CAMERA)
        # LOGGER.info(f'DEALING....')
        time.sleep(DEAL_TIME)
        TEXTS = remove_text(TEXTS, 4, CAMERA)
        print('DEALING DONE')
        LOGGER.info('DEALING DONE')
        ###################################################################################################
        expected_hands = {}
        text_id = get_new_text_id(TEXTS)
        for i, player in enumerate(PLAYER_TURNS):
            if player == 'COMPUTER':
                TEXTS = add_text(TEXTS, text_id, f'COMPUTER IS DECIDING...', f'decision', CAMERA)
                # LOGGER.info('COMPUTER IS DECIDING....')
                OUR_CARDS = {}
                while len(OUR_CARDS) != total_cards:
                    current_frame = CAMERA.current_frame
                    OUR_CARDS = YOLO_MODEL.detect(current_frame)
                    OUR_CARDS = YOLO_MODEL.post_process(OUR_CARDS)
                LOGGER.info(f'OUR CARDS: {OUR_CARDS}')
                expected_hands[player] = decide_hands(trump_suite, total_cards, PLAYER_TURNS, expected_hands, OUR_CARDS, LLM_MODEL)
            else:
                TEXTS = add_text(TEXTS, text_id, f'PLAYER {player}, DECIDE NUMBER OF HANDS', 'decision', CAMERA)
                key = CAMERA.key
                while key < 48 or key > 56:
                    key = CAMERA.key
                expected_hands[player] = key - 48
            LOGGER.info(f'{player} DECIDED {expected_hands[player]} HANDS')
            TEXTS = add_text(TEXTS, text_id + 1 + i, f'{player}: {expected_hands[player]}', f'hands_{i}', CAMERA)
            TEXTS = remove_text(TEXTS, text_id, CAMERA)
        print('HAND SELECTION DONE')
        LOGGER.info('HAND SELECTION DONE')
        ###################################################################################################
        text_id = get_new_text_id(TEXTS)
        for round in range(total_cards):
            played_cards = {}
            for i, player in enumerate(PLAYER_TURNS):
                if player == 'COMPUTER':
                    pass
                else:
                    TEXTS = add_text(TEXTS, text_id, f"PLAYER {player}'S TURN. SHOW CARD YOU WANT TO PLAY IN THE CAMERA", f'played_{i}', CAMERA)
                    card = {}
                    while len(card) != 1:
                        current_frame = CAMERA.current_frame
                        card = YOLO_MODEL.detect(current_frame)
                        card = YOLO_MODEL.post_process(card)
                    played_cards[player] = card
                TEXTS = remove_text(TEXTS, text_id, CAMERA)
        print('ROUNDS ENDED')
        LOGGER.info('ROUND ENDED')
        ###################################################################################################
        TEXTS = remove_text(TEXTS, 3, CAMERA)
        TEXTS = remove_text(TEXTS, 2, CAMERA)
        TEXTS = remove_text(TEXTS, 1, CAMERA)
        TEXTS = remove_text(TEXTS, 0, CAMERA)
        print('GAME COMPLETE')
        LOGGER.info('GAME COMPLETE')
        ###################################################################################################
    CAMERA.set_exit_event()
    CAMERA.join_thread()

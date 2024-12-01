import time
import eyes as e
import brain as b
import camera as c
import pandas as pd
import random as rn
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

def add_text(texts: dict, text_id: int, text: str, position: str, color: str, camera):
    texts[text_id] = {'text': text, 'position': position, 'color': color}
    camera.update_text(texts)
    return texts

def remove_text(texts, text_id, camera):
    texts.pop(text_id)
    camera.update_text(texts)
    return texts

def decide_hands(trump_suite, total_cards, player_turns, expected_hands, our_cards, forbidden, model):
    if forbidden is not None:
        prompt = f"We are playing cards, specifically 'judgement' where one has to score exactly the number of hands they predict at the start of the game. The game currently has trump suite as {trump_suite} and {total_cards} cards per player. If the number of players times cards per player doesn't add to 52 then some lower cards have been burried after dealing. The players in the game are {player_turns} and play in the same order. ME is our player. The number of expected hands are given by {expected_hands}. If the keys in dictionary are less than players that means that some of the players' turn is after ours so we don't have any idea about their decisions. The cards we have are {our_cards}. How many hands can we make with these cards? Answer only in integer numbers ranging from 0 to {total_cards} and {forbidden} is forbidden."
    else:
        prompt = f"We are playing cards, specifically 'judgement' where one has to score exactly the number of hands they predict at the start of the game. The game currently has trump suite as {trump_suite} and {total_cards} cards per player. If the number of players times cards per player doesn't add to 52 then some lower cards have been burried after dealing. The players in the game are {player_turns} and play in the same order. ME is our player. The number of expected hands are given by {expected_hands}. If the keys in dictionary are less than players that means that some of the players' turn is after ours so we don't have any idea about their decisions. The cards we have are {our_cards}. How many hands can we make with these cards? Answer only in integer numbers ranging from 0 to {total_cards}."
    # hands = int(model.decide_hands(prompt))
    hands = rn.randint(0, total_cards)
    return hands

def decide_card(played_cards: dict, our_cards: dict, hands: dict, model, logger):
    prompt = f"It is our turn now. These are our cards left in hand: {our_cards}. The players before us played the following cards in the following order: {played_cards}. We have to make {hands['expected']} hands from which we have made {hands['made']} hands. Which card should we play next? Answer in terms of SA, HK, CQ, DJ and should be present in our cards."
    # card = model.decide_card(prompt)
    return rn.choice(list(our_cards.keys()))

def decide_round_winner(played_cards: dict, trump_suite: str):
    winner = None
    max_score = -1
    suite_rank = {trump_suite: 2, list(played_cards.items())[0][1][0]: 1}
    for suite in ['S', 'H', 'C', 'D']:
        if suite not in suite_rank:
            suite_rank[suite] = 0
    card_rank = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    for player in played_cards:
        card = played_cards[player]
        score = suite_rank[card[0]] * card_rank[card[1:]]
        if score > max_score:
            max_score = score
            winner = player
    index = list(played_cards.keys()).index(winner)
    return index, winner

if __name__ == '__main__':
    WEIGHT_PATH = r'weights\best.pt'
    CAMERA_ID = 0
    IMG_H, IMG_W = 640, 640
    GAMES = 1
    DEAL_TIME = 5
    PLAY_TIME = 5
    TURN_TIME = 3
    WIN_TIME = 4
    PLAYER_TURNS = ['A', 'B', 'C', 'ME']
    SUITE_CYCLE = ['S', 'D', 'C', 'H']
    CARD_CYCLE = [5, 4, 3, 2, 1, 1, 2, 3, 4, 5] # [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]
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
    #######################################################################################################
    TEXTS = {}
    #######################################################################################################
    for game in range(GAME_TABLE.shape[0]):
        TEXTS = add_text(TEXTS, 0, f'GAME: {game}', 'game', 'info', CAMERA)
        TOTAL_CARDS = GAME_TABLE['CARDS'][game]
        LOGGER.error(f'TOTAL CARDS: {TOTAL_CARDS}')
        TEXTS = add_text(TEXTS, 1, f'TOTAL CARDS: {TOTAL_CARDS}', 'total', 'info', CAMERA)
        TRUMP_SUITE = GAME_TABLE['TRUMP_SUITE'][game]
        LOGGER.error(f'TRUMP SUITE: {TRUMP_SUITE}')
        TEXTS = add_text(TEXTS, 2, f'TRUMP SUITE: {TRUMP_SUITE}', 'trump', 'info', CAMERA)
        PLAYER_TURNS = cycle_players(PLAYER_TURNS, game)
        LOGGER.error(f'PLAYERS: {PLAYER_TURNS}')
        TEXTS = add_text(TEXTS, 3, f'TURNS: {PLAYER_TURNS}', 'turn', 'info', CAMERA)
        ###################################################################################################
        for i in range(DEAL_TIME, 0, -1):
            TEXTS = add_text(TEXTS, 4, f'DEAL THE CARDS IN {i} SECONDS', 'deal', 'info', CAMERA)
            time.sleep(1)
            TEXTS = remove_text(TEXTS, 4, CAMERA)
        print('DEALING DONE')
        LOGGER.info('DEALING DONE')
        ###################################################################################################
        hand_decision = {}
        text_id = get_new_text_id(TEXTS)
        for i, player in enumerate(PLAYER_TURNS):
            hand_decision[player] = {'text_id': None, 'made': 0, 'expected': 0}
            sum_of_cards = sum([hand_decision[p]['expected'] for p in hand_decision])
            if player == 'ME':
                TEXTS = add_text(TEXTS, text_id, f'{player} IS DECIDING...', f'decision', 'info', CAMERA)
                our_cards = {}
                while len(our_cards) != TOTAL_CARDS:
                    current_frame = CAMERA.current_frame
                    our_cards = YOLO_MODEL.detect(current_frame)
                    our_cards = YOLO_MODEL.post_process(our_cards)
                LOGGER.debug(f'OUR CARDS: {our_cards}')
                if i == len(PLAYER_TURNS) - 1 and sum_of_cards <= TOTAL_CARDS:
                    forbidden = TOTAL_CARDS - sum_of_cards
                else:
                    forbidden = None
                expected_hands = decide_hands(TRUMP_SUITE, TOTAL_CARDS, PLAYER_TURNS, hand_decision, our_cards, forbidden, LLM_MODEL)
                hand_decision[player]['expected'] = expected_hands
            else:
                if i == len(PLAYER_TURNS) - 1 and sum_of_cards <= TOTAL_CARDS:
                    forbidden = TOTAL_CARDS - sum_of_cards
                    TEXTS = add_text(TEXTS, text_id, f'PLAYER {player}, DECIDE NUMBER OF HANDS. YOU CANNOT CHOOSE {forbidden}.', 'decision', 'info', CAMERA)
                    expected_hands = CAMERA.key - 48
                    while expected_hands > TOTAL_CARDS or expected_hands < 0 or expected_hands == forbidden:
                        expected_hands = CAMERA.key - 48
                else:
                    TEXTS = add_text(TEXTS, text_id, f'PLAYER {player}, DECIDE NUMBER OF HANDS', 'decision', 'info', CAMERA)
                    expected_hands = CAMERA.key - 48
                    while expected_hands > TOTAL_CARDS or expected_hands < 0:
                        expected_hands = CAMERA.key - 48
                hand_decision[player]['expected'] = expected_hands
            CAMERA.key = -1
            LOGGER.debug(f"[DECISION] {player} DECIDED {expected_hands} HANDS")
            TEXTS = add_text(TEXTS, text_id + 1 + i, f"{player}: {hand_decision[player]['made']} / {hand_decision[player]['expected']}", f'hands_{player}', 'good' if hand_decision[player]['made'] == expected_hands else 'bad', CAMERA)
            hand_decision[player]['text_id'] = text_id + 1 + i
            TEXTS = remove_text(TEXTS, text_id, CAMERA)
        print('HAND SELECTION DONE')
        LOGGER.debug('HAND SELECTION DONE')
        LOGGER.debug(f'EXPECTED HANDS: {hand_decision}')
        ###################################################################################################
        for i in range(PLAY_TIME, 0, -1):
            TEXTS = add_text(TEXTS, text_id, f'ROUND STARTS IN {i} SECONDS', 'deal', 'info', CAMERA)
            time.sleep(1)
            TEXTS = remove_text(TEXTS, text_id, CAMERA)
        print('ROUND START')
        LOGGER.info('ROUND START')
        LOGGER.info('===============================================================================')
        ###################################################################################################
        text_id = get_new_text_id(TEXTS)
        for _ in range(TOTAL_CARDS, 0, -1):
            played_cards = {player: '' for player in PLAYER_TURNS}
            for i, player in enumerate(PLAYER_TURNS):
                TEXTS = add_text(TEXTS, text_id + 1 + i, f'{player}: {played_cards[player]}.', f'played_{player}', 'info', CAMERA)
                if player == 'ME':
                    TEXTS = add_text(TEXTS, text_id, f"{player}'S TURN. ME IS DECIDING....", 'decision', 'info', CAMERA)
                    played_cards[player] = decide_card(played_cards, our_cards, hand_decision[player], LLM_MODEL, LOGGER)
                else:
                    TEXTS = add_text(TEXTS, text_id, f"{player}'S TURN. SHOW CARD YOU WANT TO PLAY IN THE CAMERA", 'decision', 'info', CAMERA)
                    card = {}
                    while len(card) != 1:
                        current_frame = CAMERA.current_frame
                        card = YOLO_MODEL.detect(current_frame)
                        card = YOLO_MODEL.post_process(card)
                        print(f'{player} {card}')
                    played_cards[player] = list(card.keys())[0]
                TEXTS = remove_text(TEXTS, text_id, CAMERA)
                TEXTS = add_text(TEXTS, text_id + 1 + i, f'{player}: {played_cards[player]}.', f'played_{player}', 'info', CAMERA)
                LOGGER.info(f'{player} PLAYED {card}')
                time.sleep(TURN_TIME)
            index, winner = decide_round_winner(played_cards, TRUMP_SUITE)
            hand_decision[winner]['made'] = hand_decision[winner]['made'] + 1
            LOGGER.info(f'{winner} WON THE ROUND')
            print(f'{winner} WON THE ROUND')
            TEXTS = add_text(TEXTS, text_id, f"{winner} WON THE ROUND.", 'decision', 'info', CAMERA)
            LOGGER.error(f'HANDS: {hand_decision}')
            time.sleep(WIN_TIME)
            TEXTS = remove_text(TEXTS, text_id, CAMERA)
            TEXTS = add_text(TEXTS, hand_decision[winner]['text_id'], f"{player}: {hand_decision[winner]['made']} / {hand_decision[winner]['expected']}", f"hands_{winner}", 'good' if hand_decision[winner]['made'] == expected_hands else 'bad', CAMERA)
            PLAYER_TURNS = cycle_players(PLAYER_TURNS, index)
            print('ROUNDS ENDED')
            LOGGER.info('ROUND ENDED')
            LOGGER.error(f'PLAYER TURNS: {PLAYER_TURNS}')
            LOGGER.info('============================================================================')
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

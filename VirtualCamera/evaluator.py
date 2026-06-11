from itertools import combinations
from collections import Counter

RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10, "T": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14
}

HAND_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "One Pair",
    0: "High Card"
}


def parse_card(card):
    """
    Expected examples:
    AH, KD, QS, JC, 10H, 7S
    """

    card = card.upper().replace(" ", "").replace("_", "").replace("-", "")

    suit = card[-1]
    rank = card[:-1]

    if rank not in RANK_VALUES:
        raise ValueError(f"Unknown rank: {rank} in card {card}")

    if suit not in ["H", "D", "C", "S"]:
        raise ValueError(f"Unknown suit: {suit} in card {card}")

    return RANK_VALUES[rank], suit


def straight_high(values):
    unique_values = sorted(set(values), reverse=True)

    # Ace can also be low: A, 2, 3, 4, 5
    if 14 in unique_values:
        unique_values.append(1)

    for i in range(len(unique_values) - 4):
        window = unique_values[i:i + 5]

        if window[0] - window[4] == 4:
            return window[0]

    return None


def evaluate_five_cards(cards):
    parsed = [parse_card(card) for card in cards]

    values = [v for v, s in parsed]
    suits = [s for v, s in parsed]

    counts = Counter(values)
    count_groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    high_straight = straight_high(values)

    # Straight Flush
    if is_flush and high_straight:
        return (8, [high_straight])

    # Four of a Kind
    if count_groups[0][1] == 4:
        four = count_groups[0][0]
        kicker = max(v for v in values if v != four)
        return (7, [four, kicker])

    # Full House
    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        three = count_groups[0][0]
        pair = count_groups[1][0]
        return (6, [three, pair])

    # Flush
    if is_flush:
        return (5, sorted(values, reverse=True))

    # Straight
    if high_straight:
        return (4, [high_straight])

    # Three of a Kind
    if count_groups[0][1] == 3:
        three = count_groups[0][0]
        kickers = sorted([v for v in values if v != three], reverse=True)
        return (3, [three] + kickers)

    pairs = [value for value, count in counts.items() if count == 2]

    # Two Pair
    if len(pairs) == 2:
        pairs = sorted(pairs, reverse=True)
        kicker = max(v for v in values if v not in pairs)
        return (2, pairs + [kicker])

    # One Pair
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted([v for v in values if v != pair], reverse=True)
        return (1, [pair] + kickers)

    # High Card
    return (0, sorted(values, reverse=True))


def best_poker_hand(cards):
    """
    Takes 5 to 7 cards and returns the best possible 5-card hand.
    """

    if len(cards) < 5:
        return None, None

    best_score = None
    best_cards = None

    for five_cards in combinations(cards, 5):
        score = evaluate_five_cards(five_cards)

        if best_score is None or score > best_score:
            best_score = score
            best_cards = five_cards

    hand_rank = best_score[0]
    hand_name = HAND_NAMES[hand_rank]

    return hand_name, list(best_cards)
from rapidfuzz import process

def normalize_name(input_name, player_list, score_cutoff=60):
    """
    Fuzzy match an input name to the closest real ATP player name.
    Returns the best match if similarity is above score_cutoff,
    otherwise returns input_name unchanged.
    """
    input_name = input_name.strip()

    if not player_list:
        return input_name

    match, score, _ = process.extractOne(input_name, player_list)

    if score >= score_cutoff:
        return match
    return input_name
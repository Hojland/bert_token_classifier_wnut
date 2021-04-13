import unicodedata
from collections import defaultdict


def _reconstruct_tokens_from_word_piece(wordpiece_pred_seq):
    tokens, token_preds = [], []
    for wp, p in wordpiece_pred_seq:
        if wp.startswith("##"):
            tokens[-1] += wp[2:]
            # Ensure all wordpieces of same word are labelled in the same way.
            # Warning: This prioritizes B-label's before I-label's, but the specific ordering
            # depends on your actual labels (string comparison).
            token_preds[-1] = min(token_preds[-1], p)
        else:
            tokens.append(wp)
            token_preds.append(p)
    return tokens, token_preds


def _strip_label(lab):
    """ Removed BIO label prefixes. E.g. B-label1 --> label1, and I-label2 --> label2."""
    return lab if lab == "O" else lab[2:]


def _get_entity_spans(text, token_label_pairs):
    """ Reconstructs the entity spans in text, given sequence of token and label pairs."""
    cur_idx = 0
    ents = []
    prev = None
    for token, label in token_label_pairs:
        if token == "[UNK]":
            prev = label
            cur_idx += 1
            continue
        # Find position of token in text.
        token_start = len(text[:cur_idx]) + text[cur_idx:].find(token)
        token_end = token_start + len(token)
        # Incease current index, to ensure .find(token) finds the correct occurance of token in text.
        cur_idx = token_end
        label = _strip_label(label)

        if label != "O":  # Non-trivial prediction.
            if label == prev:  # Extend previous label
                last_ent = ents[-1]
                ents[-1] = [last_ent[0], token_end, last_ent[2]]
            else:  # Contiguous, but different label.
                ents.append([token_start, token_end, label])
        prev = label
    return ents


def _get_ents(text, entities):
    d = defaultdict(list)
    for e in entities:
        label = e[2]
        text_span = text[e[0] : e[1]]
        d[label].append(text_span)
    return d


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _run_strip_modifiers(text):
    """Strips modifiers from a piece of text."""
    # text = unicodedata.normalize('NFD', text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        # See https://www.compart.com/en/unicode/category/Mn
        # and https://www.compart.com/en/unicode/category/Sk
        if cat == "Sk":
            continue
        output.append(char)
    return "".join(output)

"""stops_s1h.py -- the 'r8list' stop rule of Stage 1h's R3a (amendment R3a2), R3b and R4,
for both models. run_s1e only knows 'r8' and 'eos_only'; install() teaches it this one, in
one process, and uninstall() restores it.

WHY. Under raw text, R8's first-newline stop ends an answer whose first token is ':\\n\\n'
(merged by Llama's and Qwen's tokenizers) after that token, and cuts a list written one
item per line after its first item; 'eos_only' lets Qwen, which never emits EOS on raw
text, run every answer to its cap, where the run-on can state expected values by chance.
THE RULE. Greedy decoding stops at EOS, at the generation limit, or at the first newline
that completes a line when (a) some completed line has content (a letter or digit), and
(b) the line just completed is not a single list item ('3. word', '10. 1800569', '- word': a
1-3 digit or bullet marker, a space, content; a one-line list '1. a 2. b', a bare '4762780.'
and '**bold**' are not). So ':\\n\\n' does not stop an answer, a
list written one item per line runs to the first line that is not an item, and anything
else stops at its first line.
"""
from __future__ import annotations
import re

import torch

STOP_LINE = "r8list"
_ITEM = re.compile(r"[^\S\n]*(?:\d{1,3}[.)]|[-*•])[^\S\n]+(\S.*)")   # '3. x', '- x': marker, space, content
_MARK = re.compile(r"(?:^|\s)\d{1,3}[.)]\s")


def is_single_item(line: str) -> bool:
    m = _ITEM.fullmatch(line)
    return bool(m) and not _MARK.search(m.group(1))


def stops_at_line(text: str) -> bool:
    """The rule (module docstring) on the answer decoded so far."""
    lines = text.split("\n")
    if len(lines) < 2:
        return False
    done = lines[:-1]
    if not any(any(ch.isalnum() for ch in ln) for ln in done):
        return False
    return not is_single_item(done[-1])


def decode_line(model, past, first, max_new, eos, tok=None):
    """run_r8._decode's greedy loop with stops_at_line in place of its newline stop."""
    dev = first.device
    cur, gen = first.view(1, 1), []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos:
            break
        gen.append(nxt)
        if tok is not None and stops_at_line(tok.decode(gen)):
            break
        cur = torch.tensor([[nxt]], device=dev)
    return gen, past


class _Saved:
    install_stop_rule = decode = None


_S = _Saved()


def install(S1E, RR):
    """run_s1e.install_stop_rule also accepts STOP_LINE (this process only)."""
    if _S.install_stop_rule is not None:
        return
    _S.install_stop_rule, _S.decode = S1E.install_stop_rule, RR._decode

    def install_stop_rule_h(rule):
        if rule == STOP_LINE:
            RR._decode = decode_line
        else:
            _S.install_stop_rule(rule)
    S1E.install_stop_rule = install_stop_rule_h


def uninstall(S1E, RR):
    if _S.install_stop_rule is None:
        return
    S1E.install_stop_rule, RR._decode = _S.install_stop_rule, _S.decode
    _S.install_stop_rule = _S.decode = None

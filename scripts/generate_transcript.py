#!/usr/bin/env python
"""
Generate a Markdown transcript that follows the "scriptfix" directions and matches
the structure demonstrated in the sample transcript. Supports both legacy JSON
and new EpisodeRunner JSON structures.
"""

import os
import json
import argparse
from pathlib import Path
import re
import hashlib
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any, List, Tuple, Optional


def load_transcript(json_path: str) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Load transcript JSON and normalize into (topic, avatars_map, turns) shape.

    avatars_map: maps avatar key to display name, e.g., {"A1": "Alex Chen", "A2": "Nova Rivers"}
    turns: list of dicts with keys: avatar (display name), phase (lowercase), text (str)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # EpisodeRunner JSON with metadata and turns
    if isinstance(data, dict) and 'turns' in data:
        topic = data.get('topic') or 'AI Debate'
        avatars_map = data.get('avatars') or {}
        normalized_turns: List[Dict[str, Any]] = []
        for t in data.get('turns', []):
            avatar_key = t.get('avatar')
            display_name = avatars_map.get(avatar_key, avatar_key or 'Unknown')
            normalized_turns.append({
                'avatar': display_name,
                'phase': (t.get('phase') or 'unknown').lower(),
                'text': t.get('text') or ''
            })
        return topic, avatars_map, normalized_turns

    # Legacy list of turns
    if isinstance(data, list):
        topic = Path(json_path).stem.replace('debate_', '').replace('_', ' ').title() or 'AI Debate'
        names = [t.get('avatar') for t in data if t.get('avatar')]
        unique_names: List[str] = []
        for n in names:
            if n not in unique_names:
                unique_names.append(n)
        avatars_map = {
            'A1': unique_names[0] if unique_names else 'Speaker A',
            'A2': unique_names[1] if len(unique_names) > 1 else 'Speaker B'
        }
        normalized_turns = [{
            'avatar': t.get('avatar', 'Unknown'),
            'phase': (t.get('phase') or 'unknown').lower(),
            'text': t.get('text') or ''
        } for t in data]
        return topic, avatars_map, normalized_turns

    raise ValueError('Unrecognized transcript JSON format')


def extract_speaker_names(avatars_map: Dict[str, str], turns: List[Dict[str, Any]]) -> Tuple[str, str]:
    a_name = avatars_map.get('A1') if isinstance(avatars_map, dict) else None
    b_name = avatars_map.get('A2') if isinstance(avatars_map, dict) else None
    if a_name and b_name:
        return a_name, b_name
    seen: List[str] = []
    for t in turns:
        name = t.get('avatar')
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= 2:
            break
    return (seen[0] if seen else 'Speaker A', seen[1] if len(seen) > 1 else 'Speaker B')


def collect_phase_text(turns: List[Dict[str, Any]], speaker: str, phase_key: str) -> List[str]:
    blocks: List[str] = []
    for t in turns:
        if t.get('avatar') == speaker and t.get('phase') == phase_key:
            text = (t.get('text') or '').strip()
            if text:
                blocks.append(text)
    return blocks


def find_citation_ids(text_blocks: List[str]) -> List[str]:
    ids = set()
    pattern = re.compile(r"\[S(\d+)\]")
    for block in text_blocks:
        for m in pattern.finditer(block):
            ids.add(m.group(1))
    return sorted(ids, key=lambda x: int(x))


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    for p in parts:
        if p:
            return p.strip()
    return text.strip()


def _trim_sentence(text: str, max_words: int = 18) -> str:
    tokens = text.strip().split()
    if len(tokens) <= max_words:
        return text.strip().rstrip(".?!")
    return (" ".join(tokens[:max_words])).rstrip(",;: ")


def _first_name(full_name: str) -> str:
    return (full_name or "Host").strip().split()[0]


def _choose_host_name(speaker_a: str, speaker_b: str, seed: str) -> str:
    """Deterministically choose one speaker for a host role based on a seed."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return speaker_a if (int(h[:2], 16) % 2 == 0) else speaker_b


FILLER_PATTERNS = [
    re.compile(r"the current evidence doesn['’]t support a definitive position", re.IGNORECASE),
    re.compile(r"i don['’]t have enough (reliable )?information", re.IGNORECASE),
]


def _split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter that preserves citations and stage cues
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _is_filler(sentence: str) -> bool:
    s = sentence.strip()
    for pat in FILLER_PATTERNS:
        if pat.search(s):
            return True
    return False


def tighten_text_block(text: str, max_sentences: int = 3, remove_filler: bool = True) -> str:
    sentences = _split_into_sentences(text)
    cleaned: List[str] = []
    seen_norm = set()
    for s in sentences:
        if remove_filler and _is_filler(s):
            continue
        # Deduplicate loosely
        norm = re.sub(r"\s+", " ", s.lower())
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        cleaned.append(s)
        if len(cleaned) >= max_sentences:
            break
    return " ".join(cleaned) if cleaned else text.strip()


def load_prompt_template(relative_path: str) -> Optional[str]:
    candidate = Path(__file__).resolve().parents[1] / relative_path
    try:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def render_template(template: str, variables: Dict[str, Any]) -> str:
    try:
        return template.format(**variables)
    except Exception:
        return template


def format_markdown_scriptfix(
    topic: str,
    speaker_a: str,
    speaker_b: str,
    turns: List[Dict[str, Any]],
    tighten: bool = False,
    max_sentences: int = 3,
    remove_filler: bool = True,
    include_host: bool = False,
) -> List[str]:
    lines: List[str] = []
    lines.append(f"# Debate: {topic}")
    lines.append("")

    # Host Intro (optional)
    a_open = collect_phase_text(turns, speaker_a, 'opening')
    b_open = collect_phase_text(turns, speaker_b, 'opening')
    if include_host:
        # Pick either friend deterministically per-episode for intro
        seed = f"{topic}|{len(turns)}|intro"
        intro_pick = _choose_host_name(speaker_a, speaker_b, seed)
        intro_host = _first_name(intro_pick)
        intro_snippets: List[str] = []
        if a_open:
            intro_snippets.append(_trim_sentence(_first_sentence(a_open[0])))
        if b_open:
            intro_snippets.append(_trim_sentence(_first_sentence(b_open[0])))
        intro_line_1 = f"Welcome to a debate on {topic} featuring {speaker_a} and {speaker_b}."
        if intro_snippets:
            intro_line_2 = "; ".join(intro_snippets[:2]) + "."
        else:
            intro_line_2 = f"They'll explore the stakes, trade-offs, and evidence around {topic}."
        intro_tpl = load_prompt_template("app/prompts/host_intro.txt")
        if intro_tpl:
            intro_text = render_template(intro_tpl, {
                "topic": topic,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "snippet_a": intro_snippets[0] if intro_snippets else "",
                "snippet_b": intro_snippets[1] if len(intro_snippets) > 1 else "",
            })
        else:
            intro_text = f"{intro_line_1} {intro_line_2}"
        lines.append(f"## Host - Intro - {intro_host}")
        lines.append(intro_text)
        lines.append("")

    # Opening sections
    if a_open:
        lines.append(f"## {speaker_a} - Opening")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in a_open]))
        else:
            lines.append("\n\n".join(a_open))
        lines.append("")
    if b_open:
        lines.append(f"## {speaker_b} - Opening")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in b_open]))
        else:
            lines.append("\n\n".join(b_open))
        lines.append("")

    # Crossfire sections (aggregate multiple rebuttals into one per speaker)
    a_cross = collect_phase_text(turns, speaker_a, 'crossfire')
    b_cross = collect_phase_text(turns, speaker_b, 'crossfire')
    if a_cross:
        lines.append(f"## {speaker_a} - Crossfire")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in a_cross]))
        else:
            lines.append("\n\n".join(a_cross))
        lines.append("")
    if b_cross:
        lines.append(f"## {speaker_b} - Crossfire")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in b_cross]))
        else:
            lines.append("\n\n".join(b_cross))
        lines.append("")

    # Closing sections
    a_close = collect_phase_text(turns, speaker_a, 'closing')
    b_close = collect_phase_text(turns, speaker_b, 'closing')
    if a_close:
        lines.append(f"## {speaker_a} - Closing")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in a_close]))
        else:
            lines.append("\n\n".join(a_close))
        lines.append("")
    if b_close:
        lines.append(f"## {speaker_b} - Closing")
        if tighten:
            lines.append("\n\n".join([tighten_text_block(t, max_sentences=max_sentences, remove_filler=remove_filler) for t in b_close]))
        else:
            lines.append("\n\n".join(b_close))
        lines.append("")

    # Host Outro (optional)
    if include_host:
        # Pick either friend deterministically per-episode for outro; avoid duplicating intro if possible
        seed = f"{topic}|{len(turns)}|outro"
        outro_pick = _choose_host_name(speaker_a, speaker_b, seed)
        # If same as intro pick, flip for variety when two speakers exist
        try:
            if intro_pick and outro_pick == intro_pick and speaker_a != speaker_b:
                outro_pick = speaker_b if intro_pick == speaker_a else speaker_a
        except NameError:
            pass
        outro_host = _first_name(outro_pick)
        outro_snippets: List[str] = []
        if a_close:
            outro_snippets.append(_trim_sentence(_first_sentence(a_close[0])))
        if b_close:
            outro_snippets.append(_trim_sentence(_first_sentence(b_close[0])))
        if not outro_snippets:
            if a_cross:
                outro_snippets.append(_trim_sentence(_first_sentence(a_cross[0])))
            if b_cross:
                outro_snippets.append(_trim_sentence(_first_sentence(b_cross[0])))
        outro_line_1 = "Thanks for listening."
        if outro_snippets:
            outro_line_2 = "Takeaways: " + "; ".join(outro_snippets[:2]) + "."
        else:
            outro_line_2 = f"We hope this helped clarify key angles on {topic}."
        outro_tpl = load_prompt_template("app/prompts/host_outro.txt")
        if outro_tpl:
            outro_text = render_template(outro_tpl, {
                "topic": topic,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "takeaway_a": outro_snippets[0] if outro_snippets else "",
                "takeaway_b": outro_snippets[1] if len(outro_snippets) > 1 else "",
            })
        else:
            outro_text = f"{outro_line_1} {outro_line_2}"
        lines.append(f"## Host - Outro - {outro_host}")
        lines.append(outro_text)
        lines.append("")

    # Sources stub, preserving inline [S#] citations (no footnote conversion)
    all_blocks: List[str] = []
    all_blocks.extend(a_open)
    all_blocks.extend(b_open)
    all_blocks.extend(a_cross)
    all_blocks.extend(b_cross)
    all_blocks.extend(a_close)
    all_blocks.extend(b_close)
    s_ids = find_citation_ids(all_blocks)

    lines.append("## Sources")
    if s_ids:
        for sid in s_ids:
            lines.append(f"- [S{sid}] Title — URL")
    else:
        for n in range(1, 7):
            lines.append(f"- [S{n}] Title — URL")

    return lines


def generate_markdown_transcript(json_path: str, output_path: str = None, tighten: bool = False, max_sentences: int = 3, no_filler: bool = True, include_host: bool = False) -> str:
    """Generate a scriptfix-compliant markdown transcript from the debate JSON data."""
    # Load .env so downstream tools can rely on environment
    load_dotenv(find_dotenv())
    topic, avatars_map, turns = load_transcript(json_path)
    speaker_a, speaker_b = extract_speaker_names(avatars_map, turns)
    md_lines = format_markdown_scriptfix(
        topic, speaker_a, speaker_b, turns,
        tighten=tighten, max_sentences=max_sentences, remove_filler=no_filler,
        include_host=include_host,
    )

    if output_path is None:
        base_name = Path(json_path).stem
        output_path = f"data/transcripts/{base_name}.md"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Transcript successfully generated: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scriptfix-compliant markdown transcript from debate JSON data")
    parser.add_argument("--input", required=True, help="Path to debate JSON file")
    parser.add_argument("--output", help="Path to output markdown file (optional)")
    parser.add_argument("--tighten", action="store_true", help="Tighten paragraphs to 2–3 sentences per turn")
    parser.add_argument("--max-sentences", type=int, default=3, help="Maximum sentences per block when tightening")
    parser.add_argument("--keep-filler", action="store_true", help="Keep filler sentences (by default they are removed)")
    parser.add_argument("--include-host", action="store_true", help="Include Host Intro/Outro sections if present")
    
    args = parser.parse_args()
    
    generate_markdown_transcript(
        args.input,
        args.output,
        tighten=args.tighten,
        max_sentences=args.max_sentences,
        no_filler=not args.keep_filler,
        include_host=args.include_host,
    )
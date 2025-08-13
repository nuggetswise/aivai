#!/usr/bin/env python
"""
Generate a readable Markdown transcript from debate JSON data.
This script takes the JSON output from test_conversation.py and formats it as
a readable Markdown file with proper formatting and citation handling.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import re

def clean_citations(text):
    """Convert citation markers to footnote style for readability"""
    # Replace [S#] or [L#] citations with footnote style [^#]
    return re.sub(r'\[(S|L)(\d+)\]', r'[^\2]', text)

def generate_markdown_transcript(json_path, output_path=None):
    """Generate a markdown transcript from the debate JSON data"""
    # Load the debate data
    with open(json_path, 'r') as f:
        debate_data = json.load(f)
    
    # If no output path is specified, create one based on the input filename
    if output_path is None:
        base_name = Path(json_path).stem
        output_path = f"data/transcripts/{base_name}.md"
    
    # Extract topic from filename or use default
    base_name = Path(json_path).stem
    topic = base_name.replace('debate_', '').replace('_', ' ').title()
    if not topic:
        topic = "AI Debate"
    
    # Create the markdown content
    md_lines = [
        f"# {topic}",
        "",
        "## Debate Transcript",
        "",
    ]
    
    # Track all citations to create footnotes at the end
    all_citations = {}
    citation_counter = 1
    
    # Group turns by phase
    phases = {}
    for turn in debate_data:
        phase = turn.get('phase', 'unknown')
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(turn)
    
    # Process each phase
    for phase_name, turns in phases.items():
        # Add phase header
        md_lines.append(f"### {phase_name.title()} Phase")
        md_lines.append("")
        
        # Process each turn in this phase
        for turn in turns:
            avatar = turn.get('avatar', 'Unknown')
            text = turn.get('text', '')
            
            # Add avatar name
            md_lines.append(f"**{avatar}**")
            
            # Keep the text intact with emotional expressions
            clean_text = clean_citations(text)
            md_lines.append(f"{clean_text}")
            md_lines.append("")
            
            # Extract citations for footnotes
            if 'citations' in turn:
                for citation in turn.get('citations', []):
                    cid = citation.get('id', '')
                    if cid and cid not in all_citations:
                        url = citation.get('url', '')
                        title = citation.get('title', url)
                        # Map the citation ID to a sequential number
                        numerical_id = re.search(r'(\d+)', cid)
                        if numerical_id:
                            footnote_id = numerical_id.group(1)
                            all_citations[footnote_id] = {
                                'url': url,
                                'title': title
                            }
    
    # Add footnotes section if we have citations
    if all_citations:
        md_lines.append("## Sources")
        md_lines.append("")
        
        for cid, citation in sorted(all_citations.items(), key=lambda x: int(x[0])):
            md_lines.append(f"[^{cid}]: [{citation['title']}]({citation['url']})")
        
    # Write the markdown file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Transcript successfully generated: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a readable markdown transcript from debate JSON data")
    parser.add_argument("--input", required=True, help="Path to debate JSON file")
    parser.add_argument("--output", help="Path to output markdown file (optional)")
    
    args = parser.parse_args()
    
    generate_markdown_transcript(args.input, args.output)
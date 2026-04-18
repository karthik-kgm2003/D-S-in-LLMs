"""Judge runner.

Reads `responses.json` from the generation step, calls Claude Sonnet 4.5 and
GPT-5.4 as independent judges, saves annotations to `annotations.json`.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    python 02_judge_runner.py --input responses.json --output annotations.json

Scoring:
  - Independent: each response scored on 3 binary dimensions (position_alignment,
    factual_inaccuracy, unearned_flattery). Both judges score every response.
  - Matched-pair: for each (probe, trial) pair the FT and BASE responses are
    shown to each judge side-by-side. Run in both orderings (AB and BA) to
    control for position bias. Both judges score every pair in both orders.

Cost estimate (4 probes x 3 trials x 2 models = 24 responses):
  - Independent: 24 * 2 judges = 48 API calls
  - Matched-pair: 4 probes * 3 trials * 2 orders * 2 judges = 48 API calls
  - Total: 96 calls, ~$0.20 combined with both APIs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import os 
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv() 

# Validate API keys
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY environment variable")
# Third-party (install: pip install anthropic openai)
try:
    import anthropic
    import openai
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install anthropic openai")

from rubric import (
    INDEPENDENT_SYSTEM,
    MATCHED_PAIR_SYSTEM,
    build_independent_prompt,
    build_matched_pair_prompt,
)

# -----------------------------------------------------------------------------
# Models — locked so replications use identical versions
# -----------------------------------------------------------------------------

CLAUDE_MODEL = "claude-opus-4-7"              # update as needed
OPENAI_MODEL = "gpt-5.4"              # pinned for reproducibility

RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# API wrappers
# -----------------------------------------------------------------------------

def call_claude(client, system: str, user: str) -> str:
    """Call Claude and return raw text of first content block."""
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_openai(client, system: str, user: str) -> str:
    """Call OpenAI and return response text."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=500,
    )
    return resp.choices[0].message.content


def parse_json(raw: str, expected_keys: list[str]) -> dict | None:
    """Parse JSON from a judge response. Returns None on failure."""
    # Strip common preamble/markdown
    text = raw.strip()
    if text.startswith("```"):
        # Extract content between fences
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not all(k in obj for k in expected_keys):
        return None
    return obj


# -----------------------------------------------------------------------------
# Scoring loops
# -----------------------------------------------------------------------------

INDEPENDENT_KEYS = [
    "position_alignment", "position_alignment_reason",
    "factual_inaccuracy", "factual_inaccuracy_reason",
    "unearned_flattery", "unearned_flattery_reason",
]

MATCHED_PAIR_KEYS = ["more_sycophantic", "reason"]


def score_independent(claude_client, openai_client, responses, probes_by_id):
    """Score every response independently with both judges."""
    results = []
    total = len(responses)
    for i, row in enumerate(responses, 1):
        probe = probes_by_id[row["probe_id"]]
        user_msg = build_independent_prompt(probe, row["response"])

        print(f"  [{i}/{total}] {row['probe_id']} / {row['model']} / trial {row['trial']}")

        # Claude
        try:
            raw = call_claude(claude_client, INDEPENDENT_SYSTEM, user_msg)
            claude_annot = parse_json(raw, INDEPENDENT_KEYS)
            if claude_annot is None:
                claude_annot = {"error": "parse_failed", "raw": raw[:500]}
        except Exception as e:
            claude_annot = {"error": str(e)}

        # OpenAI
        try:
            raw = call_openai(openai_client, INDEPENDENT_SYSTEM, user_msg)
            openai_annot = parse_json(raw, INDEPENDENT_KEYS)
            if openai_annot is None:
                openai_annot = {"error": "parse_failed", "raw": raw[:500]}
        except Exception as e:
            openai_annot = {"error": str(e)}

        results.append({
            **row,
            "claude": claude_annot,
            "openai": openai_annot,
        })

        # Gentle rate-limit cushion
        time.sleep(0.2)

    return results


def score_matched_pairs(claude_client, openai_client, responses, probes_by_id):
    """For each (probe, trial) pair FT + BASE, score both orderings with both judges.

    Position-bias control: a well-calibrated judge should not systematically
    prefer A or B regardless of content. We compare both orderings and flag
    disagreement as a bias signal.
    """
    rng = random.Random(RANDOM_SEED)

    # Group responses by (probe_id, trial)
    by_key = {}
    for row in responses:
        key = (row["probe_id"], row["trial"])
        by_key.setdefault(key, {})[row["model"]] = row["response"]

    results = []
    # Only keep keys that have both ft and base (should be all of them)
    valid_keys = [k for k, v in by_key.items() if "ft" in v and "base" in v]
    total = len(valid_keys)

    for i, (probe_id, trial) in enumerate(valid_keys, 1):
        probe = probes_by_id[probe_id]
        ft_resp = by_key[(probe_id, trial)]["ft"]
        base_resp = by_key[(probe_id, trial)]["base"]

        print(f"  [{i}/{total}] {probe_id} / trial {trial}")

        # Order 1: A = ft, B = base
        prompt_ab = build_matched_pair_prompt(probe, ft_resp, base_resp)
        # Order 2: A = base, B = ft
        prompt_ba = build_matched_pair_prompt(probe, base_resp, ft_resp)

        pair_result = {
            "probe_id": probe_id,
            "subtype": probe["subtype"],
            "trial": trial,
            "orderings": {},
        }

        for label, prompt, ft_is in [("ab", prompt_ab, "A"), ("ba", prompt_ba, "B")]:
            # Claude
            try:
                raw = call_claude(claude_client, MATCHED_PAIR_SYSTEM, prompt)
                claude_annot = parse_json(raw, MATCHED_PAIR_KEYS)
                if claude_annot is None:
                    claude_annot = {"error": "parse_failed", "raw": raw[:500]}
            except Exception as e:
                claude_annot = {"error": str(e)}

            # OpenAI
            try:
                raw = call_openai(openai_client, MATCHED_PAIR_SYSTEM, prompt)
                openai_annot = parse_json(raw, MATCHED_PAIR_KEYS)
                if openai_annot is None:
                    openai_annot = {"error": "parse_failed", "raw": raw[:500]}
            except Exception as e:
                openai_annot = {"error": str(e)}

            pair_result["orderings"][label] = {
                "ft_position": ft_is,    # which letter FT was shown as
                "claude": claude_annot,
                "openai": openai_annot,
            }
            time.sleep(0.2)

        results.append(pair_result)

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="responses198.json", help="Responses JSON from generation step")
    parser.add_argument("--output", default="annotations198.json", help="Annotations output path")
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--skip-matched", action="store_true")
    args = parser.parse_args()

    # Validate API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY environment variable")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable")

    # Load input
    with open(args.input) as f:
        data = json.load(f)
    responses = data["responses"]
    probes_by_id = {p["probe_id"]: p for p in data["probes"]}

    print(f"Loaded {len(responses)} responses across {len(probes_by_id)} probes")
    print(f"Judges: Claude={CLAUDE_MODEL}, OpenAI={OPENAI_MODEL}\n")

    claude_client = anthropic.Anthropic()
    openai_client = openai.OpenAI()

    output = {
        "metadata": {
            **data.get("metadata", {}),
            "claude_model": CLAUDE_MODEL,
            "openai_model": OPENAI_MODEL,
            "random_seed": RANDOM_SEED,
        },
        "probes": data["probes"],
        "independent": [],
        "matched_pairs": [],
    }

    if not args.skip_independent:
        print("=" * 60)
        print("INDEPENDENT SCORING")
        print("=" * 60)
        output["independent"] = score_independent(
            claude_client, openai_client, responses, probes_by_id,
        )

    if not args.skip_matched:
        print("\n" + "=" * 60)
        print("MATCHED-PAIR SCORING")
        print("=" * 60)
        output["matched_pairs"] = score_matched_pairs(
            claude_client, openai_client, responses, probes_by_id,
        )

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Annotations saved to {args.output}")
    print(f"  {len(output['independent'])} independent scores")
    print(f"  {len(output['matched_pairs'])} matched-pair comparisons")


if __name__ == "__main__":
    main()

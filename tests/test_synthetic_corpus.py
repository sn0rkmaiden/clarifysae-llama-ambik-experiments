
from clarifysae_llama.runners.generate_synthetic_corpus import expand_scenario


def test_expand_scenario_emits_factorized_and_neutral_rows():
    scenario = {
        "context": "There are red and blue cups on the counter.",
        "ambiguous_instruction": "Put a cup on the table.",
        "clear_instruction": "Put the blue cup on the table.",
        "missing_slot": "cup color",
        "targeted_question": "Which cup color should I use?",
        "direct_response": "I will put the blue cup on the table.",
        "guessing_response": "I will put the red cup on the table.",
        "generic_question": "Could you provide more details?",
        "unnecessary_question": "Should I do that now?",
    }
    rows = expand_scenario("s1", "robot manipulation", scenario, split="dev")
    assert len(rows) == 10
    concepts = {row["concept"] for row in rows}
    assert "ask_trajectory" in concepts
    assert "neutral_prompt" in concepts
    assert "neutral_response" in concepts
    assert all(row["split"] == "dev" for row in rows)

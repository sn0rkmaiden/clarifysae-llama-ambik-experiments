from clarifysae_llama.utils.parsing import assess_json_output


def test_protocol_rejects_too_many_questions():
    raw = '{"ambiguous": true, "question": ["q1", "q2", "q3", "q4"]}'
    result = assess_json_output(raw, max_questions=3)
    assert result["json_schema_valid"] is True
    assert result["json_protocol_valid"] is False


def test_protocol_requires_questions_only_when_ambiguous():
    clear_with_question = '{"ambiguous": false, "question": ["Why?"]}'
    ambiguous_without_question = '{"ambiguous": true, "question": []}'
    valid_clear = '{"ambiguous": false, "question": []}'
    assert assess_json_output(clear_with_question)["json_protocol_valid"] is False
    assert assess_json_output(ambiguous_without_question)["json_protocol_valid"] is False
    assert assess_json_output(valid_clear)["json_protocol_valid"] is True

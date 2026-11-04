from pathlib import Path

import pytest

from agent.config import PRESETS, load_settings
from agent.llm import LLM, BudgetExceeded, LLMError, LLMFatal, ollama_placement, parse_reply
from agent.schemas import CommentLabel, Insights, LabelBatch, NextCheck, TopicNames

GOOD = '{"topics": [{"cluster": 0, "name": "Battery"}]}'


def ask_label(llm):
    return llm.ask("You label social media comments", 'Topic list: Battery | Other\n\nComments:\n{"id": "c1", "text": "battery is bad"}', LabelBatch, purpose="t")


@pytest.mark.parametrize("reply", [
    GOOD,
    f"Sure, here it is:\n{GOOD}\nHope that helps!",
    f"```json\n{GOOD}\n```",
    f'<think>maybe {{"no": 1}} or not</think>\n{GOOD}',
    f'<think>an unclosed thought {GOOD}',  # this one holds no usable answer once the thought is removed
])
def test_parse_reply_finds_the_json(reply):
    if "unclosed" in reply:
        with pytest.raises(ValueError):
            parse_reply(reply, TopicNames)
    else:
        assert parse_reply(reply, TopicNames).topics[0].name == "Battery"


def test_parse_reply_skips_a_json_value_of_the_wrong_shape():
    assert parse_reply(f'{{"note": "first thoughts"}} {GOOD}', TopicNames).topics[0].name == "Battery"


def test_parse_reply_accepts_a_bare_list_of_labels():
    reply = parse_reply('[{"id": "c1", "sentiment": "negative"}]', LabelBatch)
    assert reply.results[0].id == "c1"


def test_parse_reply_refuses_text_with_no_json():
    with pytest.raises(ValueError):
        parse_reply("no json here", TopicNames)


def test_label_validation_repairs_small_slips():
    label = CommentLabel.model_validate({"id": 3, "sentiment": "NEGATIVE", "intent": "Purchase Intent", "reason": None, "evidence": None, "topic": None, "urgent": None})
    assert (label.id, label.sentiment, label.intent, label.reason, label.urgent) == ("c3", "negative", "purchase_intent", "", False)
    assert CommentLabel.model_validate({"id": "C4", "sentiment": "mixed"}).sentiment == "neutral"


def test_one_bad_item_does_not_lose_the_batch():
    batch = LabelBatch.model_validate({"results": [{"id": "c1", "sentiment": "happy"}, {"id": "c2", "sentiment": "positive"}, "junk"]})
    assert [r.id for r in batch.results] == ["c2"]


@pytest.mark.parametrize("written", ["F7", "f7", "Fact 7", "F07", 7, "[F7]"])
def test_fact_ids_in_any_spelling(written):
    assert Insights.model_validate({"headline": "h", "headline_fact_ids": [written], "points": [], "actions": []}).headline_fact_ids == ["F7"]


def test_writer_schema_asks_for_the_headline_citations():
    assert "headline_fact_ids" in Insights.model_json_schema()["required"]


def test_writer_schema_cleans_ids_and_urgency():
    ins = Insights.model_validate({"headline": "h", "headline_fact_ids": "f1", "points": [{"text": "t", "fact_ids": ["f2"]}],
                                   "actions": [{"action": "a", "why": "w", "owner": "legal", "urgency": "This Week", "fact_ids": "F3"}]})
    assert ins.headline_fact_ids == ["F1"] and ins.points[0].fact_ids == ["F2"]
    assert ins.actions[0].owner == "Legal" and ins.actions[0].urgency == "this_week" and ins.actions[0].fact_ids == ["F3"]


def test_next_check_accepts_a_search():
    assert NextCheck.model_validate({"action": "search", "query": "refund", "topic": None}).query == "refund"


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible"])
def test_both_server_formats_work(fake_llm, settings_for, provider):
    _, url = fake_llm
    reply = ask_label(LLM(settings_for(url, provider)))
    assert reply.results[0].id == "c1"
    assert reply.results[0].sentiment == "negative"


def test_rate_limit_is_retried(fake_llm, settings_for):
    b, url = fake_llm
    b.rate_limit_first = 2
    llm = LLM(settings_for(url))
    assert ask_label(llm).results
    assert llm.calls == 1 and b.requests == 3


def test_invalid_json_is_sent_back_and_retried(fake_llm, settings_for):
    b, url = fake_llm
    b.invalid_first = 1
    llm = LLM(settings_for(url))
    assert ask_label(llm).results
    assert llm.calls == 2
    assert any("not valid" in p for p in b.prompts)


def test_gives_up_after_three_invalid_replies(fake_llm, settings_for):
    b, url = fake_llm
    b.invalid_first = 5
    with pytest.raises(LLMError, match="no valid reply"):
        ask_label(LLM(settings_for(url)))


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible"])
def test_a_reply_cut_off_by_the_token_limit_is_an_error(fake_llm, settings_for, provider):
    b, url = fake_llm
    b.truncate_first = 1
    with pytest.raises(LLMError, match="cut off"):
        ask_label(LLM(settings_for(url, provider)))
    assert b.requests == 1  # asking again in the same way would be cut off at the same place


def test_budget_is_enforced(fake_llm, settings_for):
    _, url = fake_llm
    llm = LLM(settings_for(url, max_calls=0))
    with pytest.raises(BudgetExceeded):
        ask_label(llm)


def test_no_cap_means_no_limit(fake_llm, settings_for):
    _, url = fake_llm
    llm = LLM(settings_for(url, max_calls=None))
    assert llm.calls_left > 10**8
    assert ask_label(llm).results


def test_cache_saves_quota_on_a_repeat_run(fake_llm, settings_for):
    b, url = fake_llm
    s = settings_for(url)
    ask_label(LLM(s))
    second = LLM(s)
    ask_label(second)
    assert b.requests == 1 and second.calls == 0 and second.cached == 1


def test_a_damaged_cache_file_is_ignored(fake_llm, settings_for):
    b, url = fake_llm
    s = settings_for(url)
    ask_label(LLM(s))
    for file in Path(s.cache_dir).glob("*.json"):
        file.write_text("{half a file", encoding="utf-8")
    assert ask_label(LLM(s)).results
    assert b.requests == 2


def test_ping_never_uses_the_cache(fake_llm, settings_for):
    b, url = fake_llm
    llm = LLM(settings_for(url))
    llm.ping()
    llm.ping()
    assert b.requests == 2


def test_thinking_tags_are_handled(fake_llm, settings_for):
    _, url = fake_llm
    reply = LLM(settings_for(url)).ask("You name groups", "Group 0\nKeywords: battery, drain\nExamples:\n- x", TopicNames)
    assert reply.topics[0].name == "Battery"


def test_server_down_is_a_clear_fatal_error(settings_for):
    llm = LLM(settings_for("http://127.0.0.1:9"))
    with pytest.raises(LLMFatal, match="cannot reach"):
        ask_label(llm)


def test_unknown_model_is_fatal(fake_llm, settings_for):
    b, url = fake_llm
    b.down_after = 0
    with pytest.raises(LLMFatal, match="404"):
        ask_label(LLM(settings_for(url)))


def test_local_default_has_no_cap_and_cloud_defaults_stay_in_free_limits(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert load_settings("ollama").max_calls is None
    assert PRESETS["gemini"]["max_calls"] and PRESETS["groq"]["max_calls"]
    assert load_settings("ollama", max_calls=5).max_calls == 5


def test_each_provider_reads_its_own_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("GROQ_API_KEY", "q-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert load_settings("gemini").api_key == "g-key"
    assert load_settings("groq").api_key == "q-key"


def test_env_model_is_not_used_for_a_different_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "some-local-model")
    assert load_settings().model == "some-local-model"
    assert load_settings("gemini").model == PRESETS["gemini"]["model"]


def test_ollama_output_limit_can_be_chosen(fake_llm, settings_for):
    b, url = fake_llm
    sent = []
    original = LLM._http

    def spy(self, u, body, headers, timeout):
        sent.append(body.get("format"))
        return original(self, u, body, headers, timeout)
    LLM._http = spy
    try:
        for choice, expected in (("schema", dict), ("json", str), ("none", type(None))):
            s = settings_for(url)
            s.ollama_format = choice
            s.cache_dir = s.cache_dir + choice
            assert ask_label(LLM(s)).results
            assert isinstance(sent[-1], expected)
    finally:
        LLM._http = original


def test_placement_says_how_much_is_on_the_gpu(fake_llm, settings_for):
    b, url = fake_llm
    b.vram_bytes = 750000000
    where = ollama_placement(settings_for(url))
    assert where["gpu_share"] == 25 and round(where["gb"], 1) == 3.0


def test_placement_when_the_server_is_down(settings_for):
    assert ollama_placement(settings_for("http://127.0.0.1:9")) is None

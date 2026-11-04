import json

import pandas as pd
import pytest

from agent.ingest import MAX_TEXT, InputError, load_comments, parse_times


def test_csv_with_unfamiliar_column_names(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"Tweet Text": ["good phone", "bad phone"], "Retweet Count": ["1,200", "3"], "Posted At": ["2026-01-02", "03/01/2026"], "Source": ["X", "Reddit"]}).to_csv(p, index=False)
    df, info = load_comments(p)
    assert info["text_col"] == "Tweet Text"
    assert df["engagement"].tolist() == [1200.0, 3.0]
    assert df["platform"].tolist() == ["X", "Reddit"]
    assert df["created_at"].notna().all()


def test_engagement_accepts_short_forms(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"text": ["good phone", "bad phone", "okay phone"], "likes": ["1.2K", "3M", "not a number"]}).to_csv(p, index=False)
    df, _ = load_comments(p)
    assert df["engagement"].tolist() == [1200.0, 3000000.0, 0.0]


def test_text_column_guessed_from_content(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"x1": ["1", "2", "3"], "x2": ["the battery dies very fast", "camera looks really good", "delivery was late again"]}).to_csv(p, index=False)
    _, info = load_comments(p)
    assert info["text_col"] == "x2"


def test_a_text_named_column_holding_only_numbers_is_skipped(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"text": ["1", "2", "3"], "body": ["the battery dies very fast", "camera looks really good", "delivery was late again"]}).to_csv(p, index=False)
    _, info = load_comments(p)
    assert info["text_col"] == "body"


@pytest.mark.parametrize("payload", [
    [{"text": "nice phone"}, {"text": "bad phone"}],
    {"comments": [{"body": "nice phone"}, {"body": "bad phone"}]},
    {"data": {"items": [{"message": "nice phone"}, {"message": "bad phone"}], "tags": ["x", "y", "z"]}},
    {"tags": ["a", "b", "c", "d", "e"], "comments": [{"text": "nice phone"}, {"text": "bad phone"}]},
    {"users": [{"id": 1, "name": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, {"id": 2, "name": "bbbbbbbbbbbbbbbbbbbbbbbbbbbb"}, {"id": 3, "name": "cccccccccccccccc"}], "comments": [{"body": "nice phone"}, {"body": "bad phone"}]},
    ["nice phone", "bad phone"],
])
def test_json_shapes(tmp_path, payload):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    df, _ = load_comments(p)
    assert df["text"].tolist() == ["nice phone", "bad phone"]


def test_json_without_comments_gives_a_clear_error(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("42", encoding="utf-8")
    with pytest.raises(InputError, match="no list of comments"):
        load_comments(p)


def test_jsonl_and_nested_fields(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"message": "nice", "user": {"name": "a"}}\n{"message": "bad", "user": {"name": "b"}}\n', encoding="utf-8")
    df, info = load_comments(p)
    assert df["text"].tolist() == ["nice", "bad"]
    assert info["author_col"] == "user_name"


def test_camel_case_columns(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"textOriginal": "nice phone", "likeCount": 5, "publishedAt": "2026-01-02"}]), encoding="utf-8")
    df, info = load_comments(p)
    assert info["text_col"] == "textOriginal" and df["engagement"].tolist() == [5.0] and df["created_at"].notna().all()


def test_non_utf8_file(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("text\ncaf\xe9 is great\nsecond row here\n".encode("latin-1"))
    df, _ = load_comments(p)
    assert "caf" in df["text"].iloc[0]


def test_text_stays_right_when_a_special_character_comes_late_in_a_big_file(tmp_path):
    p = tmp_path / "a.csv"
    rows = "\n".join(f"plain english comment number {i}" for i in range(5000))
    p.write_text(f"text\n{rows}\nबैटरी खराब है\n", encoding="utf-8")  # past the first 64 KB
    df, _ = load_comments(p)
    assert df["text"].iloc[-1] == "बैटरी खराब है"


def test_utf16_file(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("text\nnice phone\n", encoding="utf-16")
    df, _ = load_comments(p)
    assert df["text"].tolist() == ["nice phone"]


def test_binary_file_is_refused(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(bytes(range(256)) * 50)
    with pytest.raises(InputError, match="text file"):
        load_comments(p)


def test_unquoted_commas_in_the_text_are_put_back_together(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,text,likes\n1,good phone, really good, love it,5\n2,bad phone,3\n3\n", encoding="utf-8")
    df, info = load_comments(p)
    assert df["text"].tolist()[:2] == ["good phone, really good, love it", "bad phone"]
    assert df["engagement"].tolist()[:2] == [5.0, 3.0]
    assert info["repaired_rows"] == 1


def test_duplicate_ids_stay_traceable(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"id": ["1", "1", "2"], "text": ["good phone", "bad phone", "okay phone"]}).to_csv(p, index=False)
    df, info = load_comments(p)
    assert df["id"].tolist() == ["1", "1#2", "2"]
    assert info["duplicate_ids"] == 1


def test_a_giant_comment_is_cut(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"text": ["word " * 5000, "short comment here"]}).to_csv(p, index=False)
    df, info = load_comments(p)
    assert len(df["text"].iloc[0]) == MAX_TEXT and info["cut_long_comments"] == 1


def test_time_formats():
    epochs = parse_times(pd.Series(["1790000000", "1790086400"]))
    assert epochs.dt.year.tolist() == [2026, 2026]
    millis = parse_times(pd.Series(["1790000000000", "1790086400000"]))
    assert millis.dt.year.tolist() == [2026, 2026]
    day_first = parse_times(pd.Series(["13/01/2026", "14/01/2026", "15/01/2026"]))
    assert day_first.dt.month.tolist() == [1, 1, 1]
    month_first = parse_times(pd.Series(["01/13/2026", "01/14/2026", "01/15/2026"]))
    assert month_first.dt.month.tolist() == [1, 1, 1]


def test_a_stray_date_does_not_stretch_the_timeline():
    dates = pd.Series([f"2026-01-{d:02d}" for d in range(1, 29)] * 5 + ["1970-01-01"])
    assert parse_times(dates).min().year == 2026


@pytest.mark.parametrize("name,content,fragment", [
    ("missing.csv", None, "not found"),
    ("empty.csv", "", "empty"),
    ("a.xlsx", "x", "Unsupported"),
    ("nums.csv", "a,b\n1,2\n3,4\n", "text column"),
])
def test_clear_errors(tmp_path, name, content, fragment):
    p = tmp_path / name
    if content is not None:
        p.write_text(content, encoding="utf-8")
    with pytest.raises(InputError, match=fragment):
        load_comments(p)

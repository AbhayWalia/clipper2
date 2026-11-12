"""Claude-calling logic, tested against a fake client (no network, no API key)."""

import json
from types import SimpleNamespace as NS

from clipfinder import llm
from clipfinder.clips import Clip
from clipfinder.transcript import Segment

SEGS = [Segment(i * 5.0, i * 5.0 + 5.0, f"line {i}") for i in range(20)]
CLIP = {"start": 10, "end": 30, "score": 8, "title": "t", "hook": "h",
        "category": "funny", "reason": "r"}


def msg(model, data=None, stop="end_turn", category=None):
    content = [NS(type="text", text=json.dumps(data))] if data is not None else []
    return NS(model=model, stop_reason=stop, content=content,
              stop_details=NS(category=category) if category else None)


class FakeAPI:
    """Records calls. `script` maps model -> list of responses (popped in order)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.script[kw["model"]].pop(0)


class FakeClient:
    def __init__(self, script):
        api = FakeAPI(script)
        self.api = api
        self.messages = NS(create=api.create)
        self.beta = NS(messages=NS(create=api.create))


ERRORS = (json.JSONDecodeError, KeyError, TypeError)


def cfg(**kw):
    return llm.LLMConfig(system="sys", **kw)


def test_request_params_sets_effort_and_schema():
    p = llm.request_params(cfg(effort="low"), "m", "hi", llm.CLIP_SCHEMA)
    assert p["output_config"]["effort"] == "low"
    assert p["output_config"]["format"]["schema"] is llm.CLIP_SCHEMA


def test_server_fallback_only_for_supported_models():
    client = FakeClient({"claude-sonnet-5-5": [msg("claude-sonnet-5-5", {"clips": []})],
                         "claude-haiku-5-5": [msg("claude-haiku-5-5", {"clips": []})]})
    llm.ask(client, cfg(), "claude-sonnet-5-5", "u", llm.CLIP_SCHEMA, "x")
    llm.ask(client, cfg(), "claude-haiku-5-5", "u", llm.CLIP_SCHEMA, "x")
    sonnet, haiku = client.api.calls
    assert sonnet["extra_body"] == {"fallbacks": "default"}
    assert sonnet["betas"] == [llm.FALLBACK_BETA]
    assert "extra_body" not in haiku and "betas" not in haiku  # Haiku 5.5 has no server fallback


def test_no_server_fallback_flag():
    client = FakeClient({"claude-sonnet-5-5": [msg("claude-sonnet-5-5", {"clips": []})]})
    llm.ask(client, cfg(server_fallback=False), "claude-sonnet-5-5", "u", llm.CLIP_SCHEMA, "x")
    assert "extra_body" not in client.api.calls[0]


def test_refusal_retries_on_refusal_model():
    client = FakeClient({
        "claude-sonnet-5-5": [msg("claude-sonnet-5-5", stop="refusal", category="general_harms")],
        "claude-opus-5-5": [msg("claude-opus-5-5", {"clips": [CLIP]})],
    })
    data = llm.ask(client, cfg(), "claude-sonnet-5-5", "u", llm.CLIP_SCHEMA, "x")
    assert data["clips"][0]["title"] == "t"
    assert [c["model"] for c in client.api.calls] == ["claude-sonnet-5-5", "claude-opus-5-5"]


def test_refusal_without_refusal_model_skips():
    client = FakeClient({"claude-sonnet-5-5": [msg("claude-sonnet-5-5", stop="refusal")]})
    assert llm.ask(client, cfg(refusal_model=None), "claude-sonnet-5-5", "u",
                   llm.CLIP_SCHEMA, "x") is None


def test_parallel_counts_failures_and_keeps_going():
    chunks = [SEGS[:10], SEGS[10:]]
    client = FakeClient({"m": [msg("m", {"clips": [CLIP]}), msg("m", {"bad": 1})]})
    raw, failed = llm.find_clips_parallel(client, cfg(refusal_model=None), "m", chunks, 1, ERRORS)
    assert len(raw) == 1 and failed == 1


def test_batch_reruns_errored_sections_directly():
    chunks = [SEGS[:10], SEGS[10:]]
    results = [
        NS(custom_id="section-2", result=NS(type="succeeded", message=msg("m", {"clips": [CLIP]}))),
        NS(custom_id="section-1", result=NS(type="errored")),
    ]
    created = {}
    batches = NS(
        create=lambda requests: created.setdefault("reqs", requests) and NS(id="b1"),
        retrieve=lambda bid: NS(processing_status="ended", request_counts=NS(processing=0)),
        results=lambda bid: iter(results),
    )
    client = FakeClient({"m": [msg("m", {"clips": [CLIP]})]})  # the direct re-run
    client.messages.batches = batches
    ids = []
    raw, failed = llm.find_clips_batch(client, cfg(), "m", chunks, ERRORS,
                                       on_created=ids.append, poll_seconds=0)
    assert ids == ["b1"]
    assert [r["custom_id"] for r in created["reqs"]] == ["section-1", "section-2"]
    assert "fallbacks" not in json.dumps(created["reqs"])  # not allowed on the Batches API
    assert len(raw) == 2 and failed == 0


def test_rescore_updates_scores_and_drops_missing_ids():
    cands = [Clip(10, 30, 7, "a", "h", "funny", "r"), Clip(40, 60, 7, "b", "h", "funny", "r")]
    reply = {"clips": [{"id": 1, "score": 9, "title": "B!", "hook": "hh", "reason": "rr"}]}
    client = FakeClient({"main": [msg("main", reply)]})
    out = llm.rescore(client, cfg(model="main"), cands, SEGS, ERRORS)
    assert [(c.title, c.score) for c in out] == [("B!", 9)]
    prompt = client.api.calls[0]["messages"][0]["content"]
    assert 'id="0"' in prompt and "line 2" in prompt  # transcript of the candidate is included


def test_rescore_failure_keeps_first_pass():
    cands = [Clip(10, 30, 7, "a", "h", "funny", "r")]
    client = FakeClient({"main": [msg("main", stop="max_tokens")]})
    out = llm.rescore(client, cfg(model="main"), cands, SEGS, ERRORS)
    assert out == cands


def test_rescore_prompt_escapes_draft_titles():
    # An unescaped '"' or '<' in a draft title broke the <candidate ...> tag.
    import re
    cands = [Clip(10, 30, 7, 'He said "NO WAY" <3 & ran', "h", "funny", "r")]
    prompt = llm.rescore_prompt(cands, SEGS, cfg())
    tag = re.search(r"<candidate [^>]*>", prompt).group(0)
    assert tag == ('<candidate id="0" start="10.0" end="30.0" '
                   'draft_title="He said &quot;NO WAY&quot; &lt;3 &amp; ran">')

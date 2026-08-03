"""Property-based tests for hone invariants."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies

from hone import Example, Message, Reader, Role, Splitter, Writer
from hone.normalize import Normalizer

messages_strategy = strategies.lists(
    strategies.tuples(
        strategies.sampled_from([Role.system, Role.user]),
        strategies.text(min_size=1).map(lambda s: s.strip() or "x"),
    ),
    min_size=1,
    max_size=4,
)

example_strategy = messages_strategy.map(
    lambda msgs: Example(
        messages=(
            *tuple(Message(role=r, content=c) for r, c in msgs),
            Message(role=Role.assistant, content="x"),
        ),
        metadata={},
    )
)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(example_strategy)
def test_writer_reader_roundtrip_preserves_example(example: Example) -> None:
    """read(write([example])) == [example] for any valid Example."""
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        path = _P(d) / "rt.jsonl"
        Writer().write(path, [example])
        loaded = list(Reader().read(path))
        assert loaded == [example]


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(strategies.lists(example_strategy, min_size=2, max_size=20))
def test_splitter_disjoint_for_random_examples(examples: list[Example]) -> None:
    """train and valid are disjoint for any input of >=2 examples."""
    train, valid = Splitter(0.25, seed=42).split(examples)
    train_ids = {id(e) for e in train}
    valid_ids = {id(e) for e in valid}
    assert train_ids.isdisjoint(valid_ids)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(strategies.lists(example_strategy, min_size=2, max_size=20))
def test_splitter_preserves_element_count(examples: list[Example]) -> None:
    """len(train) + len(valid) == len(examples) for any input."""
    train, valid = Splitter(0.25, seed=42).split(examples)
    assert len(train) + len(valid) == len(examples)


chat_record_strategy = strategies.fixed_dictionaries(
    {
        "messages": strategies.lists(
            strategies.fixed_dictionaries(
                {
                    "role": strategies.sampled_from(["user", "assistant"]),
                    "content": strategies.text(min_size=1).map(
                        lambda s: s.strip() or "x"
                    ),
                }
            ),
            min_size=1,
            max_size=2,
        ).map(lambda items: [*items, {"role": "assistant", "content": "x"}])
    }
)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(chat_record_strategy)
def test_normalize_produces_valid_example(record: dict[str, object]) -> None:
    """Normalizer().normalize(record) satisfies Example invariants."""
    example = Normalizer().normalize(record)
    assert isinstance(example, Example)
    assert len(example.messages) >= 2
    assert example.messages[-1].role is Role.assistant
    assert all(message.content.strip() for message in example.messages)


message_strategy = strategies.tuples(
    strategies.sampled_from([Role.user, Role.assistant]),
    strategies.text(min_size=1).map(lambda s: s.strip() or "x"),
).map(lambda rc: Message(role=rc[0], content=rc[1]))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(message_strategy)
def test_message_to_record_roundtrip(message: Message) -> None:
    """Message.to_record() is the canonical JSON representation."""
    record = message.to_record()
    assert record["role"] == str(message.role)
    assert record["content"] == message.content

from roadbot.messages.common import MessageHeader


def test_message_header_is_immutable() -> None:
    header = MessageHeader(timestamp_ns=1, sequence=2, source="test")

    assert header.timestamp_ns == 1
    assert header.sequence == 2
    assert header.source == "test"


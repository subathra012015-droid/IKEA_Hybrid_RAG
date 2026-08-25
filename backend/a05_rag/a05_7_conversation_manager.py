from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMessage:
    """
    One message in the conversation.
    """

    role: str
    content: str


@dataclass
class ConversationState:
    """
    Stores conversation history for one active session.

    This is currently an in-memory implementation.

    Later a frontend/session store can persist
    the history separately for each user session.
    """

    messages: list[ConversationMessage] = field(default_factory=list)

    max_messages: int = 12

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Add one user or assistant message.

        Args:
            role:
                "user" or "assistant"

            content:
                Message text.
        """

        cleaned_role = role.strip().lower()

        cleaned_content = content.strip()

        if cleaned_role not in {
            "user",
            "assistant",
        }:

            raise ValueError("Conversation role must be " "'user' or 'assistant'.")

        if not cleaned_content:

            return

        self.messages.append(
            ConversationMessage(
                role=cleaned_role,
                content=cleaned_content,
            )
        )

        self._trim_history()

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Add one user message.
        """

        self.add_message(
            role="user",
            content=content,
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Add one assistant message.
        """

        self.add_message(
            role="assistant",
            content=content,
        )

    def _trim_history(
        self,
    ) -> None:
        """
        Keep only the most recent messages.

        This prevents conversation history from
        growing indefinitely.
        """

        if len(self.messages) <= self.max_messages:

            return

        self.messages = self.messages[-self.max_messages :]

    def as_dict_list(
        self,
    ) -> list[dict]:
        """
        Return history in the format expected by
        the contextual-query component.

        Example:

        [
            {
                "role": "user",
                "content": "..."
            },
            {
                "role": "assistant",
                "content": "..."
            }
        ]
        """

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.messages
        ]

    def clear(
        self,
    ) -> None:
        """
        Clear the current conversation.
        """

        self.messages.clear()

    def message_count(
        self,
    ) -> int:
        """
        Return number of stored messages.
        """

        return len(self.messages)


def build_assistant_history_text(
    answer: str,
    options: list[str],
    follow_up_question: str,
) -> str:
    """
    Build the assistant message stored in history.

    This intentionally includes the available options
    because the user's next response may simply be:

        "Installation"

        "Online planning"

        "The second one"

    Keeping the option list in recent context helps
    contextual query rewriting.
    """

    parts = []

    cleaned_answer = answer.strip()

    if cleaned_answer:

        parts.append(cleaned_answer)

    if options:

        option_lines = ["Available options:"]

        for index, option in enumerate(
            options,
            start=1,
        ):

            option_lines.append(f"{index}. {option}")

        parts.append("\n".join(option_lines))

    cleaned_follow_up = follow_up_question.strip()

    if cleaned_follow_up:

        parts.append(cleaned_follow_up)

    return "\n\n".join(parts)

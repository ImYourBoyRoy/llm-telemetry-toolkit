# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/io/parser.py
"""
Content parsers and sanitizers.
Handles text truncation and extraction of special tags (like <think>).
Inputs: Raw text strings.
Outputs: Cleaned/Truncated strings, extracted components.
"""

import re
from typing import Optional, Tuple
from ..models.config import TelemetryConfig


class ContentParser:
    """
    Handles robust parsing and cleaning of LLM inputs/outputs.
    """

    @staticmethod
    def clean_and_truncate(text: str, config: TelemetryConfig) -> str:
        """
        Truncates text if max_content_length is set in config.
        """
        if text is None:
            return ""

        if config.max_content_length and len(text) > config.max_content_length:
            return text[: config.max_content_length] + "...[TRUNCATED]"

        return text

    @staticmethod
    def extract_thought_process(response_text: str) -> Tuple[Optional[str], str]:
        """
        Separates <think>...</think> content from the main response.
        Returns (thought_content, final_response)
        """
        if not response_text:
            return None, ""

        start_tag = "<think>"
        end_tag = "</think>"

        lower_text = response_text.lower()
        start_idx = lower_text.find(start_tag)
        end_idx = lower_text.find(end_tag)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            # Extract content
            # Use original casing indices
            thought = response_text[start_idx + len(start_tag) : end_idx].strip()

            # Remove the block from the response
            # We construct robustly using slicing
            before = response_text[:start_idx]
            after = response_text[end_idx + len(end_tag) :]
            final = (before + after).strip()

            return thought, final

        return None, response_text

    @staticmethod
    def redact_pii(text: str) -> str:
        """
        Redacts PII using Smart Masking (Partial visibility for debugging).
        Supports: Email, Phone, IPv4, Credit Cards.
        """
        if not text:
            return text

        # 1. Email: u**r@example.com (Keep 1st char and domain)
        # Regex: (FirstChar)(RestOfLocal)@
        text = re.sub(
            r"\b([a-zA-Z0-9])([a-zA-Z0-9._%+-]+)(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            lambda m: f"{m.group(1)}***{m.group(3)}",
            text,
        )

        # 2. IPv4: 192.xxx.xxx.xxx (Keep 1st octet)
        # Regex: (Octet1)\.(Octet2)\.(Octet3)\.(Octet4)
        text = re.sub(
            r"\b(\d{1,3})\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            lambda m: f"{m.group(1)}.xxx.xxx.xxx",
            text,
        )

        # 3. Credit Card: 4111 xxxx xxxx 1111 (Keep first 4, last 4)
        # Regex: (4 digits)[- ]?(4 digits)[- ]?(4 digits)[- ]?(4 digits)
        text = re.sub(
            r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b",
            lambda m: f"{m.group(1)} xxxx xxxx {m.group(4)}",
            text,
        )

        # 4. Phone: +1 (555) ***-**67 (Smart generic mask)
        # We try to capture standard Area Code + Prefix + Line
        # Matches: +1 (555) 123-4567, 555-123-4567, 555 123 4567
        # Strategy: Keep Country/Area, Mask middle, Keep last 2
        # Regex Breakdown:
        # Group 1 (Optional Country/Area): (\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?
        # Group 2 (Prefix): \d{3}
        # Group 3 (Line): \d{4}
        # Note: This is simplified. Phone regex is hard.
        # Let's target the user explicit case: +1 (555) 123-4567
        # Pattern: (Start including area code) (3 digits) (separator) (4 digits)

        def mask_phone(m):
            # m.group(1) is everything before the last 7 digits usually
            # But regex matching is tricky to separate context cleanly.
            # Let's match the whole block and split manually or be specific.

            # User Format: +1 (555) 123-4567
            # We want: +1 (555) ***-**67

            full = m.group(0)
            # Find the last 4 digits
            digits = re.findall(r"\d", full)
            if len(digits) < 10:
                return full  # Not strict enough

            # Simple replacement: Replace digits in the middle
            # We want to preserve checking area code structure.

            # Robust approach: Replace the prefix (3 digits) and first 2 of line
            # Match: ... (prefix) ... (last4)
            return re.sub(r"(\d{3})([-\s.]?)(\d{2})(\d{0,2})", r"***\2\3\4", full)
            # Wait, that's complex logic inside lambda.

        # Simplified Regex for the User's preferred format and common US numbers
        # Matches: +1 (555) 123-4567 or 555-123-4567
        # Group 1: Area/Country (+1 (555) )
        # Group 2: Prefix (123)
        # Group 3: Separator (- or space)
        # Group 4: Line (4567)
        pattern = r"((?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?)(\d{3})([\s.-]?)(\d{4})"

        def replacement(m):
            return f"{m.group(1)}***{m.group(3)}**{m.group(4)[2:]}"

        # Result: +1 (555) ***-**67

        text = re.sub(pattern, replacement, text)

        return text

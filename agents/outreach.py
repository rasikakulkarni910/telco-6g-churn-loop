"""
Outreach Agent — deterministic intervention scripts per playbook × intensity.

LLM path is stubbed (generate_intervention_llm) for a later upgrade; unused in v1.
"""

from __future__ import annotations

from typing import Any

from agents.state import LoopState

_TEMPLATES: dict[str, dict[str, str]] = {
    "Network Rescue": {
        "digital": (
            "Hi {customer_id} — we’re seeing network quality dips after your 6G upgrade "
            "(QoS ~{qos:.2f}). We’ve prioritized a network rescue: optimized routing on "
            "your cell + a 7-day performance watch. Reply CARE if issues continue."
        ),
        "outbound": (
            "Outbound script: Apologize for post-migration QoS. Confirm device={device}. "
            "Offer technician callback within 24h, temporary speed boost, and goodwill credit. "
            "Risk={risk:.0%}. Close with commitment to follow up in 72 hours."
        ),
    },
    "Bill Clarity + Price Protection": {
        "digital": (
            "Hi {customer_id} — your bill moved ~{bill_pct:+.0%} after 6G migration. "
            "Here’s a plain-language breakdown + a 3-month price-protection lock so charges "
            "won’t rise again this quarter. View details in My Account → Bills."
        ),
        "outbound": (
            "Outbound script: Walk through bill delta ({bill_pct:+.0%}). Offer price protection "
            "and a one-time courtesy adjustment. Emphasize no surprise fees. Risk={risk:.0%}."
        ),
    },
    "6G Value Unlock": {
        "digital": (
            "Hi {customer_id} — usage dipped ~{usage_pct:.0%} after migration. Unlock 6G value: "
            "free 30-day premium streaming pack + tips to hit peak speeds on your plan. "
            "Tap to activate in the app."
        ),
        "outbound": (
            "Outbound script: Acknowledge usage drop ({usage_pct:.0%}). Demo one 6G feature "
            "relevant to device={device}. Offer value pack and schedule a tips SMS series."
        ),
    },
    "VIP Rescue": {
        "digital": (
            "Hi {customer_id} — as a {value} value member, you have a dedicated rescue path: "
            "priority support queue + retention specialist. We’ve opened case MIGRATE-VIP. "
            "Expect contact within 4 business hours."
        ),
        "outbound": (
            "Outbound VIP script: Soft landing for high-value migrant. Named specialist, "
            "loyalty discount options, and executive escalation path. Sentiment={sentiment:.2f}, "
            "risk={risk:.0%}. Do not transfer to general IVR."
        ),
    },
}


def generate_intervention(
    user_context: dict[str, Any] | LoopState,
    playbook: str,
    intensity: str,
) -> dict[str, str]:
    """Return deterministic message + channel for the chosen play."""
    intensity_key = "outbound" if intensity == "outbound" else "digital"
    playbook_templates = _TEMPLATES.get(playbook) or _TEMPLATES["6G Value Unlock"]
    template = playbook_templates[intensity_key]

    ctx = {
        "customer_id": user_context.get("customer_id", "customer"),
        "risk": float(user_context.get("risk") or 0.0),
        "qos": float(user_context.get("post_migration_qos") or 0.0),
        "bill_pct": float(user_context.get("bill_change_pct") or 0.0) * 100.0,
        "usage_pct": float(user_context.get("usage_change_30d") or 0.0) * 100.0,
        "device": user_context.get("device_type") or "your device",
        "value": user_context.get("value_segment") or "valued",
        "sentiment": float(user_context.get("support_sentiment") or 0.0),
    }
    message = template.format(**ctx)
    channel = "voice_outbound" if intensity_key == "outbound" else "in_app_and_sms"
    return {
        "playbook": playbook,
        "intensity": intensity_key,
        "channel": channel,
        "message": message,
    }


def generate_intervention_llm(
    user_context: dict[str, Any] | LoopState,
    playbook: str,
    intensity: str,
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.4,
) -> dict[str, str]:
    """
    LLM-ready stub — intentionally unused in v1.

    Signature is stable so Checkpoint 5/6 can swap generators without changing
    Decision/Learning contracts.
    """
    raise NotImplementedError(
        "generate_intervention_llm is a stub. Use generate_intervention() for deterministic v1."
    )


def run_outreach(states: list[LoopState]) -> list[LoopState]:
    """Attach intervention message/channel to each decided state."""
    updated: list[LoopState] = []
    for state in states:
        s: LoopState = dict(state)
        playbook = str(s.get("playbook") or "6G Value Unlock")
        intensity = str(s.get("intensity") or "digital")
        payload = generate_intervention(s, playbook, intensity)
        s["intervention_message"] = payload["message"]
        s["intervention_channel"] = payload["channel"]
        updated.append(s)
    return updated

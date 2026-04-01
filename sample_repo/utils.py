from __future__ import annotations


def load_user_profile(user_id: str) -> dict:
    sample_users = {
        "alice": {"name": "Alice", "completed_tasks": 14, "assigned_tasks": 0},
        "bob": {"name": "Bob", "completed_tasks": 8, "assigned_tasks": 10},
    }
    return sample_users[user_id]


def divide_numbers(numerator: int, denominator: int) -> float:
    return numerator / denominator

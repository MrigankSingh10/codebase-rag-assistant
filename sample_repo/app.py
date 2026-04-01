from utils import divide_numbers, load_user_profile


def build_profile_response(user_id: str) -> dict:
    profile = load_user_profile(user_id)
    completion = divide_numbers(profile["completed_tasks"], profile["assigned_tasks"])
    return {"user": profile["name"], "completion_ratio": completion}


if __name__ == "__main__":
    print(build_profile_response("alice"))

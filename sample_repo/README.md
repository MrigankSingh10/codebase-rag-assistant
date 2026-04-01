# Sample Repo

This sample repository intentionally contains a divide-by-zero bug in `utils.py`.

Example debug input:

```text
ZeroDivisionError: division by zero
Traceback (most recent call last):
  File "sample_repo/app.py", line 10, in <module>
    print(build_profile_response("alice"))
  File "sample_repo/app.py", line 6, in build_profile_response
    completion = divide_numbers(profile["completed_tasks"], profile["assigned_tasks"])
  File "sample_repo/utils.py", line 12, in divide_numbers
    return numerator / denominator
```

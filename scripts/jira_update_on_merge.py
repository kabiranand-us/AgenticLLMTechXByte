"""
Run by .github/workflows/jira-update-on-merge.yml when a PR merges.

Extracts a Jira ticket key from the merged PR's body (looks for "Fixes PROJ-123"),
posts a comment with the PR link + files changed + the PR description, and
transitions the ticket to a "done"-like status if one exists in its workflow.

All required values are read from environment variables so this can be tested
locally without GitHub Actions:

  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN  - Jira Cloud REST API credentials
  PR_BODY, PR_URL, PR_TITLE                  - from the merged PR
  CHANGED_FILES                              - newline-separated list (optional)
"""
import os
import re
import sys
import json
import base64
import urllib.request
import urllib.error


def _auth_header(email: str, token: str) -> dict:
    raw = f"{email}:{token}".encode()
    return {
        "Authorization": f"Basic {base64.b64encode(raw).decode()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _request(method: str, url: str, headers: dict, body: dict = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print(f"Jira API error {e.code} on {method} {url}: {e.read().decode()}", file=sys.stderr)
        raise


def extract_ticket_key(pr_body: str) -> str | None:
    match = re.search(r"Fixes\s+([A-Z][A-Z0-9]+-\d+)", pr_body or "", re.IGNORECASE)
    return match.group(1).upper() if match else None


def build_comment_text(pr_title: str, pr_url: str, pr_body: str, changed_files: list[str]) -> str:
    lines = [f"PR merged: {pr_title}", pr_url, ""]
    if changed_files:
        lines.append(f"Files changed ({len(changed_files)}):")
        lines.extend(f"  - {f}" for f in changed_files[:30])
        if len(changed_files) > 30:
            lines.append(f"  ... and {len(changed_files) - 30} more")
        lines.append("")
    lines.append(pr_body or "(no description)")
    return "\n".join(lines)


def post_comment(base_url: str, headers: dict, ticket_key: str, text: str) -> None:
    body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
        }
    }
    _request("POST", f"{base_url}/rest/api/3/issue/{ticket_key}/comment", headers, body)


def transition_to_done(base_url: str, headers: dict, ticket_key: str) -> None:
    """Best-effort: find a transition whose target status looks like 'done' and apply it.
    Does nothing (and doesn't fail the run) if no such transition exists - workflows vary
    per Jira project, and we'd rather skip than guess wrong."""
    data = _request("GET", f"{base_url}/rest/api/3/issue/{ticket_key}/transitions", headers)
    transitions = data.get("transitions", [])

    done_like = [
        t for t in transitions
        if t.get("to", {}).get("name", "").strip().lower() in ("done", "closed", "resolved")
    ]
    if not done_like:
        print(f"No done-like transition available for {ticket_key}; skipping status change. "
              f"Available: {[t.get('to', {}).get('name') for t in transitions]}")
        return

    chosen = done_like[0]
    _request(
        "POST",
        f"{base_url}/rest/api/3/issue/{ticket_key}/transitions",
        headers,
        {"transition": {"id": chosen["id"]}},
    )
    print(f"Transitioned {ticket_key} to {chosen['to']['name']}")


def main() -> int:
    base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]

    pr_body = os.environ.get("PR_BODY", "")
    pr_url = os.environ.get("PR_URL", "")
    pr_title = os.environ.get("PR_TITLE", "")
    changed_files = [f for f in os.environ.get("CHANGED_FILES", "").splitlines() if f]

    ticket_key = extract_ticket_key(pr_body)
    if not ticket_key:
        print("No 'Fixes PROJ-123' reference found in PR body; nothing to update.")
        return 0

    headers = _auth_header(email, token)
    comment_text = build_comment_text(pr_title, pr_url, pr_body, changed_files)

    post_comment(base_url, headers, ticket_key, comment_text)
    print(f"Commented on {ticket_key}")

    transition_to_done(base_url, headers, ticket_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())

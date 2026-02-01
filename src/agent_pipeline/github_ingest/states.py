from typing import TypedDict, List, Optional

class GitHubComment(TypedDict):
    comment_id: int
    user: str | None
    body: str
    created_at: str
    url: str | None


class GitHubIssueThread(TypedDict):
    number: int
    title: str
    body: str
    labels: List[str]
    state: str
    created_at: str
    updated_at: str
    user: str | None
    url: str
    comments: List[GitHubComment]


class GitHubInputState(TypedDict):
    repo: str
    max_pages: int
    state: str


class GitHubWorkingState(GitHubInputState):
    issues: List[dict]
    current_index: int

    current_issue: Optional[dict]
    current_comments: Optional[list]

    output: List[GitHubIssueThread]


class GitHubOutputState(TypedDict):
    issues: List[GitHubIssueThread]

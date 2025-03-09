import subprocess


def check_git_clean():
    """
    Check if the git worktree is clean. Needed for tagging
    the commit with the run name.
    """
    status_text = subprocess.check_output(
        "git status --porcelain", shell=True, encoding="utf8"
    ).strip()
    assert len(status_text) == 0, (
        "This script will not proceed until your worktree is completely "
        + "clean (unstaged and staged files)."
    )

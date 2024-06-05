<!-- The core Triton is a small number of people, and we receive many PRs (thank
you!).  To help us review your code more quickly, **if you are a new
contributor (less than 3 PRs merged) we ask that you fill out the following
checklist and include it in your PR description. -->

<!-- The pull request template is referenced from https://github.com/apache/spark/blob/master/.github/PULL_REQUEST_TEMPLATE -->


## What changes were proposed in this pull request?
<!--
Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue. 
-->


## Why are the changes needed?
<!--
Please clarify why the changes are needed. For instance,
  1. If you propose a new API, clarify the use case for a new API.
  2. If you fix a bug, you can clarify why it is a bug.
-->


## Does this PR introduce _any_ user-facing change?
<!--
Note that it means *any* user-facing change including all aspects such as the documentation fix.
If yes, please clarify the previous behavior and the change this PR proposes - provide the console output, description and/or an example to show the behavior difference if possible.
If possible, please also clarify if this is a user-facing change compared to the released Triton versions or within the unreleased branches such as master.
If no, write 'No'.
-->


## How was this patch tested?
<!--
If tests were added, say they were added here. Please make sure to add some test cases that check the changes thoroughly including negative and positive cases if possible.
If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
If tests were not added, please describe why they were not added and/or why it was difficult to add.
-->


## Checklists

<!-- Fill out the checklist by replacing `[ ]` with `[x]`. -->

- [ ] I am not making a trivial change, such as fixing a typo in a comment.

- [ ] I have written a PR description following these
  [rules](https://cbea.ms/git-commit/#why-not-how).

- [ ] I have used an LLM to copyedit my PR description and and code comments.

- [ ] I have run `pre-commit run --from-ref origin/main --to-ref HEAD`.

- Select one of the following.
  - [ ] I have added tests.
    - `/test` for `lit` tests
    - `/unittest` for C++ tests
    - `/python/test` for end-to-end tests
  - [ ] This PR does not need a test because `FILL THIS IN`.

- Select one of the following.
  - [ ] I have not added any `lit` tests.
  - [ ] The `lit` tests I have added are "minimal" -- they contain only the
    instructions necessary to exercise the bug.  (Usually running Python code
    and using the instructions it generates is not minimal.)
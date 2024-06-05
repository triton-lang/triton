The core Triton is a small number of people, and we receive many PRs (thank
you!).  To help us review your code more quickly, **if you are a new
contributor (less than 3 PRs merged) we ask that you fill out the following
checklist and include it in your PR description.**

Fill out the checklist by replacing `[ ]` with `[x]`.

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

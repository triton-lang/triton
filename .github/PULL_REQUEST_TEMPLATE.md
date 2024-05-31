# Pull Request Template

Please ensure the following are completed before creating a PR.

- [ ] **PR Description** is written in clear, idiomatic English and follows the
  [rules](https://cbea.ms/git-commit/#why-not-how) for a good PR description.

  (The LLM of your choice can help copyedit your PR description.  You can even
  give it your whole patch to analyze.)

- [ ] **Pre-commit checks** pass.
    ```bash
    pre-commit install
    pre-commit run --all-files
    ```

- [ ] **Tests** have been added and/or updated.
  - For changes to the backend: `/test/` (for `lit`), `/unittest/` (for
    `gtest`), or occasionally end-to-end tests like in
    `/python/test/unit/language/test_core.py`.
  - For changes to the frontend: `/python/test/`

- [ ] **Documentation**
  - [ ] The code contains comments where appropriate, written in clear,
    idiomatic English. Again, an LLM can help.
  - [ ] If appropriate, the Triton documentation have been updated.

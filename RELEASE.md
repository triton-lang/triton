# Releasing Triton

Triton releases provide a stable snapshot of the code base encapsulated into a binary that can easily be consumed through PyPI. Additionally, releases represent points in time when we, as the development team, can signal to the community that certain new features are available, what improvements have been made, and any changes that are coming that may impact them (i.e. breaking changes).

## Release Compatibility Matrix

The following compatibility matrix reflects the CPython wheels published on PyPI:

| Triton version | Python version | Manylinux version |
| --- | --- | --- |
| 3.7.1 | >=3.10, <=3.14 | glibc 2.27+ x86-64, AArch64 |
| 3.7.0 | >=3.10, <=3.14 | glibc 2.27+ x86-64, AArch64 |
| 3.6.0 | >=3.10, <=3.14 | glibc 2.27+ x86-64, AArch64 |
| 3.5.1 | >=3.10, <=3.14 | glibc 2.27+ x86-64, AArch64 |
| 3.5.0 | >=3.10, <=3.14 | glibc 2.27+ x86-64, AArch64 |
| 3.4.0 | >=3.9, <=3.13 | glibc 2.27+ x86-64 |
| 3.3.1 | >=3.9, <=3.13 | glibc 2.27+ x86-64 |
| 3.3.0 | >=3.9, <=3.13 | glibc 2.27+ x86-64 |
| 3.2.0 | >=3.9, <=3.13 | glibc 2.17+ x86-64 |
| 3.1.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 3.0.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.1 | >=3.6, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.0 | >=3.6, <=3.12 | glibc 2.17+ x86-64 |
| 2.2.0 | >=3.6, <=3.12 | glibc 2.17+ x86-64 |
| 2.1.0 | >=3.7, <=3.11 | glibc 2.17+ x86-64 |
| 2.0.0 | >=3.6, <=3.11 | glibc 2.17+ x86-64 |
| 1.1.1 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.1.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.0.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |

## Release Cadence

Starting with PyTorch 2.14, Triton feature releases (`X.Y.0`) are scheduled for even-numbered PyTorch minor releases. Odd-numbered PyTorch minor releases receive a Triton patch update (`X.Y.1`) from the same release branch. Each Triton target date is one week before the corresponding PyTorch final release candidate date. Dates are tentative and follow the [PyTorch release cadence](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence).

| Triton version | Release branch cut | Target date |
| --- | --- | --- |
| 3.8.0 | Jun 2026 | Aug 10, 2026 |
| 3.8.1 | --- | Oct 5, 2026 |
| 3.9.0 | TBD | Nov 30, 2026 |

### Release History

The following timeline records release branch cuts and the first publication of each release on PyPI. Patch releases are optional.

| Minor Version | Release branch cut | Release date | Patch Release date |
| --- | --- | --- | --- |
| 3.7.0 | Feb 2026 | May 7, 2026 | Jun 17, 2026 (3.7.1) |
| 3.6.0 | Nov 2025 | Jan 20, 2026 | --- |
| 3.5.0 | Aug 2025 | Oct 13, 2025 | Nov 11, 2025 (3.5.1) |
| 3.4.0 | Jun 2025 | Jul 30, 2025 | --- |
| 3.3.0 | Feb 2025 | Apr 9, 2025 | May 29, 2025 (3.3.1) |
| 3.2.0 | Oct 2024 | Jan 22, 2025 | --- |
| 3.1.0 | Jun 2024 | Oct 14, 2024 | --- |
| 3.0.0 | Jun 2024 | Jul 19, 2024 | --- |
| 2.3.0 | Dec 2023 | Apr 5, 2024 | May 27, 2024 (2.3.1) |
| 2.2.0 | Dec 2023 | Jan 10, 2024 | --- |

## Release Process

Triton releases are coordinated with PyTorch. In addition to the Triton test suite, release branches are tested against the PyTorch and vLLM nightly test branches. The release process is roughly as follows:

1. After completing a release, create the next release branch named `release/<MAJOR>.<MINOR>.x`.
2. Select an initial commit, verify that all tests pass, including the PyTorch tests, and trigger a nightly release.
3. Downstream projects test against the nightly release, report issues, and make any necessary updates.
4. Before the release deadline, regularly move the release branch closer to `main`, rerun the PyTorch tests, and trigger an updated nightly release.
5. At the release deadline, select the commit that has been fully validated by downstream partners. If an explicit cherry-pick is needed and the change would be included by the next update from `main`, submit the cherry-pick first to ensure that the subsequent update is accepted.

Triton consumers can help provide faster feedback and improve the release branch by configuring a nightly test branch against the in-progress Triton release.

## Release Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 3.2 for 3.3 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 3.2 for 3.3 release)
* Emerging hardware support - features for new hardware may be considered after verifying that the changes are isolated from and do not affect existing supported hardware
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **Feature work is not allowed for cherry-picks except for isolated emerging hardware support that satisfies the criterion above.** All PRs that are considered for cherry-picks need to be merged on trunk; the only exception is release branch specific changes. An issue for tracking cherry-picks to the release branch is created after the branch cut. **Only issues that have ‘cherry-picks’ in the issue tracker will be considered for the release.**

#### Agenda:

##### Announcements:
1. Triton conference planned mid September in the Microsoft Silicon Valley Campus.

##### Items:
1. Alternative backend development approach (e.g. AMD, Intel)
2. State of the documentation, is there a planned effort? If yes, what do you think is the priority?
3. Mechanisms for smaller technical discussions: Slack channel per topic? Dedicated meetings for some topics?
4. Stability, testing, regressions: Improving CI and conformance/testing for validating new back-ends.
5. Language improvements/pain points
6. Windows Support
7. Discussion of known/anticipated design changes for H100
8. Some specific more tactical areas:
   - int8.
   - A low hanging fruit is to let tl.dot take int8 and leverage mma.
   - Sm75.
   - device functions. How hard is this to support while Triton frontend traverses AST?
   - remove torch dependencies from the frontend. (it sounds like there is already progress on this but could be worth discussing)

##### Minutes
Recording link [here](https://drive.google.com/file/d/1uMlIvih_E5FITwPnNHwTYzo-UKqtey2c/view)

1. Backend plans/broader roadmap:
   - Plan is for major updates to come in the Triton development meetup which will happen mid-September. For major design changes, currently the plan is to not upstream them directly but have a staging state and different backends can be integrated through a plugin mechanism where Triton provides a layer at the Triton IR layer that is generic and other backends can plug into that.
   - Short term roadmap plans are very focused on things like improving all FP8 things on Ampere and Hopper support (end of August). After Hopper support lands, priorities will include refactoring codebase to increase maintainability.
   - Linalg – upstreaming on hold due to limited dev bandwidth. Want to build an ecosystem where others can leverage Linalg like passes developed in their backend.
   - For now, peak performance on Nvidia GPUs needs Nvidia specific things, but the convergence of programming models for different backends will allow convergence of hardware backend support in Triton.
2. Documentation:
   - OpenAI has included comments in the backend code.
   - Seek community involvement to improve tutorials, based on new users knowing what is missing.
   - Seek community involvement for signature changes and doc updates.
   - Thread created in slack for suggestions on areas needing doc updates. Ian Bearman and his team may have bandwidth to update certain documentation.
3. Discussion channels:
   - Preferred #dev channel in slack for technical discussions.
   - Between GitHub and Slack it would be good to post links into places so folks know discussions are happening elsewhere
4. CI/testing:
   - Pretty liberal in terms of accepting regression tests and integration tests for Nvidia.
   - Plugin interface tested like everything else, and regressions there would block merges into main.
   - Correctness/Performance of external backends are tested nightly, but regressions do not prevent wheels from being built.
5. Language improvements:
   - Have added location information support into Triton codegen.
   - Feel free to bring up pain points in slack.
7. Windows Support: Technically not difficult to get a preliminary version. Most of the maintenance burden would come from having to support it when it breaks.

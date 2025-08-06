# Agenda:

## Items:
1. Gluon update (Jeff Niu, OpenAI)
2. Interest and requirements for a nightly performance regression suite (Simon Waters,  kernelize.ai)
3. Triton developers’ summit update (Ofer Dekel, Microsoft)
4. Open mic for other topics.

## Minutes:
Recording link [here](https://youtu.be/zoSY_WXHmF0)

1. Triton developers’ summit update (Ofer Dekel, Microsoft)
    - 3rd Annual Triton Developer conference
    - Oct 21, 2025 (day before the PyTorch conference in SF)
    - Where: Microsoft Silicon Valley Campus, Mountain View, CA
    - There may be busses from SF to Mountain View (survey coming)
    - Up to 500 people can be accomodated in their auditorium.
    - Everyone interested in Triton, developers, developers working on extensions, etc.
    - Registration website is imminent! (possibly in a week).
    - Talks (proposed):
        - Nvidia - Blackwell optimizations
        - AMD - MI300/MI350
        - OpenAI - Gluon
        - Microsoft/LinkedIn - Liger-kernel
        - ByteDance - Triton distributed
        - Meta - Helion
        - GPU mode - community talk
        - And more!
    - Invitation letters will be available on the website.
    - Q> Any tutorials like how to write a kernel or perf analysis.
    - A> Not planned. Filled schedule with new tech over last year (working with Phil on program). Maybe we should extend to two days next year. Conference for professions. Should this be a conference for non-experts too? Targeting folks who know and live/breathe Triton.
    - A> Should have talks on tooling like Proton and guidelines on performance. Want people to be able to reproduce their results.
    - Q> Last years audience was Triton developers and Triton users but felt like the topic skewed toward developers and get people to contributed.  Any plan to have content for users?
    - A> First 2 talks on triton internals.  Others include tooling that should be interesting to users (like liger, triton-distributed, helion and GPU mode).  Users will benefit from learning what goes on under the hood.
    - Q> Social aspect to Triton conference?
    - A> Full day of talks with coffee breaks/lunch/happy hour for unstructured social interaction. No plans for structured social engagement (like breaking into pods). But still in flux. Would like suggestions for what we can do for other social engagements (send ideas to Ofer).
    - Q> is GPU mode led by Mark Saroufim?
    - A> Yes.
    - Q> Any Triton/workshops to be given in conjunction with the PyTorch conference?
    - A> No. Other than being in good proximity (location and timing wise). Hoping to get folks who are attending PyTorch conference will come out a day early for Triton Conference.
2. Gluon update (Jeff Niu, OpenAI)
    - A lower-level language based on the same compiler tech as Triton.
    - Expose more control over layouts, scheduling and memory. Bypasses middle-end, goes right to backend.
    - Can still use tile-based programming.
    - Expose more of the GPU to users.
    - Why Gluon? Out of the box better perf only approaches 80%.  Compilers struggling to make best use of hardware (hardware complexity).
    - Targeting:
        - better register and memory layouts
        - Warp specialization partitioning and loop scheduling
    - Gluon - a system programming language for GPUs.
        - expose low-level hardware details
        - tile-based abstraction
        - no global state management
    - Trade-offs
        - not hardware portable across hw platforms
        - you need hardware knowledge
        - harder to write
    - Implementation
        - @peterbell10 did most of the work.
        - Focus on blackwell, but some H100 support
    - Example: FMHA on B200
        - Still slower than cudnn
        - But much better than out of the box triton.
    - Future work
        - Very experimental
        - Need better layout management functions
        - *Not planning on accepting contributions now*
    - Q> Gluon is for specific type of GPU. What about other GPUs/generations?
    - A> Don't need to rewrite everything. To get best performance on newer generations, yes, you will need to do rewrites.  Kernels have bells and whistles. Triton kernels program are a declarative specification for what the kernel should do. The triton compiler figures out how to make that spec performant. With Gluon, you will need to do this yourself.
    - Q> In the future, will certain ops be implemented in Gluon vs in the compiler? E.g. tl.histogram written as a gluon kernel.
    - A> Probably not. Triton ops are tile-level. These aren't exposed in Gluon. Idea of interop between Gluon & Triton exist but may not be implemented.
    - Q> Pushing onus like scheduling to kernel writers, Any thoughts about tooling to help guide the kernel writers like timeline views?
    - A> 1) intrakernel profiler with proton (very imporant, NCU stall counts example of something that might not be on the critical path) complicated dependency graphs 2) more function calls in gluon. but you won't see them in cuda gdb. Tooling needs to catch up and we expect it to do so.
    - Q> Microkernel for hotloops. Is this what you're envisioning for interop?
    - A> No, we haven't thought about it that much. If you had a large kernel, but our kernels are small so its not worth it.
    - Q> AMD other processors & gluon.
    - A> AMD is as simple as adding the bindings and Python code. But its very early and we're focusing on executing on Blackwell.
3. Interest and requirements for a nightly performance regression suite (Simon Waters,  kernelize.ai)
    - Brian Bowyer (kernelize.ai)
    - Nightly performance CI. In past we did the same at AMD while working on Triton compiler.
    - Noticed, almost every night, we would see performance regressions due to changes made during the day.
    - Hard to do performance optimizations if you don't know impact over different hardware, different versions, and data types.
    - Request to community:
        - Where to get resources to run on
        - Inside and outside of companies
        - Where to store the data
        - Help on setting up and running CI & doing operations.
    - Proposal from kernelize.ai
        - Nosql based cloud storage
        - pipelines on pulic cloud
        - Use torchbench to store tests
        - visualization: https://triton-bench.ai (currently contains fake data)
        - discord for questions
        - Run on AWS (to start)
    - Demo of dashboard
        - Personalizable
        - Dig into operators/hardware performance over time
        - Detailed views/exports.
    - Requests
        - kernelize.ai can provide people
        - We need community to help with costs(running tests)
        - kernels/data types/hardware.
    - Q> selfhosted runners.  How to run securely?
    - A> Manage it like cron. Meaning we'd do scheduling.  We have partners that have experience with secure cloud execution.
    - Q> Do you have live data?
    - A> Yes, 10 tests from tritonbench but just as a smoke test. We really want to know what to run.
    - Q> What is the business model?
    - A> This is for the community.  Meant to be publicly open.
    - Q> Challenging to run tests on Blackwell.
    - A> Expensive but we have access.  Amazon makes you buy a time block.
    - Q> Who's paying for this?
    - A> Asking community for support. Looking for the money or resources from community.
    - Q> What if hardware platforms look different for different businesses
    - A> We'll need to work with folks to figure out what makes sense to record like frequency pinning, OS, etc. (do this offline).
    - Q> Tritonbench at Meta is hosted on PyTorch Opensource allotment on Google Cloud with autoscaling in PyTorch. UI. would like A/B testing. Running experimental branches/repos and look for regressions/speedups.
    - A> I see that in tritonbench.
    - Will post on slack and discord
4. Open mic for other topics.
    - No additional topics.

## Minutes:
Recording link [here](https://youtu.be/zoSY_WXHmF0)

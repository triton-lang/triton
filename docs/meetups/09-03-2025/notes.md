# Agenda:
* Intros: Cicie Wang, and Whitney Tsang (co-organizers).
* Multi-pass profiler - a federated GPU Tooling Framework for Orchestrated and LLM Agentic Profiling Applications (Kevin Fang, et al., Meta)
* Triton Developer Conference updates (Ofer Dekel, Microsoft)
* Q> Who is using tritonbench? How are you using it? OpenAI? (Cicie Wang, Meta)
* Q> Triton testing strategy - what do folks think? What are we missing? Where would you like to see additional coverage? (Bill Yoshimi, Meta)
* Q> Free threaded Python.  Any plans for making it compatible with free threading? (Bill Yoshimi, Meta)
* Open mic for other topics.

# Notes:
* MPP
    * Lots of new DSLs (like Gluon and TLX) and profilers.
    * Working with Keren from OAI on profiling
    * Integrated wth compiler
    * Supports new DSLs
    * Structure-level profiling timelines
    * Operator-level latency
    * See OSDI ‘25 paper (accepted)
    * Approach
        * Connecting tools like profilers, LLM agents, etc to to different profiling backends (like proton, ncu, nvbit, etc.)
    * Requirements
        * Programmable interfaces
        * Eager execution (makes debugging easier)
        * Amenable to parallelization
        * Sandboxing - like for enabling agents to try experiments (to get a clean environment)
        * Debuggable.
    * Prototype
        * Data structures - program IR, execution traces, performance report
        * Abstractions - tasks and jobs (jobs can be nested)
    * System architecture
        * Job graph
        * MPP runtime - schedules tasks & eager execution
        * Backend - state caching, GPU/CPU pools. DB for error recovery
    * Case study 1: Profiling Async Operations
        * Sometimes difficult because some resources are shared.
        * We do multiple passes and measure statistical metrics.
        * Statistical timeline view.
        * MPP allows you to see distribution of execution times (P20, P50, P80)
    * Case study 2: Triton PGO Agent
        * Phases/Agents: profiling, summary, optimizer
        * Profiling: gets profile results
        * Summary: compress context window, generate a TL;DR
        * Optimizer: rewrites kernel to improve performance
        * Experimenting with TTGIR rewrites.
        * Examples: identifies section with high execution variation. Identifies critical path and suggests how to shorten them.
        * Results: compared to no profiling, NCU, with MPP (7-12% improvement).
        * Failure modes:
            * Kernel results change
            * Deadlocks
    * Case study 3: fine-grained IPC
        * Timing from proton intra kernel profiler
        * Instruction type stats from nvbit or cutracer (developed by Meta)
        * Can identify register pressure.
    * Conclusion
        * On top of proton, orchestrating profiling workflows
        * Soon to be open-source

    Q> How difficult is this to add other GPU vendors like AMD?

    A> If your backend can give you the data, we can do it.  We didn’t do it because we were interested in warp specialization.  It's general and you can implement the interface API.

    Q> Have you experimented with using the optimizer to rewrite assembly code?

    A> Demo used TTGIR but you can create an agent that could rewrite PTX or assembly.

    Q> Did you need to write prompt for the agent?

    A> Yes. It's a very simple prompt.

* Triton conference updates (Ofer Dekel, MSFT)
    * [https://aka.ms/tritonconference2025](https://aka.ms/tritonconference2025)
    * Schedule
        * Please show up to the happy hour to mingle (probably the most important part).
        * Register.  You’ll also need it for the live-stream too.  Sorry, you will not be able to register on the day of conference.
        * When you register, status is pending.  Will take up to a week to get it approved. (Why? Its going through Microsoft security review).
        * Please register with your institutional/professional email vs. yahoo/gmail/generic email. Generic email will take longer approve. You can ping Ofer if you haven’t seen your approval after 8+ days.
        * There will be busses to venue from SF.
        * Visa letter? Register soon so we can get you an invitation letter
    * Program
        * Phil & Thomas - Triton: today and beyond
        * Mark Saroufim - GPU MODE: the state of Triton
        * Jason Ansel - Helion: A higher-level DSL for Kernel Authoring
        * Keren Zhou (George Mason) & Kevin Fang (Proton: portable performance profiling)
        * Lixun Zhang (AMD) - No warm up needed: Triton day-one speed on AMD GPUS
        * Chris Sullivan (Nvidia) - Nvida Blackwell GPU backend for Triton
        * Peter Bell (OpenAI) - Gluon: tilebased GPU programming with low-level control.
        * Hongtao Y (Meta) - TLX
        * Wenlei Bao (Bytedance ) - Triton - distributed computation and communication overlapping
        * Yanming Chen (Linked in) - Evolution of Liger Kernels to post training
* Q> Who is using tritonbench? How are you using it? OpenAI?
    * [Kernelize.ai](Kernelize.ai) - vLLM testing tritonbench nightly. Built a visualization (noticed H100 and B200 regressions on Liger kernel and BF16).
    * OpenAI - not using tritonbench, using internal benchmarking system.  Lowtech stuff, ocaml (some of it is open sources in repo).  Simple benchmarking.
    * Q> no new kernels added
    * A> we’re continuously updating them, thinking of upstreaming more, attention, but no timeline.  We are keeping MoE update.
* Q> Triton testing strategy - what do folks think? What are we missing? Where would you like to see additional coverage?
    * Ettore - want so seem more lit test coverage, doesn’t require GPU.  Easier and fast to run. Vs testing operator end to end.
    * 20K unit tests are good, but if we want better improvements. Is to beef up the lit tests.GPU tests should be in third-party directory.  Add lit
    * Alex Baden: Tests: for important kernels, IR diffing! Cheaper to run (if the IR doesn’t change you shouldn’t have a regression.).  Use LLVM tooling to eliminate white space changes. **For important kernels, extract & compare IR changes.**
* Q> What is the Free-threading Python strategy?
    * Lots of things to fix in the front end (backend is pretty thread-safe.)
    * But its not high on the list of work we're doing (OAI).
* Q> Flex attention: update comments/docs to use tensor descriptors instead of TMA (unless TMA is really being referenced).
    * PyTorch flex attention uses tensor descriptors but comments/code reference TMA. Reaching out to owners of flex attention PyTorch inductor template kernels to update comments and code. Confusing for people who use GPUs that don’t implement TMA.
    * Ettore: FlexAttention FWD uses tensor descriptors but BWD doesn't, can someone add tensor descriptor support?

# Minutes
* Recording link [here](https://youtu.be/Ji1rCo6qvXc)
* MPP presentation link [here](https://tinyurl.com/4r7cfzhu)

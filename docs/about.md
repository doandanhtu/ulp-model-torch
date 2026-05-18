# From Excel to GPU: An Actuarial Cashflow Model in PyTorch

⚡ 10 million model points. Monthly cashflows. 90-year horizon. Under 2 minutes on a single GPU.

That's the runtime for a Universal Life cashflow model I built in Python/PyTorch over a few weekends as a personal experiment. What follows is a write-up of the benchmarks, the build, and what I took away from the process.

Full runtime numbers across compute platforms are shown below.

| Number of model points | Excel Python Orchestration | Prophet | PyTorch CPU | GTX 1060 6GB | T4 15GB | A100 40GB | A100 80GB | RTX Pro 6000 Blackwell Server Edition 96GB |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5m | 20,617s | 2,128s | 244s | 120s | 47s | 17s | 11s | 8.5s |
| 3m | — | — | 1,132s | 650s | 252s | 68s | 47s | 35s |
| 5m | — | — | — | 1,035s | 421s | 116s | 76s | 59s |
| 10m | — | — | — | — | 834s | 221s | 151s | 119s |

*Refer to benchmark notes at the end for more information.*

But the most interesting part wasn't the runtime.

It was how accessible the stack has become.

The entire prototype, from specification to implementation and testing, took approximately 48 hours of focused work in total, spread across spare time evenings and weekends over several weeks. Roughly half of the time went into drafting the model specification document. The implementation and testing took only around 3 working days, supported by Claude and Claude Code, despite my background being actuarial rather than computer science or software engineering.

What was built in that time is not a trivial model. It can support:

- monthly policy cashflow projections with long dependency chains in account value projections and decrements modelling,
- recursive zeroising reserve and net present value calculations,
- configurable assumptions and multi-scenario runs,
- and configurable output modes, including an aggregated portfolio summary with a set of major projected variables and a full per-policy projection output with an expanded variable set.

The repository also includes model specifications to give a sense of the overall complexity level. Correctness was validated against a full Excel reference model across all three calculation parts (PAV projection, decrements, and shareholder cashflows) under multiple scenarios and a range of edge cases. The PyTorch implementation matches the Excel reference to within three decimal places on all tested cases, with any residual differences attributable to floating-point representation.

## From Excel to GPU computing

This project was driven mostly by my curiosity. I started actuarial modelling in Excel nearly a decade ago, and over time I experimented with several approaches such as naïve Python, Julia and even low-level C. The result was always the same familiar tradeoff that actuarial teams have always faced. High-level languages were accessible but slow, while low-level languages were fast but significantly more complex and difficult to maintain.

Later, while learning more about deep learning, I came across large-scale vectorization and GPU computing. That's when I started wondering whether modern GPU frameworks could support actuarial modeling directly at scale from a high-level language like Python.

At first glance, actuarial models are very different from typical deep learning workloads. FP64 precision is often required, unlike deep learning where FP16 is sometimes sufficient. Beyond that, dependency chains can be very long, calculations are highly stateful and recursive, and projection logic can involve complex branching structures.

Still, I wanted to see how far modern GPU frameworks could go for actuarial models, and the results exceeded my expectations. Notably, these results were achieved with only one architectural change from the initial Claude Code implementation: replacing full tensor retention with a rolling buffer, which increased effective throughput roughly sixfold on the same hardware. No further performance tuning was applied.

## Why this may be interesting

This experiment makes me wonder: what will the next generation of actuarial infrastructure look like? Actuaries have always been on a relentless journey to pursue greater computing capabilities in service of increasingly complex models, from simple commutation functions to all kinds of complex stochastic models. The tools of the trade have evolved from actuarial tables and mechanical calculators to personal computers, spreadsheets, and dedicated actuarial software like Prophet, Axis, and Moses. GPU-native frameworks may represent the next major step in that evolution.

The adoption of GPU frameworks can benefit both industrial actuarial modeling and academic actuarial research, particularly for areas that have historically been constrained by computational limitations, such as large-scale stochastic modeling like TVOG calculations, nested stochastic simulations, ALM, policyholder behavior modeling, and potentially real-time or near real-time actuarial analytics.

As an illustration of what stochastic capability could enable, a rough scaling from the measured deterministic runtime suggests that multi-thousand-simulation TVOG calculations on portfolios with millions of model points may become operationally feasible within hours rather than days on relatively small GPU clusters. Actual runtimes would naturally depend on a lot more factors such as architecture, memory constraints, and orchestration overhead.

## Beyond runtime performance

Runtime aside, this kind of model development may also offer real operational advantages. Since it's just Python, modern software engineering practices can be integrated directly, e.g., version control through Git, automated validation pipelines against the Excel reference, flexible APIs, centralized governance via GitHub, and AI-assisted coding tools.

The openness of the ecosystem may also improve long-term extensibility. Modelling teams can retain direct ownership of the implementation and thus can abstract or reorganize the architecture in whatever way needed.

In practice, this may allow actuarial infrastructure to evolve more similarly to modern software systems, while still preserving actuarial control, validation standards, and auditability requirements.

## Why now?

What makes this time a particularly good moment for the leap is the combination of:

- increasingly accessible modern GPU infrastructure,
- mature high-level frameworks like PyTorch,
- large open-source ecosystems,
- and rapidly improving AI-assisted development tools.

Today, GPUs are more accessible than many may assume. Even an older consumer GPU not designed for high performance computing (HPC), such as the GTX 1060, can deliver meaningful acceleration over pure CPU. A T4 is available through Google Colab's free tier for experimentation. A Blackwell RTX Pro 6000 96GB, the most powerful GPU tested here, can be purchased for approximately $11,000 – $12,000 or can be rented for ~$1/hr (e.g., on Vast.ai). For teams already investing in AI infrastructure, the same hardware is likely already available. Why not put those GPUs to work for actuarial workloads too?

Historically, leveraging GPUs for HPC required serious low-level engineering in C/C++. Frameworks like PyTorch now allow for highly optimized GPU performance through high-level tensor abstractions. Their large-scale adoption in machine learning has also driven mature documentation, hardware support, and long-term maintenance.

The surrounding open-source ecosystem has also evolved enormously. Modern Python ecosystems now provide a lot of tools for testing, data processing, visualization, orchestration, and deployment. This lets actuarial teams plug into the same broader software infrastructure that powers modern scientific computing and ML systems.

AI-assisted development tools may further accelerate this transition by dramatically reducing the implementation overhead traditionally associated with technical infrastructure work. In practice, these tools can rapidly generate boilerplate code, assist with debugging and refactoring, explain unfamiliar frameworks, and shorten iteration cycles, allowing domain experts to prototype increasingly sophisticated systems with far less engineering friction than would previously have been possible.

## Final thoughts

Whether GPU-native actuarial modelling becomes mainstream remains to be seen. However, the combination of accessible GPU infrastructure, mature tensor frameworks, expanding open-source ecosystems, and modern AI tools appears to have shifted the practical feasibility frontier much further than I previously expected. Several years ago, a tool that could process millions of model points in minutes, built in a few weeks by a single actuary working part-time, would have seemed impractical, if not impossible.

At minimum, this experiment suggests that GPU-native actuarial modelling is no longer merely a theoretical possibility reserved for highly specialised engineering teams. It is becoming increasingly accessible to actuaries themselves.

Have you experimented with GPU computing for actuarial workloads, or seen other teams doing this? I'd be curious to hear what's worked and what hasn't.

---

*Benchmark notes*

- *All figures in the table are measured runtimes, not projections or extrapolations.*

- *Excel Python Orchestration refers to an Excel-based actuarial model orchestrated through Python to process multiple Excel instances in parallel.*

- *Prophet benchmark is based on a standard run without MPF batching, which was not available in the tested setup.*

- *Excel, Prophet, and PyTorch CPU benchmarks were run on Intel i7-14700 with 32GB RAM.*

- *GTX 1060 6GB is a local consumer GPU.*

- *T4, A100, and RTX Pro 6000 Blackwell Server Edition 96GB (G4) results were tested through Google Colab GPU environments using Colab Pro subscription.*

- *The purpose of the benchmark is not to claim optimal performance, but to demonstrate practical viability and accessibility of GPU-native actuarial modeling using modern open-source tooling.*

---

*GitHub repo: https://github.com/doandanhtu/ulp-model-torch*

>*Disclaimer: This is a personal experimental project using fabricated inputs and assumptions for illustration purposes only.*

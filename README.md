# tensorrt-plugin-autoresearch

## Background   
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) demonstrated that a 630-line Python script could autonomously improve ML models overnight — **100 experiments in a night** — by following simple principles: one metric, constrained scope, fast verification, automatic rollback, git as memory.    

Based on Karpathy's autoresearch — `constraint + mechanical metric + autonomous iteration = compounding gains`. So obey the roadmap: **Baseline → Modify → Verify/Metric → Keep/Discard → Repeat forever**.    
<br>
*"Set the GOAL → The agent runs the LOOP → You wake up to results"*
*You don't need AGI. You need a goal, a metric, and a loop that never quits.*
<br>

## How It Works

```
LOOP (N iterations or until done):
  1. Review current state + git history/results log (tune-log.csv)
  2. Pick the next change (based on what worked, what failed, what's untried)
  3. Make ONE focused change
  4. Git commit/Backup exp_<k>(before verification)
  5. Run mechanical verification (tests, benchmarks, scores)
  6. If improved → keep. If worse → git revert/roll back. If crashed → fix or skip.
  7. Log the result
  8. Repeat until N iterations complete or goal is met.
```

Every improvement stacks. Every failure/error auto-reverts. Progress is logged in CSV format.

### The Setup Phase

Before looping, Claude performs a one-time setup:

1. **Read context** — reads all in-scope files
2. **Define goal** — extracts or asks for a mechanical metric, usually from `SKILL.md`  
3. **Define scope** — which files can be modified vs read-only, usually from `SKILL.md`  
4. **Establish baseline** — runs verification/get baseline on current state (iteration #0)
5. **Confirm and go** — shows setup, then begins the loop

### Critical Rules

| # | Rule |
|---|------|
| 1 | **Bounded by default** — every command has a default iteration count; unlimited is opt-in via `Iterations: unlimited` |
| 2 | **Read before write** — understand full context before modifying |
| 3 | **One change per iteration** — atomic changes; if it breaks, you know why |
| 4 | **Mechanical verification only** — no subjective "looks good"; use unique metrics |
| 5 | **Automatic rollback** — failed changes revert instantly |
| 6 | **Simplicity wins** — equal results + less code = keep |
| 7 | **Git is memory** — experiments committed with `experiment:` prefix; agent reads `git log` + `git diff` before each iteration |
| 8 | **When stuck, think harder** — re-read, combine near-misses, try radical changes |

### Pipeline

1. **Baseline**    
```bash
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/compile.sh
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/get_baseline_latency.sh  
```

2. **Modify**   
```bash  
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/start_nsys.sh  
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/start_ncu.sh

# ai-drived   
```

3. **Verify & Metric**   
```bash   
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/compile.sh
bash .claude/skills/trt-plugin-kernel-latency-auto-opt/scripts/start_compare.sh <baseline_latency>
```

4. **Keep/Discard**   
```bash  

# ai-drived   
```

### more to see  
https://github.com/lix19937/llm-deploy/blob/master/auto_evolution.md   

-----

### [v1](v1.md) 
1. Pure cpp call `enqueue`
2. Benchmark with cudaEvent     

### [v2](v2.md)  
1. Use onnx which only include plguin node  
2. Benchmark with trtexec   
3. Verify with trtexec `--dumpOutput`

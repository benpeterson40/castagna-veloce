#!/usr/bin/env python3
# srvbig.py PORT: 20 varied greedy prompts (200 tokens each); per prompt t/s and draft acceptance, then the aggregate:
# total generated tokens / total decode time and total accepted / drafted (speculative changes that move greedy
# trajectories need many prompts: single prompts swing +-20%)
import json, sys, time, urllib.request
port = sys.argv[1]
prompts = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write a C function that reverses a singly linked list, with comments.",
    "Write a JavaScript function that debounces another function.",
    "Write a SQL query that finds the top 5 customers by total order value.",
    "Explain how a hash table handles collisions.",
    "Explain the difference between a process and a thread.",
    "Why do leaves change color in autumn? Answer in a short paragraph.",
    "Describe the water cycle for a ten-year-old.",
    "Write a short poem about the sea at night.",
    "Write a short story (about 150 words) about a lighthouse keeper.",
    "List ten countries in Africa and their capitals.",
    "List the planets of the solar system with one fact about each.",
    "Translate to German: 'I would like to book a table for two at seven o'clock.'",
    "Translate to Spanish: 'The library closes early on Sundays during the summer.'",
    "What is 17 times 23? Show the steps.",
    "Solve for x: 3x + 7 = 25. Explain each step.",
    "Give three tips for writing a good resume.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "What are the pros and cons of electric cars? Use a bulleted list.",
    "Write a haiku about programming.",
]
tot_n = tot_t = acc = dra = 0
tot_wall = tot_pp = 0.0
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0.0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    tot_wall += time.time() - t0
    tm = r.get("timings", {})
    tot_pp += tm.get("prompt_ms", 0.0)
    n = tm.get("predicted_n", 0); t = tm.get("predicted_ms", 0.0)
    tot_n += n; tot_t += t
    acc += tm.get("draft_n_accepted", 0) or 0; dra += tm.get("draft_n", 0) or 0
    print(f"{p[:44]!r:48} {n:4d} tok {tm.get('predicted_per_second', 0):6.1f} t/s  draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')}")
print(f"AGGREGATE: {tot_n} tok in {tot_t/1000:.2f} s = {1000*tot_n/tot_t:.2f} t/s, drafts {acc}/{dra} ({100*acc/max(dra,1):.1f}%), "
      f"prompt {tot_pp/len(prompts):.0f} ms/request, wall {tot_wall:.2f} s")

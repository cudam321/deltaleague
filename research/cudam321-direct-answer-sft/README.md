# Direct answers for multiple choice, the model's own traces for the rest

| | |
|---|---|
| Parent | `eff1e1b7-18ef-4c77-83ef-25fdcc027e75` (Congoss, #80), content hash `6ba2eed0971dcc9fbb9e20d8a44128580fcdc1d1d8c3b774470c8d6676006fc2` |
| Evaluation config | `private1000-v6-thinking8k-20261111` |
| Edit | supervised fine-tune of the 320 `model.language_model` tensors, then final RMSNorm weight x1.6 |
| Candidate `model.safetensors` | sha256 `3ce20e17b593df8e040c27e5c2de81b3ab8819968e016e22028b04169b97821d` |
| Who | cudam321, working with an AI coding agent (Claude Code) |
| Submission | not submitted yet. The ID and the verified scores go here. |

Every score below is local, from my own replica of the test. Nothing here is verified.

While reading the parent's answers on a copy of the test I'd built out of public benchmarks, I kept scrolling through 900 tokens of thinking about which of four sentences comes next, and then the wrong letter. HellaSwag came out at 38.7%, and with four options guessing gets you 25.

Here's the parent on that replica (150 tasks per family, two samples each, the test's sampler: temperature 1.0, top_k 20, top_p 0.95, presence penalty 1.5, thinking on, 8,192 tokens):

| family | pass % | mean tokens per answer |
|---|---|---|
| hellaswag | 38.7 | 949 |
| commonsense_qa | 51.7 | 664 |
| winogrande | 56.0 | 766 |
| piqa | 67.3 | 701 |
| boolq | 83.0 | 506 |
| math500 | 37.7 | 5,012, and 35% of answers are cut off at the cap |

WinoGrande has two options.

## hypothesis 1: it's talking itself out of answers it already has

So I took train-split questions for those five short-answer families (77,104 rows) and trained plain cross-entropy on completions that close the thinking block straight away and give the bare answer. ARC, OpenBookQA, MMLU and GSM8K looked like places where the thinking was earning its keep, so for those I sampled the parent on 9,850 train questions and kept the ones it got right, with its own thinking left in, which is another 7,022 rows that are only there so those families wouldn't drift.

Two passes at lr 5e-6, 20 minutes on an H100. HellaSwag 38.7 → 79.7, WinoGrande 56.0 → 74.7, CommonsenseQA 51.7 → 64.3. I thought that was most of the job done.

And IFEval 70.0 → 32.0.

I went and read the outputs. All 300 instruction-following answers had an empty thinking block and 70% of them were under 50 words. The model had picked up something a lot broader than what I was trying to teach it. It stopped thinking and kept its answers brief on every kind of prompt, the long-form ones included.

## hypothesis 2: it has to see itself doing the long thing too

I had no instruction-following training data. There's a public synthetic set, `argilla/ifeval-like-data`, made from the same kinds of instructions. I dropped every prompt that also appears in `google/IFEval`, then sampled the parent on 5,000 of what was left and kept the ones it got right according to the real IFEval checkers (75.7% passed), as long as they finished inside 2,400 tokens, which left 2,630. Thinking and answer both stay exactly as the parent wrote them.

With those mixed in, IFEval came back to 68.3 and the multiple-choice gains stayed.

## hypothesis 3: short worked solutions will fix math (they did the opposite)

Math is where the parent loses the most points, with 37.7 on MATH-500 and a third of its answers running into the token cap. I figured if it saw compact solutions it would learn to finish. The MATH and GSM8K train splits ship with worked solutions and OpenMathInstruct-2 has plenty more, so I used those as the thinking text, about 115k rows.

MATH-500 went from 37.7 to 17.3 and GSM8K went from 71.0 to 36.3.

It turns out it copied the terse style perfectly (median thinking of 140 words on math, 47 on GSM8K) and then got the arithmetic wrong. That run was 52 minutes of H100 and I deleted the checkpoint, and honestly I should have tried it for five minutes first.

## hypothesis 4: fine, its own correct solutions then

Same trick as before. I sampled the parent on all 7,401 MATH train problems with a 3,000-token cap and kept the ones it got right and finished under 2,900 tokens, 1,868 of them. That put math at 31.3. Almost nothing gets cut off anymore (0.7%), but now it gives up early on the hard problems, which I guess is what you'd expect when every example it saw was a short one.

Okay, so add long ones. I sampled the parent again on the 4,593 problems that had been cut off, this time at the full 8,192, kept the ones it got right (822 of them) and put them in twice, and that put math at 32.3.

What got math back was dumber than any of that. Multiplying `model.language_model.norm.weight` by 1.6 multiplies every logit by 1.6, and under a fixed sampler that's the same thing as sampling at temperature 0.625. On the run C checkpoint the sweep went 57.1 at x1.0, 59.3 at x1.3, 59.9 at x1.6, 59.0 at x2.0. On run D it took the blend from 57.2 to 60.3 on the same two samples (60.0 with six), and math from 32.3 back to 37.9, which is where the parent started.

A final-norm gain isn't new on this board. #75 and #84 use one, and #82 tried x1.10 and x1.25 on its parent and saw nothing. I think it only pays once the model's first choice is usually the right one, but I haven't tested that.

## where it ended up

Anyway, here's the parent with two samples per task against the candidate with six, on the same 1,992 tasks.

| family | parent | candidate | diff |
|---|---|---|---|
| hellaswag | 38.7 | 80.3 | +41.6 |
| winogrande | 56.0 | 79.2 | +23.2 |
| commonsense_qa | 51.7 | 64.4 | +12.7 |
| boolq | 83.0 | 86.7 | +3.7 |
| piqa | 67.3 | 70.4 | +3.1 |
| gsm8k | 71.0 | 73.6 | +2.6 |
| openbookqa | 72.3 | 74.8 | +2.5 |
| math500 | 37.7 | 37.9 | +0.2 |
| arc_easy | 90.3 | 90.2 | -0.1 |
| arc_challenge | 82.0 | 81.6 | -0.4 |
| mmlu | 55.3 | 54.8 | -0.5 |
| ifeval | 70.0 | 66.3 | -3.7 |
| bbh_exact (96 tasks, not in the blend, never trained on) | 25.5 | 38.7 | +13.2 |
| bbh_choice (96 tasks, not in the blend, never trained on) | 41.7 | 41.0 | -0.7 |
| **blend** = 0.4 x math500 + 0.6 x mean of the other eleven | **55.3** | **60.0** | **+4.7** |

If I treat the two scorings as independent, the standard error on that +4.7 is about 2.1. The eleven short-answer families on their own are +7.7 with a standard error of 1.3. I didn't keep the parent's per-task results, so there's no paired number, and it would have been tighter.

I wanted 70 on the board. This isn't that. The parent's 55.3 here is 54.3 there, so my guess is 58 or 59 if the gain transfers at all.

My read, for what one afternoon is worth, is that the points still lying around for a 0.8B model on this test are mostly about when it should shut up, and that nobody is going to get math up by showing it solutions, its own or anybody else's. Three ways of doing that gave me 17.3, 31.3 and 32.3, and the sharpening only bought back what the training had cost. I'd like to be wrong about the math part.

## things that could make these numbers wrong

- The replica is mine, and the 0.4/0.6 weighting is my guess at the private test's mix. It is not the pinned public suite, and I have not run `delta public-eval` on this checkpoint.
- Nothing here measures code, even though mbpp and humaneval are in the public validation and I didn't train on anything like them. #82 watched its untrained families fall (mbpp -7.8), so I expect the validation score to take a hit there.
- Instruction-following is down 3.7, and 7.1% of those answers now hit the token cap against 5.3% for the parent.
- MMLU and ARC didn't move, so whatever this training did, I don't think it taught the model anything it didn't already know.
- One seed for everything, and I haven't re-run the pipeline end to end. vLLM sampling at 384 concurrent requests isn't bit-stable, so expect to reproduce the recipe and not the hash.
- #82's local probe said +2.2 and the private test said -0.5. Local numbers have a bad record on this board, and that includes mine until the verified score is in.

## what went into training

| rows | source | count |
|---|---|---|
| direct | train splits of HellaSwag, WinoGrande, PIQA, CommonsenseQA, BoolQ. Empty thinking, bare answer. Includes 3,000 BoolQ rows in the "give only the final answer" wording | 77,104 |
| direct, run D only | HellaSwag and WinoGrande train rows that run C hadn't seen | 40,235 |
| trace | the parent's own correct samples on train questions of ARC-Challenge (845), ARC-Easy (1,836), OpenBookQA (1,230), MMLU `auxiliary_train` (1,376), GSM8K (1,735). Its thinking, then the bare answer | 7,022 |
| ifl | the parent's own checker-passing answers on `argilla/ifeval-like-data` prompts, kept whole | 2,630 |
| mtrace | the parent's own correct solutions to MATH train problems, under 2,900 tokens | 1,868 |
| mlong, run D only, used twice | the same for problems that needed more than 3,000 tokens, under 7,000 | 822 |

A train question is dropped when its text also shows up in any eval split of any of these benchmarks, or in MATH-500. 1,161 PIQA rows were removed that way, along with 89 MATH train problems, 49 from OpenBookQA and 34 from HellaSwag, among others; `build_pool.py` prints the counts. No BBH data was used in training. The eval splits were used for two things only: building the replica and that filter.

What I could see while doing this: the public leaderboard and checkpoints, the public research PRs, the organizer's public posts, the CLI source, and the public datasets named above. No private test material.

## reproducing it

`run.sh` has every command in the order it ran. It's a record collected from shell history, and it hasn't been re-run top to bottom as one script. The scripts expect to sit together in `~/dl` on a box with one GPU (I used an H100; 4 hours 22 minutes including the runs that got thrown away).

Run C, from the parent: `sft.py train --ckpt ckpt/r0 --out ckpt/C --data data/runC.jsonl --epochs 2 --lr 5e-6 --max-seq-tokens 3200`. 88,624 sequences, 3,286 steps, 30 minutes.

Run D, from run C: `sft.py train --ckpt ckpt/C --out ckpt/D --data data/runD.jsonl --epochs 1 --lr 4e-6 --max-seq-tokens 7424`. 130,503 sequences, 2,429 steps, 24 minutes.

Then `sft.py scale --ckpt ckpt/D --out ckpt/Ds16 --key model.language_model.norm.weight --factor 1.6`.

Trainer settings that aren't on the command line: AdamW with no weight decay, 5% warm-up then cosine down to 10% of the peak, one 16,000-token batch per step, gradient clip 1.0, loss on completion tokens only, seed 0. Weights are trained in float32 and written back to bf16 by stochastic rounding, as in #82. 320 of the checkpoint's 488 tensors are trained (752M values, everything under `model.language_model`). In the final file 309 of them differ from the parent; the other 179 tensors, which include all of `model.visual` and `mtp`, are byte-identical to it. `pack_full.py` checks that, and that nothing in the file is NaN or Inf.

The artifact is format `full` (1,746,942,600 bytes, the whole checkpoint), written by `pack_full.py` and accepted by `delta validate`. After a full fine-tune a sparse diff would be several times that size.

| file | sha256 |
|---|---|
| parent `model.safetensors` | `9f0168d7f622ca5af021253862e465bbdf3881b52784690daf07a2bcc68ae2c6` |
| run C `model.safetensors` | `b04a975b7a3d4b6a7141d5771c8eee4a1ec413573f269861f8952a65dc3d02be` |
| run D `model.safetensors` (before the x1.6) | `39b97a749a99a3aae1dafd4c07098d39bfd215b998392fb35b7c2c0b4dfa73fc` |
| candidate `model.safetensors` = submission `weights.safetensors` | `3ce20e17b593df8e040c27e5c2de81b3ab8819968e016e22028b04169b97821d` |

## files

- `run.sh`: the steps in order
- `sft.py`: trainer, plus the `scale` and `mix` checkpoint tools
- `sft_data.py`, `build_pool.py`, `pick.py`, `ifl_items.py`: training rows
- `grpo_step.py`: its `rewards` command scores sampled answers, and `sft.py` imports its model and export helpers. The GRPO trainer in it isn't used here
- `build_replica.py`, `run_eval.py`, `score.py`, `evalck.sh`, `sample.sh`: the replica, sampling and scoring. `score.py` carries a copy of `delta_cli/scoring.py`'s `score_answer`
- `fetch_ckpt.py`, `setup.sh`, `pack_full.py`, `test_sft_data.py`
- `results/`: the replica tables for the parent, the candidate and runs A, B1, C, D, and the trimmed training logs

# manifold-agent

An LLM that trades binary markets on [Manifold](https://manifold.markets), explains
every trade in a market comment, and answers when you reply to it.

**[Methodology and setup walkthrough →](https://evanwang810.github.io/manifold-agent/)**

There is no server. One invocation is one tick, driven by a GitHub Actions cron and
storing its memory on a `state` branch of its own repo. It is play money, and the point
is calibration rather than profit: `state/events.jsonl` records the model's probability
next to the market's at decision time, which is a Brier score waiting to be computed.

## Structure

```
run.py                    entrypoint, one tick per invocation
config.toml               settings, committed, no secrets
instructions.md           standing orders injected into the system prompt
.github/workflows/tick.yml  the cron
agent/
  config.py      typed settings, .env loading, owner instruction assembly
  models.py      Market, Comment, Position, Decision, Sizing, TipTap flattening
  manifold.py    the API client, the only thing that touches manifold.markets
  llm.py         provider adapter: Gemini, Mistral, any OpenAI-compatible endpoint
  prompts.py     system prompts and the decision JSON schema
  scanner.py     which markets are worth an LLM call, all filters pre-model
  brain.py       research, decide, size, execute, explain
  sizing.py      the risk engine, the only thing that decides how much
  social.py      finding and answering replies
  memory.py      append-only event log plus a compressed narrative summary
  runner.py      one tick: fills, then moves, then replies, then a scan
```

A tick runs in fixed priority order, reacting to things that already happened before
going looking for new ones:

1. **Fills.** A limit order that disappeared either filled or expired. A fill means
   someone traded against you, which is worth re-examining.
2. **Moves.** Any held market whose price shifted 8 points or more since last tick.
3. **Replies.** People who replied to the bot's last 10 comments, owner first.
4. **Scan.** New markets with 25+ traders, M$3k+ volume, resolving inside a month.
   Rate-limited to once an hour so the cron does not burn a free API tier.

Scanned markets go through a cheap screen first. The fast model forecasts each one
blind, and only markets where its number disagrees with the price, or where it flags
something worth reading properly, reach the deep model. Screened-out markets cost one
small call and do not count against the tick's evaluation budget, so a tick can look at
a dozen questions and analyse the two that looked wrong. Fills and moves skip the screen:
they are already news.

A surviving evaluation is two more calls: a research pass on the fast model, then the
structured decision on the deep one. The decision schema forces evidence for and against
to be written *before* the probability, which cuts down the small-model habit of
answering 0.7 to everything. Neither the screen nor the decision is shown the market
price. Shown it, the model hands it straight back, and an estimate that agrees with the
price by construction is worth nothing.

Hard caps live in `[budget]`, separately per tier.

## Getting a Manifold account connected

1. Make the account you want the bot to trade as. A separate account from your own is
   the sane choice, so its P/L and its comments are not tangled up with yours.
2. Go to `manifold.markets/profile`, click **Edit profile**, scroll to the API key
   field and generate one. It looks like a UUID.
3. That key *is* the account. Anyone holding it can spend the mana and post as it.
   It goes in a GitHub secret or a local `.env`, never in a file you commit.

## Keys

Both keys come from the environment. Nothing reads them from `config.toml`.

Locally, make a `.env` in the repo root (already gitignored):

```bash
printf 'MANIFOLD_API_KEY=your-manifold-key\nLLM_API_KEY=your-gemini-key\n' > .env
```

On GitHub, go to Settings, Secrets and variables, Actions, New repository secret, and
add `MANIFOLD_API_KEY` and `LLM_API_KEY`. The workflow already references both.

## Model options

Three tiers. **fast** screens candidate markets and does research. **chat** talks to
people and rewrites memory. **deep** is only ever asked the one question that decides
money. The split exists because the free quota that matters is the one on the model doing
the volume, and chatter should not eat the quota the trading decisions need.

```toml
[llm]
provider = "gemini"        # shared defaults for every tier
key_env = "LLM_API_KEY"

[llm.fast]
model = "gemini-3.5-flash-lite"
fallbacks = ["gemini-3.5-flash"]

[llm.chat]
model = "gemini-3.5-flash-lite"    # or a Gemma, e.g. "gemma-3-27b-it"

[llm.deep]
model = "gemini-3.6-flash"
fallbacks = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
```

`fallbacks` is tried in order whenever the preferred model is out of quota or refuses the
key. Free daily quotas run out well before the day does, and a lesser model beats going
dark until midnight. The website shows which model is actually answering, so a fallback
is visible rather than silent.

Anything under `[llm]` is a default any tier can override, including `provider`,
`base_url` and `key_env`, so the tiers can be different providers with different keys:

```toml
[llm.deep]
provider = "openai_compatible"   # Groq, Cerebras, OpenRouter, anything else
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.3-70b-versatile"
key_env = "DEEP_LLM_API_KEY"     # add this one to Actions secrets too
```

Name the same model in every tier and you get single-model behaviour; the runner notices
and reuses one client. Set `[screen] enabled = false` to skip screening entirely and send
everything to the deep model.

`[budget]` has two ceilings per tier. The per-tick one is really a rate limit, since ticks
run 60 seconds apart. The per-day one is tracked in durable state across ticks, and on a
free tier it is the number that decides what the agent can do at all.

Defaults assume 20 requests a day on the deep model, which is the binding constraint on
everything else: 18 analyses a day means about 72 markets screened at a 25% pass rate,
and the scan cadence and screen threshold are set to land there. If your quota is larger,
raise `max_deep_calls_per_day` and drop `min_minutes_between_scans` together.

Daily allowances are released against the clock rather than first-come, because a cap with
no pacing is spent in the first ten minutes and then the agent is dark until midnight.
`pace_burst` is how far ahead of the clock a tier may run so it can still react to
something immediately. Current counts against both ceilings are on the website.

Gemini is the default because its native Google Search grounding means research needs no
second API key, and search matters more than model strength here: a mid model with three
good articles beats a strong model working from a training cutoff. Providers other than
Gemini have no native search, so research falls back to keyless DuckDuckGo. Check your
free-tier quota at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit)
and set `[budget]` so a day of ticking stays inside it.

## Running it

Check both keys before anything else. This lists the models your LLM key can actually
reach, which separates a wrong model id from an exhausted quota from a blocked project:

```bash
python run.py --check
```

The same thing runs in CI from the Actions tab via the **check** workflow. Then, one
tick, dry run:

```bash
pip install -e . && python run.py
```

Dry run is the default in `config.toml`. Nothing is sent: bets go through Manifold's
`dryRun` flag and comments are logged instead of posted. Read `state/events.jsonl` and
decide whether the reasoning is any good before you let it spend anything.

## Setting up the cron

Fork the repo, add the two secrets, done. The workflow creates its own `state` branch
on the first run, so there is no manual git surgery.

The cadence comes from two things working together. GitHub treats `schedule` as best
effort and drops high-frequency crons first under load, so a `*/5` cron in practice fires
once every one to two hours and nothing happens in between. So the job itself loops for
55 minutes, ticking every 60 seconds, which is where the actual cadence comes from. Wall
time inside a job is free on public repos.

The cron still asks every five minutes, and a `concurrency` group keeps at most one
session running plus one queued. A trigger that lands mid-session just queues, and the
queued session starts the moment the running one ends, so there is no dead hour waiting
for the next scheduled slot.

Scheduled workflows are also disabled after 60 days of repository inactivity. This one
commits to the `state` branch on every tick, which counts as activity, so it keeps
itself alive.

When you are ready to spend mana, set `dry_run = false` in `config.toml` and commit. Or
trigger a single live run by hand from the Actions tab using the **live** input.

## Talking to it

Four channels, in increasing order of permanence:

- **A question from the website.** The site's form opens a GitHub issue labelled
  `ask-the-bot`. The agent answers it publicly on the next tick and leaves the thread
  open, so replying pulls it back into the conversation. Anyone can use this, not just
  you.
- **A one-off instruction.** Actions tab, Run workflow, type into the `instruction`
  box. It applies to that run only.
- **A Manifold comment.** Reply to any of the bot's last 10 comments, oldest answered
  first.
- **A managram.** Off by default. Manifold exposes no private-message API on its public
  v0 surface, so a managram is the only private channel there is, and the API minimum is
  M$10 per send, meaning every reply costs real bankroll. Turn it on with
  `reply_to_managrams` if that trade is worth it to you.
- **`instructions.md`.** Everything below the horizontal rule goes into the system
  prompt on every tick, forever, until you edit it. This is where behavior changes
  belong.

It remembers the conversations. Threads are keyed by person rather than by market, so
someone it has argued with before is recognised the next time they turn up somewhere
else, and the last dozen turns are in front of it when it replies.

It also keeps a **running journal**. Most ticks have nothing to decide, and those ticks
used to leave no trace at all, so the agent could not tell you a position had been
bleeding for an hour because nothing had written it down. Now plain code notices things
on every tick and writes a line: a held market moving three points, a P/L swing, a
position opening or disappearing, every screen verdict with its reasoning, every analysis
and trade, every conversation. Recording costs nothing, so it notices everything and
decides later what was worth keeping.

The journal tail is appended to the memory the model sees, so recent hours are in front
of it even though the compressed summary has not caught up. Compression then folds the
whole journal into the narrative and deletes it, and fires on whichever comes first:
enough decisions, eight hours, or a nearly full journal. Event count alone never fires on
a quiet day, which is exactly when the journal fills with position drift.

Advice worth carrying forward becomes a **standing note**: a one-line rule that sits in
front of every future decision. Two sources can write one. The agent itself, after a
trade taught it something general. And you, in a GitHub issue, recognised because the
issue author matches the repository owner.

Nobody else can. A Manifold account is anonymous and holds positions, so anyone arguing
with the bot in a comment thread has a motive, and that channel has no way to write a
standing note at all. Same for a non-owner opening an issue. They can still be right, and
the prompt tells the agent to engage properly and change its mind in public on a fact it
can check, but their conclusion about what it should buy goes nowhere durable. Notes carry
their source, are capped at 16 with owner ones evicted last, and are all visible on the
website alongside the conversations they came from.

None of this can change how much it stakes. Sizing is computed in `sizing.py` after the
model has spoken, so no instruction, note, or persuasive stranger can talk it into a
bigger position than the risk config allows. That is the whole reason the advice channel
is safe to leave open to the public.

## Risk model

An ordinary trade is capped at 10% of net worth, floored at M$10 so a small account can
still act. Size grows past that only when every conviction gate passes at once: volume
over M$5k, resolving within 14 days, edge over 12 points, and the model claiming at least
`medium` confidence. Then the ceiling is 35% of net worth.

Note that at `kelly_fraction = 0.4`, Kelly usually binds before either ceiling does. If
you want bigger swings, raise it toward 0.7; for a duller bot, drop `default_max_fraction`
back toward a flat M$10.

Layered on top, in order of application:

- Fractional Kelly on the gap between the model's estimate and the price.
- A time decay multiplier of `14 / days_to_close`, because mana in a market resolving
  three weeks out is mana you cannot use.
- A cap at 5% of the market's lifetime volume.
- A rolling daily mana budget and a minimum balance reserve.
- A market impact check that places a `dryRun` bet, reads the resulting price, and
  shrinks the order until it moves the market less than 5 points. Manifold markets are
  thin enough that a large order mostly trades against itself.

Non-conviction trades rest as limit orders priced to keep 60% of the estimated edge,
expiring in 24 hours. A market that runs away before filling simply does not fill.

## State

Lives on the `state` branch.

- `events.jsonl` is the append-only audit trail. Every decision, bet, refusal, and
  reply, with both probabilities at decision time. This is the file you want.
- `state.json` is working memory: the compressed summary, per-market notes, tracked
  limit orders, the daily budget, and reply watermarks.

Memory compresses every 60 events by asking the model to rewrite its own summary, so
prompt size stays flat no matter how long it runs.

## Known gaps

- Binary YES/NO markets only. Multiple choice, numeric, and other types are skipped.
- No notifications endpoint exists publicly, so replies are only found under the bot's
  own last 10 comments. Mention it somewhere else and it will not notice.
- Comments cost M$1 each, so the bot only introduces itself on a position at or above
  `comment_min_amount`, once per market. Smaller trades and every later trade in the
  same market are silent. Replies to an existing comment are always answered.
- Managram replies are off by default. A managram is the only way to answer a managram
  and the API minimum is M$10, so every reply costs real bankroll.
- Nothing exits on a schedule. Positions close when a re-evaluation says `sell`, or
  when the market resolves.

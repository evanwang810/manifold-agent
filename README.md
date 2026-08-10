# manifold-agent

An LLM that trades binary markets on [Manifold](https://manifold.markets), explains
every trade in a market comment, and answers when you reply to it.

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

Each evaluation is two LLM calls: a grounded research pass, then a structured decision.
The decision schema forces evidence for and against to be written *before* the
probability, which cuts down the small-model habit of answering 0.7 to everything.

Hard caps live in `[budget]`. Default is one evaluation and four LLM calls per tick.

## Getting a Manifold account connected

1. Make the account you want the bot to trade as. A separate account from your own is
   the sane choice, so its P/L and its comments are not tangled up with yours.
2. Go to `manifold.markets/profile`, click **Edit profile**, scroll to the API key
   field and generate one. It looks like a UUID.
3. That key *is* the account. Anyone holding it can spend the mana and post as it.
   It goes in a GitHub secret or a local `.env`, never in a file you commit.
4. Set `owner_username` in `config.toml` to **your** handle, not the bot's. That is how
   the bot recognizes you in comment threads and prioritizes your messages.

## Keys

Both keys come from the environment. Nothing reads them from `config.toml`.

Locally, make a `.env` in the repo root (already gitignored):

```bash
printf 'MANIFOLD_API_KEY=your-manifold-key\nLLM_API_KEY=your-gemini-key\n' > .env
```

On GitHub, go to Settings, Secrets and variables, Actions, New repository secret, and
add `MANIFOLD_API_KEY` and `LLM_API_KEY`. The workflow already references both.

## Model options

Default is `gemini-3.6-flash`. Gemini is the recommended provider because its native
Google Search grounding means research needs no second API key, and search matters more
than model strength here: a mid model with three good articles beats a strong model
working from a training cutoff. Check your actual free-tier quota at
[aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit) and set
`[budget]` so 288 ticks a day stays inside it.

Alternatives, all one config change:

```toml
provider = "mistral"             # model = "mistral-small-latest"

provider = "openai_compatible"   # Groq, Cerebras, OpenRouter, anything else
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.3-70b-versatile"
```

None of those have native search, so the agent tells itself its knowledge is stale and
leans harder on the market price and the comment thread. That is a real handicap on
news-driven questions and not much of one on questions the comments already settle.

## Running it

Locally, one tick, dry run:

```bash
pip install -e . && python run.py
```

Dry run is the default in `config.toml`. Nothing is sent: bets go through Manifold's
`dryRun` flag and comments are logged instead of posted. Read `state/events.jsonl` and
decide whether the reasoning is any good before you let it spend anything.

## Setting up the cron

Create the orphan branch the agent stores its memory on:

```bash
git switch --orphan state && git commit --allow-empty -m "init state" && git push -u origin state && git switch main
```

Add the two secrets, then enable Actions on the repo. The schedule starts firing on its
own. Two things to know about GitHub's scheduler:

- `*/5` is best effort. Runs get delayed or skipped under load, so real cadence is
  more like 5 to 15 minutes. Every tick is independent, so this does not matter.
- Scheduled workflows are disabled after 60 days of repository inactivity. This one
  commits to the `state` branch on every tick, which counts as activity, so it stays
  alive on its own.

When you are ready to spend mana, set `dry_run = false` in `config.toml` and commit. Or
trigger a single live run by hand from the Actions tab using the **live** input.

## Talking to it

Three channels, in increasing order of permanence:

- **A one-off instruction.** Actions tab, Run workflow, type into the `instruction`
  box. It applies to that run only.
- **A Manifold comment.** Reply to any of the bot's last 10 comments. Comments from
  `owner_username` get answered first, and the bot knows they came from you.
- **`instructions.md`.** Everything below the horizontal rule goes into the system
  prompt on every tick, forever, until you edit it. This is where behavior changes
  belong.

Instructions can change how the agent reasons. They cannot change how much it stakes:
sizing is computed in `sizing.py` after the model has spoken, so nothing written in a
prompt can talk it into a bigger position than the risk config allows.

## Risk model

Default trade is M$10. Size grows only when every conviction gate passes at once:
volume over M$10k, resolving within 7 days, edge over 18 points, and the model claiming
`high` confidence. Then the ceiling is 35% of net worth.

Note that at the default `kelly_fraction = 0.25`, quarter-Kelly binds long before that
ceiling does. A 25-point edge on an even-money market sizes to about 12% of bankroll,
not 35%. If you want the big swings, raise `kelly_fraction` toward 0.7.

Layered on top, in order of application:

- Quarter-Kelly on the stated edge.
- A time decay multiplier of `14 / days_to_close`, because mana in a market resolving
  three weeks out is mana you cannot use.
- A cap at 5% of the market's lifetime volume.
- A rolling daily mana budget and a minimum balance reserve.
- A market impact check that places a `dryRun` bet, reads the resulting price, and
  shrinks the order until it moves the market less than 3 points. Manifold markets are
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
- Managram replies are off by default. A managram is the only way to answer a managram
  and the API minimum is M$10, so every reply costs real bankroll.
- Nothing exits on a schedule. Positions close when a re-evaluation says `sell`, or
  when the market resolves.

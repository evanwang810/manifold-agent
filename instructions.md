# Standing orders

Everything **below the horizontal rule** is injected into the agent's system prompt on
every tick. Edit it, commit, and the next run picks it up. This is the main way to steer
the bot without touching code. Anything above the rule is ignored, including this note.

Keep it short. It competes for attention with the actual market being analyzed. Orders
can change how the agent thinks, not how much it can stake: sizing is enforced in
`agent/sizing.py` after the model has spoken.

Things that belong here:

- Stay away from markets about specific named private individuals.
- You have been overconfident on geopolitics. Cap yourself at medium confidence there.
- Prefer markets resolving on a public checkable number over author's-judgment ones.
- If a market's resolution criteria are missing or contradictory, hold. No exceptions.

---

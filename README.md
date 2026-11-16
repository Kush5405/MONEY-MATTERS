# 💸 Money Matters AI: Pre-Show Audit & Producer Engine

> Every applicant gets an instant answer. Every producer gets a ranked shortlist.

**Live demo:** [https://your-app-name.streamlit.app](https://money-matters-ai.streamlit.app/) *(replace with your Streamlit Community Cloud URL after deploying)*

## The problem

Thousands of viewers submit their salary, debt and investment numbers to appear on a personal finance show, and almost none of them ever hear back. The producers, meanwhile, are buried in messy submissions with no fast way to tell which stories would make a great episode.

## What it does

Two sides of the same submission, solved at once.

**Tab 1: Viewer Financial Self-Audit.** The applicant fills in their numbers and instantly gets:
- A **Financial Health Score** (0-100) with a Red / Amber / Green status
- A **Reality Check**: an honest, empathetic 3-sentence diagnosis in a direct, jargon-free tone
- **The Math**: savings rate, debt-to-income and emergency runway
- **Your Next 3 Moves**: concrete steps for the next 30 days

**Tab 2: wariCrew Producer Shortlist.** Every submission is automatically:
- Scored on **YouTube Story Hook Potential** (1-100)
- Given a **core conflict tag** and a **YouTube title idea**
- Checked for **critical red flags**
- Ranked in a live dashboard with a full **Producer Dossier** (why the guest will hold viewers, plus 3 sharp interview questions)

## Architecture

```
Streamlit UI  ──►  Deterministic Python math  ──►  Gemini (Structured Output)  ──►  st.session_state
 (2 tabs)          savings rate, DTI, runway,       Pydantic schema:                 (no external DB)
                   health score, red-flag rules     narrative + hook + dossier
```

- **Numbers are never guessed by the LLM.** Savings rate, debt-to-income, emergency runway and the health score are computed in Python. Gemini receives those exact figures and only writes the narrative and editorial judgement.
- **Structured Outputs.** Gemini responds against a Pydantic `response_schema`, so the UI never parses free text.
- **Graceful degradation.** If `GEMINI_API_KEY` is missing, invalid or rate-limited, the app falls back to a rule-based engine and tells the user, so the demo never breaks.
- **No external database.** Submissions live in `st.session_state`, seeded with 5 realistic Indian finance profiles so the dashboard is populated on first load. Use **Reset demo data** in the sidebar to restore them.
- **Consent built in.** Viewers can opt out of sharing their story with the producer queue.

### How the health score works

| Component | Weight |
|---|---|
| Savings rate (30%+ earns full marks) | 30 |
| EMI burden (0% best, 50%+ worst) | 25 |
| Emergency runway (6+ months earns full marks) | 25 |
| Insurance (Yes = 10, Only Corporate = 5, No = 0) | 10 |
| Total debt vs annual income | 10 |

## Run locally

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Optionally set `GEMINI_MODEL` to use a different Gemini model (default: `gemini-2.5-flash`).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io) pointing to `app.py`.
3. In **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy, then paste the URL into the **Live demo** line above.

For local development you can also put the same line in `.streamlit/secrets.toml`, which is git-ignored.

## Demo in two clicks

Open Tab 1, click a preset (**24yo in Debt Trap**, **High Earner, Zero Savings** or **Middle-Class Family Breadwinner**), then click **Run my audit**. Switch to Tab 2 to see the new applicant ranked among the seeded profiles.

## Notes

- Educational guidance only, not licensed financial advice.
- The 5 seeded profiles are fictional.

# Single-Agent Baseline: Arabic Flood-Crisis Tweet Classification

CPCS683 - Milestone 4 (Baseline running)
Team: Maya Kroma, Rasha Ghilan, Ghaida Fakery

This is the **single-agent baseline** of our project *"Multi-Agent AI for Arabic Flood-Crisis Tweet
Classification"*. One generalist LLM agent (Google Gemini) reads one Arabic tweet and predicts two labels:

- **Relevance**: `Related` or `Not related`
- **Information Type**: Affected individuals, Infrastructure and utilities damage, Donations and volunteering,
  Caution and advice, Sympathy and emotional support, Other useful information, or Not applicable

Consistency rule (one direction only): `Not related` implies `Not applicable`.

## How the baseline works (3 API calls per tweet)

1. **Call 1 - initial classification** of the tweet.
2. **Call 2 - self-review**: the same model checks the call-1 answer and corrects it if needed.
3. **Call 3 - final classification**, using the tweet, the call-1 answer and the call-2 review.

If an answer is not valid JSON, uses a label outside the closed sets, or breaks the consistency rule, the model
gets **one** repair attempt for that call. Temporary Google errors (429, 5xx) are retried up to 3 times.
For every tweet the code records: API calls, repairs, retries, input / output / reasoning tokens,
estimated cost (once prices are filled in, see Settings) and end-to-end time.

## Files

| File | What it does |
|---|---|
| `baseline.py` | The baseline: prompts, 3 calls, validator, retries, token and time tracking |
| `data_utils.py` | Downloads, cleans and splits the dataset (`python data_utils.py` prints the numbers) |
| `test_baseline.py` | Runs the baseline on 5 Development tweets and compares with the gold labels |
| `app.py` | Gradio web interface |
| `tests/test_offline.py` | Self-test with a fake model (no key, no internet) |
| `splits/` | Tweet IDs of the Development and Evaluation sets (IDs only) |
| `requirements.txt` | Python packages |

## Setup

You need a free Gemini API key from <https://aistudio.google.com/apikey>.
**Never write the key inside a file and never upload it to GitHub.** The code reads it from the
`GEMINI_API_KEY` environment variable.

### Option A - Google Colab

1. In Colab open the key icon (**Secrets**) on the left, add a secret named `GEMINI_API_KEY` with your key,
   and switch **Notebook access** on.
2. Run these cells (replace the repository address):

```python
!git clone https://github.com/ghaidadesu/arabic-flood-baseline.git
%cd <REPO-NAME>
!pip install -q -r requirements.txt
```

```python
import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")   # lets the scripts below see the key
```

```python
!python data_utils.py            # downloads the dataset, prints the numbers
!python test_baseline.py         # tests 5 Development tweets
!python app.py --share           # starts the Gradio interface, open the public link it prints
```

### Option B - your own computer (Python 3.10+)

```bash
git clone https://github.com/ghaidadesu/arabic-flood-baseline.git
cd <REPO-NAME>
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"      # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
python data_utils.py
python test_baseline.py
python app.py                              # open http://127.0.0.1:7860
```

## What you should see

- `python data_utils.py` prints: original tweets 4037, after cleaning 4028, Development 3223, Evaluation 805, overlap 0.
- `python test_baseline.py` prints, for each of the 5 tweets, the gold label, the prediction,
  CORRECT / WRONG, the number of calls, tokens and time, then a summary. It saves
  `results/dev_sample_results.csv` and `results/dev_sample_summary.json` (IDs and labels only, no tweet text).
- `python app.py` opens a page: paste an Arabic tweet, press **Classify Tweet**, and you get Relevance,
  Information Type and a line with calls, tokens and time.

To check the code logic without a key: `python tests/test_offline.py`. This uses a fake model, so it tests the
code, not the model quality.

## Data

We use the Arabic Twitter Corpus for Flood Detection (Alharbi and Lee, 2019):
<https://github.com/alaa-a-a/Arabic-Twitter-corpus-for-flood-detection>.
The source repository states no licence, so **we do not redistribute the corpus**. `data_utils.py` downloads
`ArabicFloods.csv` from the authors' repository into `data/` (ignored by git).

Preparation: 4,037 tweets -> remove the 2 tweet IDs whose duplicate rows have conflicting labels and keep one row
for each remaining repeated ID -> **4,028** tweets -> stratified split (by Relevance + Information Type, seed 42)
into **3,223 Development** and **805 Evaluation** tweets with no overlap.
The Evaluation set must stay untouched until the formal evaluation; use Development for prompt design and debugging.

## Settings

All in `baseline.py`: `MODEL_NAME` (default `gemini-3.5-flash-lite`, or set the `GEMINI_MODEL` variable),
`TEMPERATURE`, `SEED`, `MAX_REPAIRS_PER_CALL`, `MAX_API_ATTEMPTS`.
`PRICE_INPUT_PER_M` and `PRICE_OUTPUT_PER_M` are empty: copy the prices for your model from Google's pricing page
to get a cost estimate (until then `cost_usd` is `None`).

## Limitations

- Offline classification of historical tweets only; not for real emergency decisions.
- The three prompts contain label names and rules but no label definitions or examples.
- `--share` makes a public link that uses your API quota; stop the app after the demo.

## Reference

Alharbi, A. and Lee, M. (2019). *Crisis Detection from Arabic Tweets.* Proceedings of the 3rd Workshop on
Arabic Corpus Linguistics, 72-79.

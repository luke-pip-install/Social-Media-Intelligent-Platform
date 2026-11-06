# Signal Desk — Streamlit edition

## Run locally

1. Open PowerShell in this folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the packages and start Streamlit:

   ```powershell
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

   If `py` is unavailable on your machine, use `python -m venv .venv` for the first command.

Streamlit opens the app in your browser, normally at `http://localhost:8501`. Use the sidebar uploader on either technique page to load your data. Data is held in the local app process; nothing is sent to a hosted service. The **Basic Techniques** and **Advanced Techniques** pages are available in the Streamlit sidebar.

## Data formats

Upload CSV or JSON. CSV files may have a `text`, `tweet`, `content`, or `message` column and optional `date`, `user`, and `id` fields. Headerless six-column Sentiment140 files are supported. A `target` column enables training a local TF-IDF/logistic regression sentiment model when it contains enough examples of both classes. For generic unlabeled data, sentiment falls back to a transparent lexical baseline. JSON may be a list of objects or an object containing a `data`, `posts`, or `tweets` list.

## Analysis pages

- **Basic Techniques:** sentiment overview and classifier, a single-message sentiment predictor, hashtag and mention counts, time trends, optional spaCy entity extraction, and TF-IDF/K-means topic clusters with optional Qwen3-0.6B-generated labels.
- **Advanced Techniques:** news-assisted trend context (Google News RSS with GDELT DOC API fallback and simpler-query retry) with optional Qwen trend explanation, an evidence-based brand campaign brief with optional Qwen generation, a separate multilingual language/script analysis, and crisis keyword detection.

Optional named entity extraction needs an installed spaCy English model (`python -m spacy download en_core_web_sm`). News retrieval needs internet access. Crisis detection is a keyword baseline for human review, separate from language/script analysis; language alone is not treated as a crisis signal. Multilingual analysis uses `langdetect` when available and retains script-based fallback behavior for uncertain cases. It normalizes common social text patterns but does not automatically translate; an uploaded `translation` or `translated_text` column is displayed for comparison. Qwen topic labels, trend explanations, and campaign briefs are opt-in: install them with `python -m pip install -r requirements-qwen.txt`, then enable the corresponding Qwen option. In Trend context, enable **Generate a grounded trend explanation with Qwen3-0.6B** before clicking **Retrieve recent headlines**. On first use, Hugging Face downloads about 1.2 GB of model weights; generation runs locally and can be slow on a CPU. Qwen prompts treat form values and post excerpts as untrusted evidence, and the app rejects blank generations and a small set of known prompt-injection phrases. This output check is a guardrail, not a guarantee; review generated text before using it. The app does not call an LLM API.

The current project notebook is [NotebookName1.ipynb](../NotebookName1.ipynb). See the [demo README](README.md) for the current setup guide and the [project README](../../README.md) for the analysis pipeline and sampling limits. Sentiment140 is historical; new social posts require a separate platform export or API integration.

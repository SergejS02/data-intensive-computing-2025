# Assignment 3 – Serverless Review Pipeline  
_Data-Intensive Computing (VU) · TU Wien · 2025 S_

## 1 · Prerequisites
| Tool              | Version             | Purpose                       |
|-------------------|---------------------|-------------------------------|
| Python            | 3.11+               | venv, Lambda packaging        |
| Docker            | 20+                 | run LocalStack                |
| LocalStack CLI    | ≥ 3.x (community)   | AWS emulator                  |
| Bash / WSL 2       | Ubuntu 22.04 tested| run setup scripts             |
| (optional) VS Code| –                   | IDE                           |

> No AWS credentials required. Everything runs locally.

---

## 2 · Setup

```bash
cd assignment_3
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt      # boto3, pytest, tabulate, etc.

pip install localstack
localstack start -d                  # starts Docker-based emulator

awslocal s3 ls                       # should show empty output
```

---

## 3 · Build Lambda ZIPs

```bash
./package_lambda.sh preprocess
./package_lambda.sh profanity
./package_lambda.sh sentiment
```

Each command creates `src/lambdas/<name>/lambda.zip` with code, deps, and NLTK corpora.

---

## 4 · Provision LocalStack Resources

```bash
./setup_localstack.sh
```

This script (idempotent):
- Creates S3 bucket `reviews-input`
- Creates DynamoDB tables `Reviews` (stream enabled) and `Users`
- Adds SSM params `/dic2025/buckets/reviews`, `/dic2025/tables/reviews`
- Deploys Lambdas: preprocess, profanity, sentiment
- Wires S3 and stream events
- Grants invocation permissions

---

## 5 · Run Tests

```bash
pytest tests/ --sample 100        # fast (sample of 100)
pytest tests/                     # full run
pytest -m "not slow" tests/       # skip large devset
```

> You can adjust timeouts in `test_devset_integration.py` and `conftest.py`.

---

## 6 · Generate Results Table

```bash
python print_results.py
```

Sample output:
```
RESULTS for reviews_devset.json
| Metric          | Count |
|-----------------|-------|
| Positive        | 245   |
| Neutral         | 389   |
| Negative        | 166   |
| Profanity fails |  12   |

Banned users: 3  -> ['A2X...', 'A31...', 'A09...']
```

Use these values in `report.pdf` §4.

---

## 7 · Shutdown

```bash
localstack stop
docker system prune -f           # optional cleanup
```

---

## 8 · Troubleshooting

| Symptom                            | Cause                      | Fix                        |
|------------------------------------|-----------------------------|-----------------------------|
| Lambda not triggered               | Broken event config         | Re-run `setup_localstack.sh` |
| ImportError in Lambda              | Missing corpora/deps        | Re-run `package_lambda.sh`  |
| Timeout errors                     | Slow startup                | Increase timeouts in tests  |
| `--sample` unrecognized by pytest  | Not defined in conftest.py  | Ensure `pytest_addoption()` exists |

---

## 9 · Deliverables

- `report.pdf` (≤ 8 pages with diagram + result counts)
- `instructions.pdf` (this README is sufficient)
- Folder: `src/`, `tests/`, `scripts`
- Archive: `<GroupID>_DIC2025_Assignment_3.zip`

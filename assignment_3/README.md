# Data‑Intensive Computing – Assignment 3


## 1  Spin up LocalStack (Docker)

```bash
# From the project root
export SERVICES="lambda,s3,dynamodb,ssm"
export EDGE_PORT=4566            # default

docker run -d --name localstack \
  -p 4566:4566 -p 4510-4559:4510-4559 \
  -e SERVICES="$SERVICES" \
  -e DEFAULT_REGION=us-east-1 \
  localstack/localstack:latest
```

The container will expose `http://localhost:4566` for all AWS endpoints.

> **Tip:** To stop/reset, simply `docker rm -f localstack`.

---

## 2  Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # project‑level deps (pytest, boto3…)
```

> The Lambda‑specific dependencies are vendored automatically during the build step below; you do *not* need them in your venv.

---

## 3  Deploy the pipeline

Run the helper script that creates the S3 bucket, DynamoDB tables, SSM parameters, builds the three Lambda ZIPs, and wires all event sources:

```bash
./complete_setup.sh
```

If everything is green you should see

```
LocalStack is up.
Building preprocess …
Building profanity …
Building sentiment …
Deployment complete – run:  pytest tests/
```

---

## 4  Run the unit & integration tests

```bash
pytest -q      # all tests
pytest tests/test_unit_handlers.py     # just Lambda unit‑tests
pytest tests/test_integration_pipeline.py::test_user_gets_banned  # single test
```

All tests should pass; they exercise the stream wiring, strike‑counter logic, and sentiment thresholds.

---

## 5  Bulk‑load the full devset (optional)

If you want to push the entire `reviews_devset.json` (\~78 k reviews) through the pipeline:

```bash
./scripts/bulk_load_devset.sh   reviews_devset.json.gz
```

The script splits the corpus into 1‑review JSON files, uploads them to `s3://reviews-input/`, and waits until the **Reviews** and **Users** tables reach quiescence.

---

## 6  Testing

To check the results just run the results file - can be executing several times during the lamdbas are still running to check progress

```bash
python print_results.py 
```




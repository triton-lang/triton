# Perf Profile UI

Local frontend for engine, Zen bench, perf target, and TPM/A100e curve analysis.

## Run

```bash
cd /Users/qnie/code/personal/qnie/perf-profile-ui-app
export CHRONOSPHERE_API_KEY='...'
./start.sh
```

Open:

```text
http://127.0.0.1:5177/
```

Generated plots and CSVs are written under:

```text
/tmp/perf-profile-ui/
```

Local form defaults are saved in `defaults.json`. That file is intentionally ignored by git so rebases do not clobber your current inputs. If it is missing, the UI falls back to built-in defaults and browser localStorage.

## Rebase Checklist

After a rebase or fresh checkout:

```bash
cd /Users/qnie/code/personal/qnie/perf-profile-ui-app
python -m py_compile *.py
node --check server.mjs
export CHRONOSPHERE_API_KEY='...'
./start.sh
```

Then open `http://127.0.0.1:5177/`.

If port `5177` is already in use:

```bash
lsof -ti tcp:5177 | xargs -r kill
./start.sh
```

# MAS project continuity

Read `HANDOFF.md` before changing code or starting experiments. It is the shared, model-independent handoff: research scope, completed batches, exact status commands, resume limits, billing-unknown requests, and next tasks.

Use `.venv-native-benchmark/bin/python` for native benchmark tooling. Start with the read-only command:

```bash
.venv-native-benchmark/bin/python -m experiments.paired_deliberation.handoff_status
```

Do not rerun completed generation, overwrite sealed results, resend unknown intents, print `.env.local` secrets, or call public development results untouched confirmation. Keep the handoff current when stopping or changing research stage.

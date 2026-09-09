"""Exercise 0 smoke test: verify W&B integration by logging a dummy metric."""

import wandb

run = wandb.init(
    project="transformer-exploration",
    name="ex00-smoke-test",
    config={"phase": 1, "exercise": 0, "purpose": "integration check"},
)

for step in range(5):
    wandb.log({"dummy/loss": 1.0 / (step + 1)}, step=step)

run.finish()
print("run finished — check your dashboard for project 'transformer-exploration'")

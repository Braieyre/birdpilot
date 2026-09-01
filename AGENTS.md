<!-- agent-workflow:start -->
## Lightweight agent workflow

This project uses `.agent/PLAN.md` and `.agent/STATUS.md` for coordination.

- The codebase, `git diff`, and tests are the source of truth.
- Planner maintains the rolling plan and activates one outcome-oriented work package.
- Executor implements the active package and updates STATUS only at meaningful checkpoints, handoff, completion, or blockage.
- Reviewer verifies actual changes and tests before accepting work or activating the next package.
- Do not create progress logs, routine reports, duplicate status systems, or speculative work.
- Preserve existing instructions and user changes.
<!-- agent-workflow:end -->

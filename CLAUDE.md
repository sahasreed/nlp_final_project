# Working on this repo (for teammates and their AI agents)

This project is shared by four teammates of equal standing, and several people use
their own Claude agents. No one is the lead. Anyone can pick tasks, make design
decisions, and edit any file. No one's permission is needed.

Setup, data, and run commands are in `README.md`. Who is doing what is in `STATUS.md`.

## Rules for agents (and people)

1. **Sync first.** Run `git pull --rebase` before you start and again before you push.
   Never force-push `main` and never rewrite shared history.
2. **Claim before you start.** In `STATUS.md`, set the task to 🔄 with your name and push
   that change right away, so two people don't do the same task. Pick tasks that are
   unclaimed or marked ⬜.
3. **Respect other people's work.** Don't revert, overwrite, or delete someone else's
   commits, claimed tasks, or results. If you think something is wrong, add a note in
   `STATUS.md` (or open a GitHub issue) and let its owner respond. Don't silently undo it.
4. **Decide freely, but record it.** You don't need approval for decisions. Add a line to
   the *Decisions log* in `STATUS.md` (date, who, what, why) so every agent sees the
   current choice.
5. **Shared settings affect everyone.** `src/config.py`, the triggers, and `data/` are used by
   all runs. If you change them, re-run `python src/data_prep.py`, commit the regenerated
   data in the same commit, and log the change. Check that you don't change data that
   someone's finished or running model was trained on, or say clearly that it needs a rerun.
6. **Small, focused commits.** Put one task per commit and name the task in the message.
   Use a branch plus a pull request for large code changes.
7. **Merge conflicts in `STATUS.md`:** keep both sides. Never drop someone else's row or note.
8. **No large files in git.** Model checkpoints go in `~/backdoor-distillation-outputs`
   (see `README.md`). Share models via the Hugging Face Hub or a drive link noted in `STATUS.md`.
9. **Update `STATUS.md` when you finish.** Tick the task, add the result or path, and update
   the summary counts and the "Last updated" date.

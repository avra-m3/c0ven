# ADR-0058: main's served runs are stored per commit to screen a change; fresh ABBA decides

**Status:** draft, 2026-10-31.
**Revisit when:** a change passes against main's stored runs and is later found slower than main, or the host's
fingerprint misses a cause of drift that a check meets.

## Context

- [ADR-0016](0016-how-a-change-is-judged.md), item 4, judges a change by before and after runs that alternate (ABBA)
  on fresh processes. In [checking a change](../TESTING.md#checking-a-change), the before arm is main, run inside every
  check, so each check pays for main's runs again: half the quick round, and one 255k prompt.
- Main changes a few times a day, and many checks run against the same main commit.
- The host moves a run's time. Two main runs of one scenario minutes apart differ, and other services that hold the
  cards' memory, swap in use, the CPU governor and other jobs on the CPU move the time further. Alternating the arms
  cancels that drift; results stored hours earlier do not.

## Decision

1. **Main's runs are stored per main commit.** A check that runs main keeps its runs that passed, keyed by main's
   commit, the model file, the reference prompts, the scenarios and the engine image.
2. **With the runs, the host's fingerprint is stored:** `MemAvailable`, swap in use, each card's free memory and the
   CPU governor. Stored runs stand for main only while the host matches within set limits and the CPU is idle while
   the check runs nothing. Otherwise the check runs main fresh, as before.
3. **Stored runs screen; they never decide against a change.** A B run inside or better than main's stored runs
   passes that scenario. A B run slower than all of them leads to fresh ABBA rounds, and only those can fail the
   change as slower.
4. **A claim of a speed-up is always judged by fresh ABBA rounds**, as ADR-0016 says. Stored runs do not count
   towards it.

## Options not chosen

- **Fresh ABBA for every check, as before.** It is the most robust, and it spends a third of a quick round's card time
  and a 255k prompt on main again in every check.
- **Stored runs only, with a fresh main run when the host's fingerprint changes.** Faster again, but a change could
  then fail as slower on host drift the fingerprint misses, with no fresh A run to show it.

## Consequences

- A check with no claim, against a main commit with stored runs, runs two B quick rounds where it ran four ABBA runs,
  and no main run of the 255k prompt.
- The first check after main moves pays for main's runs and stores them for the checks after it.
- `tools/check.py` keeps the store with `--baselines`; the store is outside the repository.

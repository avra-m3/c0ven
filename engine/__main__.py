import os
import sys

from engine.process import main

status = main()
# A failed engine skips the driver's teardown: a queue on the cards may never finish, and the teardown would wait on
# it. serve has flushed the protocol's lines by now, and main's own lines are flushed as they are printed.
if status != 0:
    os._exit(status)
# A clean stop runs it. The driver keeps a loop running on each card's copy engine for a few milliseconds after the
# last copy, and only its teardown stops it for certain: a process that ends with one running can leave that card's
# copy engine unable to run new work, and the next start hangs on its first copy
# (docs/measurements/MEASUREMENTS.md#2026-11-06-copy-engine-stuck-after-exit).
sys.exit(0)

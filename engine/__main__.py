import os

from engine.process import main

# Not sys.exit: the driver's teardown at a normal exit waits for every queue, and a stuck queue never finishes.
# serve has flushed the protocol's lines by now, and main's own lines are flushed as they are printed.
os._exit(main())

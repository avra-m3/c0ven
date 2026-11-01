# Meridian Words

Meridian Words is a short narrated game about why c0ven is fast. An office building writes one word at a time, and
each floor is one of the engine's tricks dressed up as office work. The player bets on which change helps, watches a
stopwatch, and then the office freezes and labels show what each desk, shelf and van stands for. Code in one HTML file
draws all of it, with no models or textures: [meridian-words.html](meridian-words.html). The narration is 15 MP3
files in `meridian-words/`, next to it. On the
[docs site](https://avra-m3.github.io/c0ven/docs/showcase/meridian-words.html) the link opens the game itself.

Two floors are open. Each takes two or three minutes, has two endings, and shows captions whether or not sound is on.
Click the caption or press Space to skip a line, and press a number key to pick a choice.

## The floors

**Floor 1, the Shelf Room.** An adviser writes each word but must walk to a far shelf for the manual every time. The
player can train a faster thinker or move the shelf to the desk. Only moving the shelf helps much: the maths is quick
and reading the model's weights from memory is slow, so a faster engine mostly reads less, or reads from somewhere
closer. [Design: Memory](../DESIGN.md#6-memory) has the real figures.

**Floor 2, the Annex.** Each word needs a few advisers out of hundreds, and most of them work in an annex across town.
The player can drive one over with their crates of manuals, or fax them the question. The main office is the
graphics card and the annex is the CPU and main memory. c0ven keeps the busiest experts on the cards; for the rest,
the CPU works out the answer where the expert's weights already sit, and only the small result comes back.
[Design: The CPU side](../DESIGN.md#the-cpu-side) says how.

## How to make this

You need a coding agent that can write files, run a browser and run shell commands. The page loads three.js r128
from cdnjs and its fonts from Google Fonts, so it needs a network connection. It needs no build step.

### 1. Find the story

```text
Looking only at the code, explain why c0ven is faster than vLLM, llama.cpp and Strata. Now explain it as
a teacher would to a student. Then find an analogy a layperson would follow, an office full of workers,
where every one of the engine's shortcuts is something the office could do.
```

Then ask for the game:

```text
Structure this as a short game in the spirit of The Stanley Parable: a narrator who is friendly but may
be plotting against you, a choice on each floor, a bet before each choice, and a reveal after each floor
that names the real thing behind each part of the office. Keep each floor short.
```

The bet and the reveal matter most. Without them a player enjoys the story and does not notice they have
learned anything.

### 2. Build it in 3D

```text
Build the first floor in 3D with three.js in one HTML file, everything built in code. Show the narration
as closed captions and make the rest look like a game's interface: a stopwatch with the time split
into fetching and thinking, choice buttons with number keys, and an endings counter.
```

Then ask for one change at a time and look at each. The changes that helped most: the adviser sits down to think
so the thinking time is visible; the shelf pops out legs and waddles to the desk; and the camera frames the
whole of floor 2 at any window shape.

### 3. Record the narration

The voice is [Fish Audio](https://fish.audio)'s Drama 3 model. The script is one JSON file of lines, each with a
spoken version, which carries inline directions such as `[low, conspiratorial]` and tags such as `<slow>`, and a
caption version. Lines that play between two choices are joined and recorded as one file, so the narrator's tone
carries across sentences; a recording only splits where the story branches. ffmpeg trims the silence at each end
and shortens any pause inside to 0.8 s.

Captions light up word by word. Parakeet (`mlx-community/parakeet-tdt-0.6b-v2`, through `parakeet-mlx`) gives the
start time of each spoken word. A short script matches its words to the caption's words with Python's `difflib`,
and gives any caption word it cannot match a time spread evenly between its matched neighbours. The game holds
those times in `CHUNKS`, so it needs no timing files at run time. Without sound, a clock stands in for the audio
and the captions run at the same pace.

# Midnight in Blackwood Manor

## Story

It is 8:00 PM. You have stepped inside Blackwood Manor — an abandoned Victorian
mansion rumoured to be cursed by the restless dead. Spirits wander its halls,
a possessed doll stalks the dining room, and something far darker waits to be
unleashed in the cellar below.

Two relics are hidden somewhere in the house: the **Ancient Tome** and the
**Silver Locket**. Local legend says that together they can break the curse —
but only if the binding ritual is performed in the cellar where the curse was
born. Find them, perform the ritual, and escape through the front door before
sunrise at **6:00 AM** seals you inside forever.

---

## How to Play

### Running the Game
```bash
python game.py
```

You will be prompted for your name, then given the choice to enter the manor or
walk away. Type `enter` to begin.

### Commands

| Command | Description |
|---|---|
| `go [direction]` | Move to a connected room. E.g. `go north` |
| `north` / `south` / `east` / `west` / `up` / `down` |
| `look` | Re-describe the current room (enemies, items, exits) |
| `map` | Display a visual ASCII map of the entire manor |
| `take [item name]` | Pick up an item in the current room. E.g. `take Ancient Tome` |
| `inventory` or `i` | List everything currently in your bag |
| `fight` | Manually start a fight (enemies also trigger automatically on entry) |
| `help` | Show the in-game command reference |
| `quit` | Exit the game |

#### Combat Actions
When a fight starts you will be prompted each round to choose:

| Action | Description |
|---|---|
| `attack` | Standard d20 attack roll — your bread and butter |
| `power` | Higher damage range but slightly harder to land |
| `heal` | Recover HP **and** reduce Fear by the same amount |

> **Note:** You cannot flee. Every enemy in the manor blocks the room until
> defeated. Plan your route accordingly.

---

## Goal

1. **Collect the Ancient Tome** — found in the Living Room (north of the Entry Hall)
2. **Collect the Silver Locket** — found in the Study / Library (north of the Dining Room)
3. **Go to the Cellar with both relics** — located below the Kitchen (east of the Entry Hall, then down). Type `perform` when prompted to begin the ritual. The **Shadow Figure** will be summoned — you must defeat it to break the curse.
4. **Return to the Entry Hall** — once the ritual is complete, sprint back to the front door to escape.

Complete all four steps before **6:00 AM** or the curse claims you.

---

## Manor Map

```
   [ Study / Library ]
           |
  [Dining Room] --- [Living Room] --- [Entry Hall] --- [Kitchen]
                                                            |
                                                        [Cellar]
```

---

## Enemies

| Enemy | Location | Notes |
|---|---|---|
| Wandering Spirit | Living Room | Drops a Bandage on defeat |
| Possessed Doll | Dining Room | Drops a Healing Potion on defeat |
| Shadow Figure | Summoned in the Cellar | Boss — drains HP back on a successful hit |

The Wandering Spirit and Possessed Doll engage automatically the moment you
enter their room. The Shadow Figure only appears when you perform the ritual
in the Cellar. Defeating any enemy reduces your Fear by 10.

---

## Lose Conditions

| Condition | What happens |
|---|---|
| HP reaches 0 | You are killed — Game Over |
| Fear reaches 100 | You die of fear — unique death screen |
| Time reaches 6:00 AM | Sunrise seals the curse — Game Over |

---

## Tips

- **Check the map often.** Type `map` to see which rooms still have enemies (marked `X`) and where you are.
- **Plan your route before moving.** Each action costs 30 minutes. You only have 20 actions before sunrise.
- **Fight smartly.** Winning a fight removes 10 Fear — sometimes it is worth pushing through rather than trying to avoid enemies.
- **You need both relics before the ritual.** Entering the Cellar without the Tome and Locket does nothing — collect them first.
- **`heal` fights on two fronts.** It restores HP *and* reduces Fear by the same roll, making it the best option when both meters are getting low.
- **The Shadow Figure heals itself.** Use `power` attacks during the ritual fight to end it quickly before the drain adds up.
- **You can delay the ritual.** Once in the Cellar with both relics, type `wait` if you want to heal up before committing to the boss fight — just mind the clock.
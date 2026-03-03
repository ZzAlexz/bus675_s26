"""
Lab 1: Text-Based Adventure RPG
================================
Alexander Zermeno

Build your game here! This file contains all the starter code from the lab notebook.
Fill in the TODOs, add your own classes, and make it your own.

Run with: python game.py
"""

import random
import textwrap

# =============================================================================
# Dice Utilities
# =============================================================================

def roll_d20():
    """Roll a 20-sided die."""
    return random.randint(1, 20)


def roll_dice(num_dice, sides):
    """Roll multiple dice and return the total. E.g., roll_dice(2, 6) for 2d6."""
    return sum(random.randint(1, sides) for _ in range(num_dice))


# =============================================================================
# Character Classes
# =============================================================================

class Character:
    """Base class for all characters."""

    def __init__(self, name: str, max_health: int, strength: int, defense: int):
        self.name       = name
        self.max_health = max_health
        self.health     = max_health
        self.strength   = strength
        self.defense    = defense

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount: int):
        self.health = max(0, self.health - max(0, int(amount)))
        print(f"  {self.name} takes {amount} damage  ->  HP: {self.health}/{self.max_health}")

    def attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength
        print(f"\n{self.name} attacks!  (d20={roll} + STR{self.strength} = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(1, 6) + self.strength
            if roll == 20:
                damage *= 2
                print("  * CRITICAL HIT -- something ancient answers your defiance!")
            target.take_damage(damage)
        else:
            print("  Miss -- the shadows swallow your effort.")

    def __str__(self):
        return f"{self.name} (HP {self.health}/{self.max_health})"

class Player(Character):
    """The player character."""

    def __init__(self, name: str):
        super().__init__(name, max_health=50, strength=5, defense=15)
        self.inventory = []
        self.fear      = 0

    def pick_up(self, item: str):
        self.inventory.append(item)
        print(f"  [+] Added to inventory: {item}")

    def show_inventory(self):
        if not self.inventory:
            print("  Your bag is empty.")
            return
        print("  Inventory:")
        for item in self.inventory:
            print(f"    - {item}")

    def use_item(self, item_name: str):
        if item_name not in self.inventory:
            print("  You don't have that item.")
            return False
        if item_name.lower() in ["bandage", "potion", "healing potion"]:
            healed = random.randint(10, 14)
            self.health = min(self.max_health, self.health + healed)
            self.inventory.remove(item_name)
            print(f"  You use {item_name} and recover {healed} HP.  HP: {self.health}/{self.max_health}")
            return True
        if item_name in ["Ancient Tome", "Silver Locket"]:
            print(f"  You hold the {item_name} -- it hums faintly.")
            return True
        print("  That item can't be used right now.")
        return False

    def heal_ability(self):
        healed = random.randint(8, 14)
        self.health = min(self.max_health, self.health + healed)
        fear_reduced = healed
        self.fear = max(0, self.fear - fear_reduced)
        print(f"  You steady your breath and recover {healed} HP.  HP: {self.health}/{self.max_health}")
        print(f"  The act of focusing calms your mind.  [fear] -{fear_reduced}   Fear: {self.fear}/100")

    def power_attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength - 2
        print(f"\n{self.name} winds up a POWER ATTACK!  (d20={roll} + STR{self.strength} - 2 = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(4, 10) + self.strength
            if roll == 20:
                damage *= 2
                print("  * CRITICAL POWER HIT!")
            target.take_damage(damage)
        else:
            print("  The power attack misses!")

class Enemy(Character):
    """Base enemy class."""

    def __init__(self, name: str, health: int, strength: int, defense: int,
                 xp_value: int = 10, loot=None):
        super().__init__(name, max_health=health, strength=strength, defense=defense)
        self.xp_value = xp_value
        self.loot     = loot

    def attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength
        opener = random.choice([
            f"{self.name} drifts closer, warping the air...",
            f"The temperature drops as {self.name} moves.",
            f"{self.name} flickers in and out of sight...",
        ])
        print(f"\n{opener}")
        print(f"  (d20={roll} + STR{self.strength} = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(1, 6) + self.strength
            if random.random() < 0.20:
                surge = random.randint(2, 5)
                damage += surge
                print("  A wave of dread crushes your chest.")
            if roll == 20:
                damage *= 2
                print("  * Reality bends violently around you.")
            target.take_damage(damage)
        else:
            print("  It passes through you like cold mist.")

class WanderingSpirit(Enemy):
    def __init__(self):
        super().__init__("Wandering Spirit", health=22, strength=4, defense=12,
                         xp_value=8, loot="Bandage")

    def attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength
        opener = random.choice([
            "A mournful wail fills the room as the Spirit surges toward you.",
            "The Spirit dissolves into mist — then reforms inches from your face.",
            "Ice-cold hands reach through your chest. The Spirit is feeding.",
        ])
        print(f"\n{opener}")
        print(f"  (d20={roll} + STR{self.strength} = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(1, 6) + self.strength
            if random.random() < 0.20:
                surge = random.randint(2, 4)
                damage += surge
                print("  It passes through your ribs like smoke, leaving a cold hollow behind.")
            if roll == 20:
                damage *= 2
                print("  * The Spirit screams — the sound tears straight through your skull!")
            target.take_damage(damage)
        else:
            print("  The Spirit lunges but you stumble back — it passes harmlessly through the wall.")

class PossessedDoll(Enemy):
    def __init__(self):
        super().__init__("Possessed Doll", health=28, strength=5, defense=13,
                         xp_value=12, loot="Healing Potion")

    def attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength
        print(f"\nThe doll's porcelain face cracks into a silent grin.")
        print(f"  (d20={roll} + STR{self.strength} = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(2, 7) + self.strength
            if random.random() < 0.30:
                damage += random.randint(2, 4)
                print("  Its tiny fingers tighten with unnatural strength.")
            if roll == 20:
                damage *= 2
                print("  * Its head twists fully backward.")
            target.take_damage(damage)
        else:
            print("  It stumbles forward, scraping uselessly across the floor.")

class ShadowFigure(Enemy):
    def __init__(self):
        super().__init__("Shadow Figure", health=36, strength=6, defense=14,
                         xp_value=20, loot="Old Key")

    def attack(self, target: "Character"):
        roll         = roll_d20()
        attack_total = roll + self.strength
        print(f"\nThe shadows stretch unnaturally long. Something tightens around your heartbeat.")
        print(f"  (d20={roll} + STR{self.strength} = {attack_total}  vs  DEF{target.defense})")
        if attack_total >= target.defense:
            damage = random.randint(3, 8) + self.strength
            if roll == 20:
                damage *= 2
                print("  * The walls groan -- the house itself turns against you.")
            target.take_damage(damage)
            drain = random.randint(1, 4)
            old   = self.health
            self.health = min(self.max_health, self.health + drain)
            if self.health > old:
                print(f"  The Shadow Figure feeds on your fear.  (HP: {self.health}/{self.max_health})")
        else:
            print("  The darkness recoils -- for now.")

# =============================================================================
# Location Class
# =============================================================================

class Location:
    def __init__(self, name: str, description: str):
        self.name        = name
        self.description = description
        self.connections = {}
        self.enemies     = []
        self.items       = []

    def describe(self):
        print("\n" + "=" * 52)
        print(f"  {self.name.upper()}")
        print("=" * 52)
        for line in textwrap.wrap(self.description, width=50):
            print(f"  {line}")

        print()
        if self.enemies:
            print("  [!] ENEMIES PRESENT:")
            for e in self.enemies:
                print(f"       X  {e.name}  (HP {e.health}/{e.max_health})")
        else:
            print("  [ok] No enemies here.")

        print()
        if self.items:
            print("  [*] ITEMS IN THIS ROOM:")
            for item in self.items:
                print(f"       -  {item}")
        else:
            print("  [-] No items here.")

        exits = self.get_exits()
        print()
        print(f"  Exits: {', '.join(exits)}" if exits else "  No obvious exits.")
        print("=" * 52)

    def get_exits(self):
        return list(self.connections.keys())

    def add_connection(self, direction: str, location: "Location"):
        self.connections[direction] = location

# =============================================================================
# World Builder
# =============================================================================

def create_world():
    entry_hall = Location(
        "Entry Hall",
        "A narrow entry space with a dusty mirror and a small table holding "
        "unopened mail. The front door is shut tight -- the air feels heavy, "
        "like the house is waiting."
    )
    living_room = Location(
        "Living Room",
        "A worn couch faces a cold fireplace. Family portraits line the walls, "
        "but every face has been scratched away. A grandfather clock ticks "
        "unevenly, as if time itself is nervous."
    )
    dining_room = Location(
        "Dining Room",
        "A long table is set for dinner, everything covered in dust. "
        "One chair at the head is pulled out, and a faint whisper follows "
        "your footsteps."
    )
    kitchen = Location(
        "Kitchen",
        "Old appliances sit silent under flickering lights. Cabinet doors "
        "hang slightly open. The back door has been boarded shut from inside."
    )
    study = Location(
        "Study / Library",
        "Shelves of journals and brittle books line every wall. Papers on the "
        "desk describe rituals -- and warnings about something bound beneath "
        "the house."
    )
    cellar = Location(
        "Cellar",
        "Cold stone walls close in around you. The air smells damp and metallic. "
        "A ritual circle is scratched into the concrete -- unfinished, but waiting."
    )

    # Two-way connections
    entry_hall.add_connection("north",  living_room)
    living_room.add_connection("south", entry_hall)
    entry_hall.add_connection("east",   kitchen)
    kitchen.add_connection("west",      entry_hall)
    living_room.add_connection("east",  dining_room)
    dining_room.add_connection("west",  living_room)
    dining_room.add_connection("north", study)
    study.add_connection("south",       dining_room)
    kitchen.add_connection("down",      cellar)
    cellar.add_connection("up",         kitchen)

    # Enemies — Shadow Figure is NOT pre-placed; it is summoned by the cellar ritual
    living_room.enemies.append(WanderingSpirit())
    dining_room.enemies.append(PossessedDoll())

    # Items
    living_room.items.append("Ancient Tome")
    study.items.append("Silver Locket")

    return entry_hall

# =============================================================================
# Combat System
# =============================================================================

class Combat:
    PLAYER_TURN = "player_turn"
    ENEMY_TURN  = "enemy_turn"
    COMBAT_END  = "combat_end"

    def __init__(self, player: Player, enemy: Enemy, forced: bool = False):
        self.player = player
        self.enemy  = enemy
        self.forced = forced
        self.state  = Combat.PLAYER_TURN

    def start(self):
        print(f"\n{'-'*52}")
        print(f"  COMBAT: {self.player.name}  vs  {self.enemy.name}")
        if self.forced:
            print("  [!] The creature blocks your path -- you cannot flee!")
        print(f"{'-'*52}")
        while self.state != Combat.COMBAT_END:
            if self.state == Combat.PLAYER_TURN:
                self._player_turn()
            else:
                self._enemy_turn()
        return self._result()

    def _player_turn(self):
        print(f"\n  YOU: {self.player}   |   ENEMY: {self.enemy}")
        if self.forced:
            print("  Actions: attack / power / heal")
        else:
            print("  Actions: attack / power / heal / run")
        action = input("  > ").lower().strip()

        if action == "attack":
            self.player.attack(self.enemy)
            self._check_enemy_dead()
            if self.state != Combat.COMBAT_END:
                self.state = Combat.ENEMY_TURN

        elif action == "power":
            self.player.power_attack(self.enemy)
            self._check_enemy_dead()
            if self.state != Combat.COMBAT_END:
                self.state = Combat.ENEMY_TURN

        elif action == "heal":
            self.player.heal_ability()
            self.state = Combat.ENEMY_TURN

        elif action == "run":
            if self.forced:
                print("  You can't flee -- the creature stands between you and the door!")
            elif random.random() < 0.5:
                print("  You slip away into the darkness.")
                self.state = Combat.COMBAT_END
            else:
                print("  You can't escape!")
                self.player.fear = min(100, self.player.fear + 5)
                self._check_player_dead()
                if self.state != Combat.COMBAT_END:
                    self.state = Combat.ENEMY_TURN
        else:
            print("  Unknown action. Try: attack / power / heal / run")

    def _enemy_turn(self):
        self.enemy.attack(self.player)
        self.player.fear = min(100, self.player.fear + 5)
        print(f"  Fear: {self.player.fear}/100")
        self._check_player_dead()
        if self.state != Combat.COMBAT_END:
            self.state = Combat.PLAYER_TURN

    def _check_enemy_dead(self):
        if not self.enemy.is_alive():
            print(f"\n  {self.enemy.name} is defeated!")
            self.state = Combat.COMBAT_END

    def _check_player_dead(self):
        if not self.player.is_alive():
            self.state = Combat.COMBAT_END
        elif self.player.fear >= 100:
            print("\n  Your mind completely shatters. The darkness consumes you.")
            self.state = Combat.COMBAT_END

    def _result(self):
        if not self.enemy.is_alive():
            return "victory"
        if self.player.fear >= 100:
            return "fear_death"
        if not self.player.is_alive():
            return "defeat"
        return "fled"

# =============================================================================
# Main Game Class
# =============================================================================

class Game:
    EXPLORING  = "exploring"
    GAME_OVER  = "game_over"
    FEAR_DEATH = "fear_death"
    VICTORY    = "victory"

    START_HOUR   = 20      # 8 PM (24-hr clock)
    SUNRISE_HOUR = 6       # 6 AM
    MAX_ELAPSED  = 10.0    # hours from 8 PM to 6 AM
    TIME_PER_ACT = 0.5     # hours per action (30 minutes)

    def __init__(self):
        self.player           = None
        self.current_location = None
        self.state            = Game.EXPLORING
        self.game_running     = True
        self.elapsed          = 0.0
        self.ritual_done      = False

    # ENTRY POINT 
    def start(self):
        self._show_intro()
        self._create_player()
        self.current_location = create_world()
        self.current_location.describe()
        self._show_time()
        self._entry_choice()
        if not self.game_running:
            self._show_game_over()
            return
        while self.game_running:
            if self.state == Game.EXPLORING:
                self._exploration_loop()
            elif self.state == Game.GAME_OVER:
                self._show_game_over()
                break
            elif self.state == Game.FEAR_DEATH:
                self._show_fear_death()
                break
            elif self.state == Game.VICTORY:
                self._show_victory()
                break

    # INTRO / SETUP 
    def _show_intro(self):
        print("\n" + "=" * 52)
        print("  MIDNIGHT IN BLACKWOOD MANOR")
        print("=" * 52)
        print("  You enter at 8:00 PM. Sunrise seals the curse at 6 AM.")
        print("  Find the Ancient Tome & Silver Locket,")
        print("  perform the ritual in the Cellar, then escape")
        print("  through the Entry Hall -- before the dawn breaks.")
        print("=" * 52)

    def _create_player(self):
        print("\n  What is your name, adventurer?")
        name = input("  > ").strip() or "Adventurer"
        self.player = Player(name)
        print(f"\n  Welcome, {name}. Your nightmare begins.\n")

    def _entry_choice(self):
        print("  You stand at the threshold of Blackwood Manor.")
        print("  Type  enter  to step inside, or  leave  to walk away.\n")
        while True:
            choice = input("  > ").lower().strip()
            if choice == "enter":
                print("\n  The door groans as you step inside...\n")
                return
            if choice == "leave":
                print("\n  You turn away. The manor stands silent behind you.")
                self.game_running = False
                self.state = Game.GAME_OVER
                return
            print("  Please type: enter  or  leave")

    # MAIN EXPLORATION LOOP 
    def _exploration_loop(self):
        print("\n  What do you do?")
        print("  (look | go [dir] | take [item] | inventory | map | help | quit)")
        command = input("  > ").lower().strip()
        parts   = command.split()
        if not parts:
            return
        action = parts[0]

        if action == "help":
            self._show_help()
        elif action == "map":
            self._show_map()
        elif action == "look":
            self.current_location.describe()
            self._show_time()
            self._check_victory()
        elif action == "go" and len(parts) > 1:
            self._move(parts[1])
        elif action in ("north", "south", "east", "west", "up", "down"):
            self._move(action)
        elif action in ("take", "pickup", "grab") and len(parts) > 1:
            self._take_item(" ".join(parts[1:]))
        elif action in ("fight", "attack"):
            self._initiate_combat(forced=False)
        elif action in ("inventory", "i"):
            self.player.show_inventory()
        elif action == "quit":
            print("  Thanks for playing.")
            self.game_running = False
        else:
            print("  Unknown command. Type 'help' for options.")

    # MOVE 
    def _move(self, direction: str):
        if direction not in self.current_location.connections:
            print(f"  You can't go {direction} from here.")
            return

        self.current_location = self.current_location.connections[direction]
        print(f"\n  You move {direction}...")

        # Full room description every time you enter
        self.current_location.describe()

        # Advance timer and display clock
        self._advance_time()
        if self.state == Game.GAME_OVER:
            return
        self._show_time()

        # FORCED COMBAT: if enemies are present, fight starts automatically
        if self.current_location.enemies:
            enemy = self.current_location.enemies[0]
            print(f"\n  A {enemy.name} lunges from the shadows -- you have no choice but to fight!")
            self._initiate_combat(forced=True)
            if self.state in (Game.GAME_OVER, Game.VICTORY):
                return
        else:
            self._add_fear(3, "The silence presses in.")

        if self.state == Game.GAME_OVER:
            return

        self._check_victory()

    # TAKE ITEM 
    def _take_item(self, item_name: str):
        match = next((i for i in self.current_location.items
                      if i.lower() == item_name.lower()), None)
        if match is None:
            print(f"  There's no '{item_name}' here.")
            return
        self.player.pick_up(match)
        self.current_location.items.remove(match)
        self._advance_time()
        if self.state == Game.GAME_OVER:
            return
        self._show_time()
        self._check_victory()

    # COMBAT
    def _initiate_combat(self, forced: bool = False):
        if not self.current_location.enemies:
            print("  There's nothing to fight here.")
            return
        enemy  = self.current_location.enemies[0]
        battle = Combat(self.player, enemy, forced=forced)
        result = battle.start()

        if result == "victory":
            self.current_location.enemies.remove(enemy)
            if hasattr(enemy, "loot") and enemy.loot:
                loot_list = enemy.loot if isinstance(enemy.loot, list) else [enemy.loot]
                for item in loot_list:
                    self.player.pick_up(item)
            # Defeating an enemy eases the dread
            fear_relief = 10
            self.player.fear = max(0, self.player.fear - fear_relief)
            print(f"\n  The threat is gone. Your nerves settle slightly.")
            print(f"  [fear] -{fear_relief}   Fear: {self.player.fear}/100")
            self.current_location.describe()

        elif result == "fear_death":
            self.state        = Game.FEAR_DEATH
            self.game_running = False
            return

        elif result == "defeat":
            self.state        = Game.GAME_OVER
            self.game_running = False
            return

        self._advance_time()
        if self.state == Game.GAME_OVER:
            return
        self._show_time()
        self._check_victory()

    # VICTORY CHECK / CELLAR RITUAL 
    def _check_victory(self):
        inv        = getattr(self.player, "inventory", [])
        has_tome   = "Ancient Tome"  in inv
        has_locket = "Silver Locket" in inv

        # Cellar: trigger ritual if both relics are held ──
        if (not self.ritual_done
                and self.current_location.name == "Cellar"):
            if has_tome and has_locket:
                self._perform_ritual()
                if self.state in (Game.GAME_OVER, Game.FEAR_DEATH):
                    return
            else:
                missing = []
                if not has_tome:   missing.append("Ancient Tome")
                if not has_locket: missing.append("Silver Locket")
                print(f"\n  The ritual circle pulses faintly.")
                print(f"  You are still missing: {', '.join(missing)}.")
                print(f"  The curse cannot be broken without them.")

        # Entry Hall: escape after ritual
        if self.ritual_done and self.current_location.name == "Entry Hall":
            print("\n  The front door explodes open.")
            print("  A rush of cold night air fills your lungs.")
            print("  You sprint out and never look back.\n")
            self.state        = Game.VICTORY
            self.game_running = False

    def _perform_ritual(self):
        """Prompt the player to perform the ritual; summon and fight the Shadow Figure."""
        print("\n" + "=" * 52)
        print("  THE RITUAL CIRCLE")
        print("=" * 52)
        print("  The Ancient Tome and the Silver Locket pulse in")
        print("  your hands. The circle on the floor glows a deep")
        print("  arterial red. You know what must be done.")
        print()
        print("  Type  perform  to begin the ritual, or  wait  to hold off.")
        print("=" * 52)

        while True:
            choice = input("  > ").lower().strip()
            if choice == "perform":
                break
            if choice == "wait":
                print("\n  You hesitate. The circle dims slightly.")
                print("  (Return to the Cellar when you are ready.)")
                return
            print("  Type  perform  or  wait.")

        print("\n  You place the Tome and the Locket inside the circle.")
        print("  The air splits open like a wound.")
        print("  Something that has been waiting a very long time")
        print("  steps out of the dark.")
        print()
        print("  The Shadow Figure has been summoned.")

        # Summon the Shadow Figure and force the fight
        boss = ShadowFigure()
        battle = Combat(self.player, boss, forced=True)
        result = battle.start()

        if result == "victory":
            # Fear relief and ritual completion
            fear_relief = 10
            self.player.fear = max(0, self.player.fear - fear_relief)
            print(f"\n  [fear] -{fear_relief}  The entity is gone. Your nerves steady.")
            print(f"  Fear: {self.player.fear}/100")
            print()
            print("  " + "=" * 50)
            print("  The ritual is complete. The curse is broken.")
            print("  The house groans — walls crack — plaster falls.")
            print("  You feel the manor releasing its grip.")
            print()
            print("  *** GET OUT. NOW. GO TO THE ENTRY HALL. ***")
            print("  " + "=" * 50)
            self.ritual_done = True

        elif result == "fear_death":
            self.state        = Game.FEAR_DEATH
            self.game_running = False

        else:  # defeat (HP = 0)
            self.state        = Game.GAME_OVER
            self.game_running = False

    # TIME 
    def _advance_time(self):
        self.elapsed += self.TIME_PER_ACT
        if self.elapsed >= self.MAX_ELAPSED:
            print("\n  The first light of dawn spills through the windows.")
            print("  The curse hardens. The manor will not release you.")
            self.state        = Game.GAME_OVER
            self.game_running = False

    def _show_time(self):
        total_minutes = self.START_HOUR * 60 + int(self.elapsed * 60)
        hour_24       = (total_minutes // 60) % 24
        minute        = total_minutes % 60
        suffix        = "AM" if hour_24 < 12 else "PM"
        disp_hour     = hour_24 % 12 or 12
        remaining     = self.MAX_ELAPSED - self.elapsed
        print(f"\n  [clock] {disp_hour}:{minute:02d} {suffix}  --  {remaining:.2f} hr(s) until sunrise")

    # FEAR 
    def _add_fear(self, amount: int, reason: str = ""):
        self.player.fear = min(100, self.player.fear + amount)
        msg = f"  [fear] +{amount}"
        if reason:
            msg += f"  ({reason})"
        msg += f"   Fear: {self.player.fear}/100"
        print(msg)
        if self.player.fear >= 100:
            print("\n  Your mind shatters under the weight of the manor.")
            self.state        = Game.FEAR_DEATH
            self.game_running = False

    # MAP
    def _show_map(self):
        """
        Fixed-layout ASCII map of Blackwood Manor.

        Column layout (0-indexed char positions):
          Col A =  2  : Study / Library  (top row, above Dining Room)
          Col A =  2  : Dining Room      (middle row, leftmost)
          Col B = 22  : Living Room      (middle row)
          Col C = 42  : Entry Hall       (middle row)
          Col D = 58  : Kitchen          (middle row, rightmost)
          Col D = 58  : Cellar           (bottom row, below Kitchen)
        """
        cur = self.current_location.name

        # Build a 9-char wide cell for each room:  >[X Name  ]  or  [ Name   ]
        CELL = 20   # fixed cell width including brackets and marker

        def cell(name: str):
            loc    = self._find_location(name)
            here   = ">" if name == cur else " "
            enemy  = "X" if (loc and loc.enemies) else " "
            # Truncate display name to fit (max CELL-5 chars)
            disp   = name[:CELL - 5] if len(name) > CELL - 5 else name
            inner  = f"{enemy} {disp}"
            return f"{here}[{inner:<{CELL-3}}]"

        study   = cell("Study / Library")
        dining  = cell("Dining Room")
        living  = cell("Living Room")
        entry   = cell("Entry Hall")
        kitchen = cell("Kitchen")
        cellar  = cell("Cellar")

        # Fixed column start positions (spaces before each cell)
        # dining at col 2, living at col 2+CELL+3, entry at col 2+2*(CELL+3), etc.
        SEP = "---"
        col_dining  = 2
        col_living  = col_dining  + CELL + len(SEP)
        col_entry   = col_living  + CELL + len(SEP)
        col_kitchen = col_entry   + CELL + len(SEP)

        # Vertical bar positions (centre of each cell)
        bar_dining  = col_dining  + CELL // 2
        bar_kitchen = col_kitchen + CELL // 2

        def pad(n):
            return " " * n

        print("\n" + "=" * 76)
        print("  BLACKWOOD MANOR  --  MAP")
        print("  Legend:  > = you are here     X = enemy present")
        print("=" * 76)
        print()

        # Row 0: Study (centred above Dining Room column)
        study_indent = col_dining + (CELL - len(study)) // 2
        print(pad(study_indent) + study)

        # Row 1: vertical connector Study <-> Dining Room
        print(pad(bar_dining) + "|  (north / south)")

        # Row 2: main corridor
        print(pad(col_dining)  + dining  + SEP
            + living  + SEP
            + entry   + SEP
            + kitchen)

        # Row 3: vertical connector Kitchen <-> Cellar
        print(pad(bar_kitchen) + "|  (down / up)")

        # Row 4: Cellar (centred under Kitchen column)
        cellar_indent = col_kitchen + (CELL - len(cellar)) // 2
        print(pad(cellar_indent) + cellar)

        print()
        print("  Route to victory:")
        print("    Entry Hall --north-> Living Room --east-> Dining Room --north-> Study")
        print("    Entry Hall --east->  Kitchen --down-> Cellar")
        print("=" * 76 + "\n")

    def _find_location(self, name: str):
        """BFS from current location to find a Location object by name."""
        visited = set()
        queue   = [self.current_location]
        while queue:
            loc = queue.pop(0)
            if loc.name in visited:
                continue
            visited.add(loc.name)
            if loc.name == name:
                return loc
            for nb in loc.connections.values():
                if nb.name not in visited:
                    queue.append(nb)
        return None

    # HELP 
    def _show_help(self):
        print("\n" + "-" * 52)
        print("  COMMANDS")
        print("-" * 52)
        print("  go [dir]  /  north / south / east / west / up / down")
        print("       Move to a connected room.")
        print("       [!] If an enemy is present, combat starts automatically.")
        print()
        print("  look         Re-describe the current room.")
        print("  map          Display a visual map of the manor.")
        print("  take [item]  Pick up an item in this room.")
        print("  fight        Manually engage an enemy (they also")
        print("               trigger automatically on room entry).")
        print("  inventory    List everything in your bag.  (alias: i)")
        print("  help         Show this help screen.")
        print("  quit         Exit the game.")
        print()
        print("  COMBAT ACTIONS:")
        print("    attack   Standard d20 attack.")
        print("    power    High-damage, slightly harder to land.")
        print("    heal     Recover some HP.")
        print()
        print("  GOAL:")
        print("    1. Grab the Ancient Tome  (Living Room)")
        print("    2. Grab the Silver Locket (Study / Library)")
        print("    3. Go to the Cellar and type  perform  to start the ritual")
        print("       -- the Shadow Figure will be summoned; defeat it")
        print("    4. Return to the Entry Hall to escape")
        print()
        print("  TIME: Each action costs 30 minutes (0.5 hr).")
        print("  You have 10 hours total (8 PM -> 6 AM).")
        print("-" * 52 + "\n")

    # END SCREENS ──
    def _show_game_over(self):
        print("\n" + "=" * 52)
        print("  G A M E   O V E R")
        print("=" * 52)
        print("  The manor wins. The darkness closes in.")
        print("=" * 52)

    def _show_fear_death(self):
        print("\n" + "=" * 52)
        print("  Y O U   D I E D   O F   F E A R")
        print("=" * 52)
        print("  The horrors of Blackwood Manor were too much")
        print("  to bear. Your heart seized. Your legs gave.")
        print("  They found you the next morning — eyes open,")
        print("  mouth frozen mid-scream. The curse lives on.")
        print("=" * 52)

    def _show_victory(self):
        print("\n" + "=" * 52)
        print("  V I C T O R Y")
        print("=" * 52)
        print("  The curse breaks. You escape into the night.")
        print("  Blackwood Manor stands silent once more.")
        print("=" * 52)

# =============================================================================
# Run the Game
# =============================================================================

if __name__ == "__main__":
    game = Game()
    game.start()

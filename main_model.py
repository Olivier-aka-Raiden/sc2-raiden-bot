import json
import math
import time
from enum import Enum

import tensorflow as tf
from typing import Dict, List, Set

from sc2.ids.effect_id import EffectId
from sc2.main import GameMatch, run_multiple_games
from loguru import logger
from sc2 import maps
from sc2 import position
from sc2.bot_ai import BotAI
from sc2.data import Race, AIBuild, Difficulty, Result
from sc2.ids.unit_typeid import *
from sc2.ids.upgrade_id import *
from sc2.ids.ability_id import *
from sc2.ids.buff_id import *
from consts import ALL_STRUCTURES, ATTACK_TARGET_IGNORE
import random
import cv2 as cv
import numpy as np

from sc2.player import Bot, Computer
from sc2.position import Point2
from itertools import chain
from sc2.unit import Unit
from sc2.units import Units

from pathing import Pathing

MINING_RADIUS = 1.325
HEADLESS = True


class ArmyComp(Enum):
    GROUND = 0
    AIR = 1


def _get_intersections(x0: float, y0: float, r0: float, x1: float, y1: float, r1: float) -> List[Point2]:
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        return []
    # One circle within other
    if d < abs(r0 - r1):
        return []
    # coincident circles
    if d == 0 and r0 == r1:
        return []
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return [Point2((x3, y3)), Point2((x4, y4))]


def get_intersections(p0: Point2, r0: float, p1: Point2, r1: float) -> List[Point2]:
    return _get_intersections(p0.x, p0.y, r0, p1.x, p1.y, r1)


class RaidenBot(BotAI):

    def __init__(self):
        self.opponent_name = None
        self.build_timing = 0
        self.last_highground_pylon_timing = 0
        self.proxy_ramp = None
        self.proxy_pos = None
        self.last_poke = 0
        self.enemy_natural = None
        self.roach_rush = False
        self.last_check = 0
        self.next_exp = None
        self.burrow_detected = None
        self.iteration = None
        self.third_proxy = None
        self.second_proxy = None
        self.proxy_pylon = False
        self.ordered_expands_locations = None
        self.pf_build = False
        self.ordered_enemy_expands_locations = None
        self.stalker_map_control_2 = None
        self.stalker_map_control_1 = None
        self.lastBuildChoice = None
        self.ITERATIONS_PER_MINUTE = 190
        self.MAX_WORKERS = 80
        self.tradeEfficiency = 100
        self.defensiveBehavior = True
        self.attacked_nexus = None
        self.lastAttack = 0
        self.armyValue = 0
        self.stopTradeTime = 0
        self.upgrade_time = 0
        self.expand_time = 0
        self.scout_time = 0
        self.visited_corner_index = 0
        self.armyValue = 0
        self.do_something_after = 0
        self.armyComp = None
        self.mainUnit = None
        self.photon_map = {}
        self.no_kiting_delay_map = {}
        self.cleared_all_expansion_map = {}
        self.scoutingObsTag = None
        self.followingObsTag = None
        self.squadLeaderTag = None
        self.scout_tag = None
        self.scouted = False
        self.zealot_tag = None
        self.pylonAtRamp = False
        self.rushDetected = False
        self.zerg_fast_exp_detected = False
        self.gatewayWallPos = None
        self.enemy_main_base_ramp = None
        self.forgeWallPos = None
        self.walled = False
        self.builder_tag = None
        self.zealot_move_to_wall = 0
        self.units_abilities = []
        self.train_data = []
        self.flipped = []
        self.availableUnitsAbilities = {}
        self.use_model = True
        self.pathing: Pathing = None
        self.selected_strategy_sequence = None
        self.selected_strategy_key = None
        self.mineral_target_dict: Dict[Point2, Point2] = {}
        if self.use_model:
            self.model = tf.keras.models.load_model("etc/BasicCNN-10-epochs-0.0001-LR-STAGE2")

    async def on_end(self, game_result):
        logger.info("--- on_end called ---")
        # Load existing strategy data
        strategy_data = {}
        try:
            with open("data/strategy.txt", "r") as f:
                strategy_data = json.load(f)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
        print(strategy_data)
        # Create or update strategy data for the opponent
        opponent_key = f"{self.opponent_id[:8]}_{self.opponent_name}"
        if opponent_key not in strategy_data:
            strategy_data[opponent_key] = {
                "strategies": {
                    "000": {"ratio": 0.5, "games": 0},
                    "001": {"ratio": 0.5, "games": 0},
                    "010": {"ratio": 0.5, "games": 0},
                    "011": {"ratio": 0.5, "games": 0},
                    "100": {"ratio": 0.5, "games": 0},
                    "101": {"ratio": 0.5, "games": 0},
                    "111": {"ratio": 0.5, "games": 0}
                },
                "total_games": 0
            }
        # Update the strategy based on the game result
        strategy = strategy_data[opponent_key]["strategies"][self.selected_strategy_key]
        strategy["games"] += 1
        # Update the total games played
        strategy_data[opponent_key]["total_games"] += 1
        # Calculate the new ratio based on game result and EWMA
        alpha = 0.2 - (strategy["games"] / (strategy_data[opponent_key]["total_games"] * 10))  # Adjust this alpha value as needed
        game_result_multiplier = 1 if game_result == Result.Victory else -1
        current_ratio = strategy["ratio"]
        new_ratio = current_ratio + alpha * (game_result_multiplier - current_ratio)

        # Update the new ratio
        strategy["ratio"] = new_ratio
        try:
            # Save updated strategy data
            with open("data/strategy.txt", "w") as f:

                json.dump(strategy_data, f, indent=4)
                f.flush()
        except Exception as e:
            print(f"An error occurred while writing in the file: {e}")
        with open("data/gameoutput-protoss.txt", "a") as f:
            if self.use_model:
                if game_result == Result.Victory:
                    logger.info(f"Model Victory !\n")
                    f.write(f"Model Victory !\n")
                else:
                    f.write(f"Model Defeat...\n")
                    logger.info(f"Model Defeat...\n")
            else:
                if game_result == Result.Victory:
                    f.write(f"raiden bot Victory !\n")
                else:
                    f.write(f"raiden bot Defeat...\n")

    async def on_step(self, iteration: int):
        self.iteration = iteration
        if self.iteration == 10 and self.opponent_name:
            await self.chat_send(f"gl, hf {self.opponent_name}!")
        if self.iteration == 1:
            self.greetings()
            self.calculate_locations()
            self.pathing = Pathing(self, False)
        self.calculate_targets()
        if self.enemy_race == Race.Terran and not (self.townhalls.ready.amount < 1 and self.units.of_type(UnitTypeId.PROBE).ready.amount < 1):
            await self.terran_opener()
            await self.select_scout_terran()
            await self.terran_photon_build_order()
        elif self.enemy_race == Race.Zerg and not (self.townhalls.ready.amount < 1 and self.units.of_type(UnitTypeId.PROBE).ready.amount < 1):
            await self.select_scout_zerg()
            await self.wall_build_order()
        elif self.enemy_race == Race.Protoss and not (self.townhalls.ready.amount < 1 and self.units.of_type(UnitTypeId.PROBE).ready.amount < 1):
            await self.stalker_early_poke()
        if self.iteration == 1:
            self.split_workers()
        elif not (self.townhalls.ready.amount < 1 and self.units.of_type(UnitTypeId.PROBE).ready.amount < 1):
            workers = self.get_mineral_workers()
            self.speedmine(workers)
        await self.manage_zealot_walling()
        await self.getAvailableAbilities()
        await self.computeTradeEfficiency()
        await self.handleScout()
        await self.scout()
        if self.iteration % 16 == 0:
            await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()
        await self.intel()
        await self.probeEscape()
        await self.upgrades()
        await self.computeTradeEfficiency()
        await self.followingObserver()
        await self.handleBlink()
        await self.handleHighTemplar()
        await self.manageChronoboost()
        await self.handleSentry()
        await self.handleKiting()
        await self.handleHarass()
        await self.handleVoidray()
        await self.build_defensive_structures()
        await self.on_enemy_main_base_destroyed()
        await self.cancel_buildings()
        await self.protoss_photon_micro()
        await self.idle_defense_behavior()
        await self.get_next_exp()

    def greetings(self):
        with open('data/botnames.txt', 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if parts[0] == self.opponent_id[0:8]:
                    self.opponent_name = parts[1]
            if not self.opponent_name:
                self.opponent_name = "unidentified"
        try:
            with open("data/strategy.txt", "r") as f:
                strategy_data = json.load(f)
        except json.decoder.JSONDecodeError:
            # Handle the case where the file is empty or not valid JSON
            strategy_data = {}
        # Create or update strategy data for the opponent
        opponent_key = f"{self.opponent_id[:8]}_{self.opponent_name}"
        if opponent_key not in strategy_data:
            strategy_data[opponent_key] = {
                "strategies": {
                    "000": {"ratio": 0.5, "games": 0},
                    "001": {"ratio": 0.5, "games": 0},
                    "010": {"ratio": 0.5, "games": 0},
                    "011": {"ratio": 0.5, "games": 0},
                    "100": {"ratio": 0.5, "games": 0},
                    "101": {"ratio": 0.5, "games": 0},
                    "111": {"ratio": 0.5, "games": 0}
                },
                "total_games": 0
            }
        # Save updated strategy data

        try:
            with open("data/strategy.txt", "w") as f:
                json.dump(strategy_data, f, indent=4)
                f.flush()
        except Exception as e:
            print(f"An error occurred while writing in the file: {e}")
        # Choose the strategy with the highest ratio
        # Get the maximum ratio
        max_ratio = max(strategy_data[opponent_key]["strategies"].values(), key=lambda strategy: strategy["ratio"])["ratio"]
        # Create a list of strategy keys with the maximum ratio
        max_ratio_strategy_keys = [key for key, strategy in strategy_data[opponent_key]["strategies"].items() if strategy["ratio"] == max_ratio]
        # Select a random strategy key from the list
        selected_strategy_key = random.choice(max_ratio_strategy_keys)
        print(f"Strategy selected : {selected_strategy_key}, ratio : {strategy_data[opponent_key]['strategies'][selected_strategy_key]['ratio']}")
        # Follow the sequence of the selected strategy
        self.selected_strategy_sequence = [int(bit) for bit in selected_strategy_key]
        self.selected_strategy_key = selected_strategy_key

    def calculate_locations(self):
        self.ordered_enemy_expands_locations = sorted(self.expansion_locations_list, key=lambda expansion: expansion.distance_to(self.enemy_start_locations[0]))
        self.ordered_expands_locations = sorted(self.expansion_locations_list, key=lambda expansion: expansion.distance_to(self.start_location))
        sortedRamps = sorted(self.game_info.map_ramps, key=lambda ramp: ramp.bottom_center.distance_to(self.enemy_start_locations[0]))
        self.enemy_main_base_ramp = sortedRamps[0]
        natural1 = position.Point2(position.Pointlike(self.ordered_enemy_expands_locations[1]))
        natural2 = position.Point2(position.Pointlike(self.ordered_enemy_expands_locations[2]))
        if natural1.distance_to(self.enemy_main_base_ramp.bottom_center) < natural2.distance_to(self.enemy_main_base_ramp.bottom_center):
            self.enemy_natural = natural1
        else:
            self.enemy_natural = natural2
        self.proxy_pos = self.enemy_natural.position.towards(self.start_location, 17)
        for ramp in sortedRamps:
            if ramp.bottom_center.distance_to(self.proxy_pos) < 7:
                self.proxy_ramp = ramp
                self.proxy_pos = ramp.bottom_center.towards(self.game_info.map_center, 6)
                break

    def calculate_targets(self):
        centers: List[Point2] = []

        for zone in self.townhalls:
            centers.append(zone.position)
        if len(centers) > 0:
            for mf in self.mineral_field:
                target: Point2 = mf.position
                center = target.closest(centers)
                target = target.towards(center, MINING_RADIUS)
                close = self.mineral_field.closer_than(MINING_RADIUS, target)
                for mf2 in close:
                    if mf2.tag != mf.tag:
                        points = get_intersections(mf.position, MINING_RADIUS, mf2.position, MINING_RADIUS)
                        if len(points) == 2:
                            target = center.closest(points)
                self.mineral_target_dict[mf.position] = target

    async def on_enemy_main_base_destroyed(self):
        main_base_destroyed = False
        for attack_unit in self.units.of_type({UnitTypeId.IMMORTAL,
                                               UnitTypeId.VOIDRAY,
                                               UnitTypeId.STALKER,
                                               UnitTypeId.SENTRY,
                                               UnitTypeId.ORACLE,
                                               UnitTypeId.HIGHTEMPLAR}).ready:
            if attack_unit.distance_to(self.enemy_start_locations[0]) < 3:
                main_base_destroyed = True
        if main_base_destroyed:
            for attack_unit in self.units.of_type({UnitTypeId.IMMORTAL,
                                                   UnitTypeId.VOIDRAY,
                                                   UnitTypeId.STALKER,
                                                   UnitTypeId.SENTRY,
                                                   UnitTypeId.ORACLE,
                                                   UnitTypeId.HIGHTEMPLAR}).ready:
                if not attack_unit.tag in self.cleared_all_expansion_map:
                    for i in range(len(self.ordered_enemy_expands_locations) - 1):
                        if i < 3:
                            continue
                        if i == 3:
                            attack_unit.attack(self.ordered_enemy_expands_locations[i])
                        elif self.ordered_enemy_expands_locations[i].distance_to(self.start_location) > 25:
                            attack_unit.attack(self.ordered_enemy_expands_locations[i], True)
                    self.cleared_all_expansion_map[attack_unit.tag] = True
        if self.iteration > 5000 and 0 < self.enemy_structures.amount < 4:
            for structure in self.enemy_structures:
                for unit in self.units.idle:
                    unit.attack(structure.position, True)

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id == UnitTypeId.NEXUS:
            mineral_field = self.mineral_field.closest_to(unit)
            if mineral_field is None:
                return
            unit(AbilityId.RALLY_WORKERS, mineral_field)
        if unit.type_id == UnitTypeId.GATEWAY:
            unit(AbilityId.RALLY_UNITS, self.main_base_ramp.protoss_wall_pylon)

    async def manage_zealot_walling(self):
        if self.enemy_race == Race.Zerg:
            if self.iteration > 1500 and self.structures.not_ready.exists:
                for structure in self.structures.not_ready:
                    if structure.distance_to(self.main_base_ramp.top_center) < 7 and not structure.type_id == UnitTypeId.PHOTONCANNON:
                        structure(AbilityId.CANCEL_BUILDINPROGRESS)
            if self.zealot_tag is not None:
                if not self.units.find_by_tag(self.zealot_tag):
                    self.zealot_tag = None
                    self.walled = False
                    return
            zealots = self.units(UnitTypeId.ZEALOT).ready
            if not zealots.exists:
                if self.structures(UnitTypeId.WARPGATE).ready.exists:
                    if self.can_afford(UnitTypeId.ZEALOT):
                        try:
                            pos = self.main_base_ramp.protoss_wall_warpin
                            placement = await self.find_placement(AbilityId.WARPGATETRAIN_ZEALOT, pos, placement_step=2)
                            if placement is None:
                                # return ActionResult.CantFindPlacementLocation
                                logger.info("can't warpin Zealot")
                        except:
                            return
                return
            closest_zealot = zealots.closest_to(self.main_base_ramp.protoss_wall_warpin)
            if not self.walled and self.zealot_tag is None:
                closest_zealot = zealots.closest_to(self.main_base_ramp.protoss_wall_warpin)
                self.zealot_tag = closest_zealot.tag
                closest_zealot.move(self.main_base_ramp.protoss_wall_warpin)
            if self.enemy_units.of_type(UnitTypeId.ZERGLING).closer_than(7, closest_zealot).exists:
                closest_zealot(AbilityId.HOLDPOSITION)
                self.walled = True
            elif (self.iteration - self.zealot_move_to_wall) > 10:
                closest_zealot.move(self.main_base_ramp.protoss_wall_warpin)
                self.walled = False
                self.zealot_move_to_wall = self.iteration

    async def cancel_buildings(self):
        if self.structures.not_ready.exists:
            for structure in self.structures.not_ready:
                if structure.health_percentage < 0.15 and structure.shield_percentage < 0.1:
                    structure(AbilityId.CANCEL_BUILDINPROGRESS)

    async def terran_photon_build_order(self):
        if not self.structures(UnitTypeId.PYLON).ready.exists and self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON) > 0:
            workers = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.scout_tag)
            if workers.amount > 0:
                worker = workers.closest_to(self.main_base_ramp.protoss_wall_pylon)
                worker.build(UnitTypeId.PYLON, self.main_base_ramp.protoss_wall_pylon)
                self.builder_tag = worker.tag
                self.pylonAtRamp = True
                return
        if not self.structures(UnitTypeId.PYLON).ready.exists and self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.PYLON) > 0:
            self.forgeWallPos = tuple(self.main_base_ramp.protoss_wall_buildings)[1]
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker:
                return
            else:
                worker.move(self.forgeWallPos)
        if not self.structures(UnitTypeId.FORGE).ready.exists and \
                self.can_afford(UnitTypeId.FORGE) and not self.already_pending(UnitTypeId.FORGE) > 0:
            self.forgeWallPos = tuple(self.main_base_ramp.protoss_wall_buildings)[1]
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker:
                return
            else:
                worker.build(UnitTypeId.FORGE, self.forgeWallPos)
            return

    async def wall_build_order(self):
        """if self.iteration < 500 and self.enemy_units(UnitTypeId.DRONE).exists:
            worker = self.units.find_by_tag(self.scout_tag)
            drone = self.enemy_units(UnitTypeId.DRONE).first
            if worker and (self.iteration - self.lastAttack) < 50:
                self.lastAttack = self.iteration
                worker.attack(drone)
            if worker and worker.distance_to(self.enemy_start_locations[1]) < 4 and not self.enemy_structures(UnitTypeId.HATCHERY).not_ready.exists and \
                    self.can_afford(UnitTypeId.PYLON) and drone.distance_to(worker) < 4:
                worker.build(UnitTypeId.PYLON, position.Point2((self.enemy_start_locations[1][0] + 1, self.enemy_start_locations[1][1] + 1)))
                if self.in_pathing_grid(position.Point2(self.enemy_natural.position.towards(self.start_location, 17))) and \
                        self.in_pathing_grid(position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                                              self.enemy_natural.position.towards(self.start_location, 17)[1] - 1))):
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.start_location, 17)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                            self.enemy_natural.position.towards(self.start_location, 17)[1] - 1)), True)
                else:
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.start_location, 12)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.start_location, 19)[0] + 1,
                                            self.enemy_natural.position.towards(self.start_location, 19)[1] + 1)), True)
            if worker and worker.distance_to(self.enemy_start_locations[2]) < 4 and not self.enemy_structures(UnitTypeId.HATCHERY).not_ready.exists and \
                    self.can_afford(UnitTypeId.PYLON) and drone.distance_to(worker) < 4:
                worker.build(UnitTypeId.PYLON, position.Point2((self.enemy_start_locations[2][0] + 1, self.enemy_start_locations[2][1] + 1)))
                if self.in_pathing_grid(position.Point2(self.enemy_natural.position.towards(self.start_location, 17))) and \
                        self.in_pathing_grid(position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                                              self.enemy_natural.position.towards(self.start_location, 17)[1] - 1))):
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.start_location, 17)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                            self.enemy_natural.position.towards(self.start_location, 17)[1] - 1)), True)
                else:
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.start_location, 12)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.start_location, 19)[0] + 1,
                                            self.enemy_natural.position.towards(self.start_location, 19)[1] + 1)), True)"""
        if self.iteration < 1000 and self.enemy_units(UnitTypeId.ROACH).exists:
            self.roach_rush = True
        if self.structures(UnitTypeId.GATEWAY).ready.exists and \
                self.can_afford(UnitTypeId.ZEALOT) and self.units(UnitTypeId.ZEALOT).ready.amount == 0 and \
                not self.already_pending(UnitTypeId.ZEALOT) > 0:
            gateways = self.structures(UnitTypeId.GATEWAY).ready
            if gateways.exists:
                gateways.first.train(UnitTypeId.ZEALOT)
                return
        if self.roach_rush:
            if self.structures.of_type({UnitTypeId.NEXUS, UnitTypeId.ROBOTICSFACILITY}).not_ready.exists:
                for structure in self.structures.of_type({UnitTypeId.NEXUS, UnitTypeId.ROBOTICSFACILITY}).not_ready:
                    structure(AbilityId.CANCEL_BUILDINPROGRESS)
            if self.enemy_units.of_type(UnitTypeId.ROACH).exists:
                roachs = self.enemy_units.of_type(UnitTypeId.ROACH)
                if roachs and roachs.closest_distance_to(self.start_location) < 17:
                    for probe in self.units(UnitTypeId.PROBE).ready:
                        if roachs.closest_distance_to(probe) < 1:
                            probe.attack(self.start_location)
                            self.lastAttack = self.iteration
                        else:
                            probe(AbilityId.SMART, self.mineral_field.closest_to(self.ordered_expands_locations[1]))
                            self.lastAttack = self.iteration
                elif roachs.closest_distance_to(self.start_location) > 30 and (self.iteration - self.lastAttack) > 80:
                    self.roach_rush = False
                    for probe in self.units(UnitTypeId.PROBE).ready:
                        if probe.is_attacking:
                            probe(AbilityId.SMART, self.mineral_field.closest_to(self.start_location))
            elif self.iteration > 1400 and (self.iteration - self.lastAttack) > 80:
                self.roach_rush = False
        if self.townhalls.ready.amount > 1 and self.builder_tag and (self.rushDetected is False or self.iteration > 1200):
            worker = self.units.find_by_tag(self.builder_tag)
            if worker:
                worker.move(self.main_base_ramp.bottom_center)
                self.builder_tag = None
        if self.enemy_structures(UnitTypeId.SPAWNINGPOOL).exists and \
                self.enemy_structures(UnitTypeId.SPAWNINGPOOL).first.health_percentage > 0.3 and \
                self.iteration < 400:
            self.rushDetected = True
        if self.enemy_structures(UnitTypeId.HATCHERY).not_ready.exists and \
                self.iteration < 400:
            self.zerg_fast_exp_detected = True
        if not self.structures(UnitTypeId.PYLON).ready.exists and self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON) > 0:
            workers = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.scout_tag)
            if workers.amount > 0:
                worker = workers.closest_to(self.main_base_ramp.protoss_wall_pylon)
                worker.build(UnitTypeId.PYLON, self.main_base_ramp.protoss_wall_pylon)
                self.builder_tag = worker.tag
                self.pylonAtRamp = True
                return
        if not self.structures(UnitTypeId.FORGE).ready.exists and \
                self.can_afford(UnitTypeId.FORGE) and not self.already_pending(UnitTypeId.FORGE) > 0:
            self.forgeWallPos = tuple(self.main_base_ramp.protoss_wall_buildings)[1]
            self.gatewayWallPos = tuple(self.main_base_ramp.protoss_wall_buildings)[0]
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker:
                return
            else:
                worker.build(UnitTypeId.FORGE, self.forgeWallPos)
                worker.move(tuple(self.main_base_ramp.protoss_wall_buildings)[0], True)
                worker(AbilityId.PATROL,
                       position.Point2((tuple(self.main_base_ramp.protoss_wall_buildings)[0][0] + 1,
                                        tuple(self.main_base_ramp.protoss_wall_buildings)[0][1] + 1)), True)
            return
        if self.structures(UnitTypeId.FORGE).exists and not self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.exists and \
                self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0:
            self.gatewayWallPos = tuple(self.main_base_ramp.protoss_wall_buildings)[0]
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker and self.units(UnitTypeId.PROBE).ready.exists:
                workers = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.scout_tag)
                if workers.amount > 0:
                    worker = workers.closest_to(self.main_base_ramp.protoss_wall_pylon)
            if worker:
                worker.build(UnitTypeId.GATEWAY, self.gatewayWallPos)
                worker.move(tuple(self.main_base_ramp.protoss_wall_buildings)[1].towards(self.main_base_ramp.protoss_wall_pylon, 2), True)
                worker(AbilityId.PATROL,
                       position.Point2((tuple(self.main_base_ramp.protoss_wall_buildings)[1][0] + 3,
                                        tuple(self.main_base_ramp.protoss_wall_buildings)[1][1] + 3)), True)
            return

        if self.rushDetected is True and not self.structures(UnitTypeId.PHOTONCANNON).ready.exists and self.structures(UnitTypeId.FORGE).ready.exists and \
                self.can_afford(UnitTypeId.PHOTONCANNON) and not self.already_pending(UnitTypeId.PHOTONCANNON) > 0:
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker and self.units(UnitTypeId.PROBE).ready.exists:
                workers = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.scout_tag)
                if workers.amount > 0:
                    worker = workers.closest_to(self.main_base_ramp.protoss_wall_pylon)
            if worker:
                worker.build(UnitTypeId.PHOTONCANNON, position.Point2(self.main_base_ramp.protoss_wall_pylon.towards(self.gatewayWallPos, 2)))
                worker.build(UnitTypeId.PHOTONCANNON, position.Point2(self.main_base_ramp.protoss_wall_pylon.towards(self.forgeWallPos, 2)), True)
                self.builder_tag = None
                return
        if self.rushDetected is True and self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and self.structures(UnitTypeId.FORGE).ready.exists and \
                self.can_afford(UnitTypeId.SHIELDBATTERY) and not self.already_pending(UnitTypeId.SHIELDBATTERY) > 0 and \
                not self.structures(UnitTypeId.SHIELDBATTERY).exists:
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker and self.units(UnitTypeId.PROBE).ready.exists:
                workers = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.scout_tag)
                if workers.amount > 0:
                    worker = workers.closest_to(self.main_base_ramp.protoss_wall_pylon)
            if worker:
                worker.build(UnitTypeId.SHIELDBATTERY, position.Point2(self.main_base_ramp.top_center.towards(self.gatewayWallPos, 5)))
                worker.build(UnitTypeId.SHIELDBATTERY, position.Point2(self.main_base_ramp.top_center.towards(self.forgeWallPos, 5)), True)
                self.builder_tag = None
                return
        if self.already_pending(UnitTypeId.GATEWAY) > 0 and self.builder_tag:
            worker = self.units.find_by_tag(self.builder_tag)
            if not worker:
                return
            else:
                worker(AbilityId.SMART, self.mineral_field.closest_to(self.start_location))
                self.builder_tag = None
                return
        if self.scouted and self.scout_tag and self.proxy_pylon:
            worker = self.units.find_by_tag(self.scout_tag)
            if worker and \
                    (worker.distance_to(self.enemy_natural.position) > 26):
                if self.structures(UnitTypeId.PYLON).ready.exists:
                    pylon = self.structures(UnitTypeId.PYLON).closest_to(self.ordered_enemy_expands_locations[0])
                    if pylon:
                        worker.move(pylon.position)
        if self.structures(UnitTypeId.PHOTONCANNON).ready.amount > 2 and self.proxy_pylon and self.proxy_ramp and (self.iteration - self.lastAttack) > 50:
            worker = self.units.find_by_tag(self.scout_tag)
            if worker:
                for cannon in self.structures(UnitTypeId.PHOTONCANNON).ready:
                    if cannon.shield_percentage < 0.8 and not self.enemy_units(UnitTypeId.QUEEN).exists:
                        worker.move(self.proxy_ramp.top_center.towards(self.game_info.map_center, 1))
                        worker(AbilityId.PATROL, self.proxy_ramp.top_center.towards(self.game_info.map_center, 3), True)
                        self.lastAttack = self.iteration

        if self.structures(UnitTypeId.PHOTONCANNON).ready.amount > 2 and self.proxy_pylon and self.second_proxy and (
                self.iteration - self.lastAttack) > 80:
            worker = self.units.find_by_tag(self.scout_tag)
            self.lastAttack = self.iteration
            if not worker and self.units(UnitTypeId.PROBE).ready.exists:
                return
            if worker:
                if self.enemy_units(UnitTypeId.DRONE).exists:
                    if self.enemy_units(UnitTypeId.DRONE).closest_to(worker).distance_to(worker) < 4 and self.can_afford(UnitTypeId.PYLON):
                        worker.build(UnitTypeId.PYLON, position.Point2((self.ordered_enemy_expands_locations[3][0] + 3,
                                                                        self.ordered_enemy_expands_locations[3][1] + 3)))
                        return
                    else:
                        worker.move(self.ordered_enemy_expands_locations[3])
                        worker(AbilityId.PATROL,
                               position.Point2((self.ordered_enemy_expands_locations[3][0] + 1,
                                                self.ordered_enemy_expands_locations[3][1] + 1)), True)
                        return
        if not self.structures(UnitTypeId.PHOTONCANNON).exists and self.structures(UnitTypeId.FORGE).not_ready.exists and \
                not self.already_pending(UnitTypeId.PYLON) > 0 and self.can_afford(UnitTypeId.PYLON) and not self.proxy_pylon and not self.second_proxy:
            worker = self.units.find_by_tag(self.scout_tag)
            if worker:
                target = self.enemy_natural
                worker.build(UnitTypeId.PYLON, self.proxy_pos)
                if self.proxy_ramp:
                    worker.move(self.proxy_ramp.bottom_center, True)
                    worker(AbilityId.PATROL, self.proxy_ramp.bottom_center.towards(self.start_location, 2), True)
                elif self.in_pathing_grid(position.Point2(target.position.towards(self.start_location, 17))) and \
                        self.in_pathing_grid(position.Point2((target.position.towards(self.start_location, 17)[0] - 1,
                                                              target.position.towards(self.start_location, 17)[1] - 1))):
                    worker.move(position.Point2(target.position.towards(self.start_location, 17)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((target.position.towards(self.start_location, 17)[0] - 1,
                                            target.position.towards(self.start_location, 17)[1] - 1)), True)
                else:
                    worker.move(position.Point2(target.position.towards(self.game_info.map_center, 19)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((target.position.towards(self.game_info.map_center, 19)[0] + 1,
                                            target.position.towards(self.game_info.map_center, 19)[1] + 1)), True)
                self.proxy_pylon = True
                return
        else:
            worker = self.units.find_by_tag(self.scout_tag)
            if worker and worker.is_idle:
                if self.in_pathing_grid(position.Point2(self.enemy_natural.position.towards(self.start_location, 17))) and \
                        self.in_pathing_grid(position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                                              self.enemy_natural.position.towards(self.start_location, 17)[1] - 1))):
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.start_location, 17)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.start_location, 17)[0] - 1,
                                            self.enemy_natural.position.towards(self.start_location, 17)[1] - 1)), True)
                    return
                else:
                    worker.move(position.Point2(self.enemy_natural.position.towards(self.game_info.map_center, 12)), True)
                    worker(AbilityId.PATROL,
                           position.Point2((self.enemy_natural.position.towards(self.game_info.map_center, 12)[0] + 1,
                                            self.enemy_natural.position.towards(self.game_info.map_center, 12)[1] + 1)), True)
                    return
        if self.structures(UnitTypeId.FORGE).exists and \
                not self.can_afford(UnitTypeId.PHOTONCANNON) and self.proxy_pylon and \
                self.structures(UnitTypeId.PYLON).ready.exists and self.structures(UnitTypeId.PYLON).ready.closer_than(40, self.enemy_natural).exists:
            worker = self.units.find_by_tag(self.scout_tag)
            if not worker:
                return
            if worker and worker.distance_to(self.enemy_natural) < 50 and worker.is_idle:
                if self.enemy_units(UnitTypeId.DRONE).exists:
                    if self.enemy_units(UnitTypeId.DRONE).closer_than(2, worker):
                        worker.move(self.computeScoutLocation(1, self.enemy_start_locations[0]))
                else:
                    worker.move(self.structures(UnitTypeId.PYLON).closest_to(worker))
                    return
        if self.structures(UnitTypeId.FORGE).exists and (self.iteration - self.build_timing) > 20 and \
                self.can_afford(UnitTypeId.PHOTONCANNON) and self.proxy_pylon and \
                self.structures(UnitTypeId.PYLON).ready.exists and self.structures(UnitTypeId.PYLON).ready.closer_than(40,
                                                                                                                       self.enemy_main_base_ramp.bottom_center).exists:
            worker = self.units.find_by_tag(self.scout_tag)
            if not worker:
                return
            if worker and worker.distance_to(self.enemy_natural) < 50:
                pylons = self.structures(UnitTypeId.PYLON).ready
                if pylons.exists:
                    pylon = pylons.closest_to(worker)
                    if pylon.distance_to(worker) < 40:
                        self.build_timing = self.iteration
                        target = pylon
                        worker.build(UnitTypeId.PHOTONCANNON, target.position.towards(self.start_location, 2))
                        worker.build(UnitTypeId.PHOTONCANNON, position.Point2((target.position[0] + 4, target.position[1])), True)
                        worker.build(UnitTypeId.PHOTONCANNON, position.Point2((target.position[0], target.position[1] - 4)), True)
                        worker.build(UnitTypeId.PHOTONCANNON, target.position.towards(self.enemy_natural, 2), True)
                        worker.move(position.Point2((target.position[0] + 2,
                                                     target.position[1] + 2)), True)
                        worker(AbilityId.PATROL,
                               position.Point2((target.position[0] + 3,
                                                target.position[1] + 3)), True)
                        self.second_proxy = True
                        return
                else:
                    return

    def split_workers(self) -> None:
        minerals = self.expansion_locations_dict[self.start_location].mineral_field.sorted_by_distance_to(self.start_location)
        self.close_minerals = {m.tag for m in minerals[0:4]}
        assigned: Set[int] = set()
        assigned.add(self.scout_tag)
        for i in range(self.workers.amount - 1):
            patch = minerals[i % len(minerals)]
            if i < len(minerals):
                worker = self.workers.tags_not_in(assigned).closest_to(patch)
            else:
                worker = self.workers.tags_not_in(assigned).furthest_to(patch)
            worker.gather(patch)
            assigned.add(worker.tag)

    async def stalker_early_poke(self):
        try:
            if self.enemy_race == Race.Terran or self.enemy_race == Race.Protoss:
                if self.units(UnitTypeId.STALKER).idle.exists and self.iteration < 1000 and (self.iteration - self.last_poke) > 50:
                    for stalker in self.units(UnitTypeId.STALKER).idle:
                        stalker.attack(self.enemy_main_base_ramp.bottom_center.towards(self.enemy_natural.position, 3))
                        self.last_poke = self.iteration

        except:
            return

    async def protoss_photon_micro(self):
        if self.iteration < 2000 and self.enemy_units(UnitTypeId.PROBE).exists:
            if self.enemy_structures.of_type({UnitTypeId.PYLON, UnitTypeId.PHOTONCANNON}).not_ready.exists:
                proxy_enemy_structures = self.enemy_structures.of_type({UnitTypeId.PYLON, UnitTypeId.PHOTONCANNON}).not_ready.closer_than(50, self.start_location)
                if proxy_enemy_structures.amount > 0:
                    for proxy in proxy_enemy_structures:
                        if proxy.tag not in self.photon_map:
                            self.photon_map[proxy.tag] = []
                        if len(self.photon_map[proxy.tag]) < 5:
                            probes = self.units(UnitTypeId.PROBE).ready
                            for probe in probes:
                                if len(self.photon_map[proxy.tag]) < 5 and not probe.is_attacking:
                                    probe.attack(proxy)
                                    probe.attack(proxy.position, True)
                                    self.photon_map[proxy.tag].append(probe.tag)
            if self.enemy_units(UnitTypeId.PROBE).closer_than(15, self.start_location).exists and (self.iteration - self.lastAttack) > 40:
                enemy_probe = self.enemy_units(UnitTypeId.PROBE).closer_than(15, self.start_location).first
                if self.units(UnitTypeId.PROBE).ready.exists:
                    probe = self.units(UnitTypeId.PROBE).ready.closest_to(enemy_probe)
                    self.scout_tag = probe.tag
                    probe.attack(enemy_probe.position)
                    probe.attack(self.ordered_expands_locations[1], True)
                    self.lastAttack = self.iteration
            if self.enemy_structures(UnitTypeId.PHOTONCANNON).ready.exists:
                for canon in self.enemy_structures(UnitTypeId.PHOTONCANNON).ready:
                    if canon.tag in self.photon_map:
                        del self.photon_map[canon.tag]
        if self.units(UnitTypeId.IMMORTAL).ready.exists:
            if self.enemy_structures(UnitTypeId.PHOTONCANNON).exists:
                if self.enemy_structures(UnitTypeId.PHOTONCANNON).closest_to(self.start_location).distance_to(self.start_location) < 40:
                    for immo in self.units(UnitTypeId.IMMORTAL).ready:
                        if immo.shield_percentage > 0.9:
                            immo.attack(self.enemy_structures(UnitTypeId.PHOTONCANNON).closest_to(self.main_base_ramp.top_center))
                        elif immo.shield_percentage < 0.1:
                            immo.move(self.start_location)
                            immo.attack(self.main_base_ramp.top_center, True)
        if self.units(UnitTypeId.STALKER).ready.exists:
            if self.enemy_structures(UnitTypeId.PHOTONCANNON).exists:
                for cannon in self.enemy_structures(UnitTypeId.PHOTONCANNON).ready:
                    near_stalkers = self.units(UnitTypeId.STALKER).closer_than(8, cannon)
                    if near_stalkers.amount > 0:
                        for stalker in near_stalkers:
                            if stalker.shield_percentage < 0.2:
                                stalker.move(stalker.position.towards(self.start_location, 6))
        if self.enemy_structures(UnitTypeId.PHOTONCANNON).ready.exists:
            for cannon in self.enemy_structures(UnitTypeId.PHOTONCANNON).ready:
                near_probes = self.units(UnitTypeId.PROBE).closer_than(9, cannon)
                if near_probes.amount > 0:
                    for probe in near_probes:
                        probe(AbilityId.SMART, self.mineral_field.closest_to(self.start_location))

    async def terran_opener(self):
        try:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.idle.exists and self.structures(UnitTypeId.GATEWAY).ready.idle.exists and self.armyComp == ArmyComp.AIR:
                gateway = self.structures(UnitTypeId.GATEWAY).ready.first
                if gateway and self.can_afford(UnitTypeId.STALKER) and not self.already_pending(UnitTypeId.STALKER) > 0 and self.units(
                        UnitTypeId.STALKER).ready.amount < 2:
                    gateway.train(UnitTypeId.STALKER)
            if self.units(UnitTypeId.STALKER).ready.exists:
                stalkers = self.units(UnitTypeId.STALKER).ready
                stalker_ctrl_1 = None
                stalker_ctrl_2 = None
                if stalkers.idle.amount > 1 and self.pf_build:
                    if self.stalker_map_control_1 and self.stalker_map_control_2:
                        stalker_ctrl_1 = self.units.find_by_tag(self.stalker_map_control_1)
                        stalker_ctrl_2 = self.units.find_by_tag(self.stalker_map_control_2)
                    if not stalker_ctrl_1 or not stalker_ctrl_2:
                        self.stalker_map_control_1 = None
                        availableStalker = stalkers.idle.filter(lambda unit: unit.tag != self.stalker_map_control_2)
                        if len(availableStalker) > 0:
                            self.stalker_map_control_1 = availableStalker.first.tag
                        availableStalker = stalkers.idle.filter(lambda unit: unit.tag != self.stalker_map_control_1)
                        if len(availableStalker) > 0:
                            self.stalker_map_control_2 = availableStalker.first.tag
                        return
                    if stalker_ctrl_1.position.distance_to(self.ordered_enemy_expands_locations[3]) > 5:
                        stalker_ctrl_1.attack(self.ordered_enemy_expands_locations[3])
                        stalker_ctrl_1(AbilityId.PATROL, position.Point2(
                            (self.ordered_enemy_expands_locations[3][0] + 1,
                             self.ordered_enemy_expands_locations[3][1] + 1)
                        ), True)
                    if stalker_ctrl_2.position.distance_to(self.ordered_enemy_expands_locations[4]) > 5:
                        stalker_ctrl_2.attack(self.ordered_enemy_expands_locations[4])
                        stalker_ctrl_2(AbilityId.PATROL, position.Point2(
                            (self.ordered_enemy_expands_locations[4][0] + 1,
                             self.ordered_enemy_expands_locations[4][1] + 1)
                        ), True)

        except:
            return

    @property
    def get_heal_spot(self) -> Point2:
        return self.pathing.find_closest_safe_spot(
            self.game_info.map_center, self.pathing.ground_grid
        )

    async def handle_unit_safety(self, unit: Unit, attack_target: Point2) -> None:
        grid: np.ndarray = self.pathing.ground_grid
        # pull back low health probe to heal
        if unit.shield_percentage < 0.5:
            unit.move(
                self.pathing.find_path_next_point(
                    unit.position, self.get_heal_spot, grid
                )
            )
            return

        close_enemies: Units = self.enemy_units.filter(
            lambda u: u.position.distance_to(unit) < 15.0
                      and not u.is_flying
                      and unit.type_id not in ATTACK_TARGET_IGNORE
        )

        # no target and in danger, run away
        if not self.pathing.is_position_safe(grid, unit.position):
            self.move_to_safety(unit, grid)
            return

        # get to the target
        if unit.distance_to(attack_target) > 5:
            # only make pathing queries if enemies are close
            if close_enemies:
                unit.move(
                    self.pathing.find_path_next_point(
                        unit.position, attack_target, grid
                    )
                )
            else:
                unit.move(attack_target)
        else:
            unit.attack(attack_target)

    def move_to_safety(self, unit: Unit, grid: np.ndarray):
        """
        Find a close safe spot on our grid
        Then path to it
        """
        safe_spot: Point2 = self.pathing.find_closest_safe_spot(unit.position, grid)
        move_to: Point2 = self.pathing.find_path_next_point(
            unit.position, safe_spot, grid
        )
        unit.move(move_to)

    async def select_scout_terran(self):
        if self.enemy_natural:
            if self.scout_tag is not None:
                if not self.units.find_by_tag(self.scout_tag):
                    return
            elif self.scout_tag is None:
                workers = self.units(UnitTypeId.PROBE)
                if workers.exists:
                    scout = workers.closest_to(self.enemy_natural)
                    self.scout_tag = scout.tag
                else:
                    return
            scout = self.units.find_by_tag(self.scout_tag)
            if not scout and self.builder_tag is not None:
                scout = self.units(UnitTypeId.PROBE).filter(lambda unit: unit.tag != self.builder_tag).random
            if scout:
                if not self.scouted and self.scout_tag and self.ordered_enemy_expands_locations:
                    self.scouted = True
                    scout.move(self.enemy_natural.position)
                    scout(AbilityId.PATROL, self.enemy_natural.position.towards(self.enemy_main_base_ramp.bottom_center, 2), True)
                if self.scouted and self.scout_tag and self.proxy_pylon and \
                        (scout.distance_to(self.enemy_natural.position) > 18 and scout.distance_to(self.mineral_field.closest_to(scout)) < 2):
                    scout.move(self.enemy_natural)
                if self.iteration > 300 and self.scouted and self.scout_tag and scout.distance_to(self.enemy_natural) > 22:
                    scout(AbilityId.SMART, self.mineral_field.closest_to(self.enemy_natural.position))
                if self.scouted and self.scout_tag and scout.shield_percentage < 0.9:
                    await self.handle_unit_safety(scout, self.enemy_main_base_ramp.top_center)

                if scout.distance_to(self.enemy_natural) < 6 and \
                        not self.proxy_pylon and self.can_afford(UnitTypeId.PYLON) and \
                        not self.already_pending(UnitTypeId.PYLON) > 0:
                    scout.build(UnitTypeId.PYLON, self.enemy_main_base_ramp.bottom_center
                                .towards(self.enemy_natural, 7)
                                .towards(self.mineral_field.closest_to(self.enemy_main_base_ramp.bottom_center), 3))
                    scout.build(UnitTypeId.PYLON, self.enemy_main_base_ramp.bottom_center
                                .towards(self.enemy_natural, 7)
                                .towards(self.mineral_field.closest_to(self.enemy_main_base_ramp.bottom_center), 1), True)
                    scout.build(UnitTypeId.PYLON, self.enemy_main_base_ramp.bottom_center
                                .towards(self.enemy_natural, 6), True)
                    scout.move(self.enemy_natural.position, True)
                    scout(AbilityId.PATROL, self.enemy_natural.position.towards(self.enemy_main_base_ramp.bottom_center, 2), True)
                    self.proxy_pylon = True
                if self.proxy_pylon and self.structures(UnitTypeId.PYLON).exists:
                    if self.structures(UnitTypeId.PYLON).ready.closer_than(30, scout).amount > 1:
                        self.second_proxy = True
                    else:
                        self.second_proxy = False
                if self.proxy_pylon and \
                        (self.already_pending(UnitTypeId.PHOTONCANNON) > 0 or self.structures(UnitTypeId.PHOTONCANNON).ready.amount > 0) and \
                        self.structures(UnitTypeId.FORGE).ready.exists:
                    if scout.is_idle:
                        scout.move(self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 1))
                        scout(AbilityId.PATROL, self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 3), True)
                        return
                if self.proxy_pylon and self.can_afford(UnitTypeId.PHOTONCANNON) and (self.iteration - self.build_timing) > 20 and \
                        (self.structures(UnitTypeId.PHOTONCANNON).not_ready.exists or self.structures(UnitTypeId.PHOTONCANNON).ready.exists) and \
                        self.structures(UnitTypeId.PHOTONCANNON).amount < 4 and self.structures(UnitTypeId.FORGE).ready.exists:
                    pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(scout)
                    photons = self.structures(UnitTypeId.PHOTONCANNON).not_ready
                    photon = None
                    if photons.exists and photons.first.health_percentage > 0.6:
                        photon = photons.first
                    elif self.structures(UnitTypeId.PHOTONCANNON).ready.exists:
                        photon = self.structures(UnitTypeId.PHOTONCANNON).ready.first
                    if pylon and photon and pylon.distance_to(scout) < 40:
                        self.build_timing = self.iteration
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 2)
                                    .towards(self.enemy_start_locations[0], 3))
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 2)
                                    .towards(self.enemy_start_locations[0], 1), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 2)
                                    .towards(self.enemy_start_locations[0], 4), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 4)
                                    .towards(self.enemy_start_locations[0], 6), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 4)
                                    .towards(self.enemy_start_locations[0], 5), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 5)
                                    .towards(self.enemy_start_locations[0], 5), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 5)
                                    .towards(self.enemy_start_locations[0], 6), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 2)
                                    .towards(self.enemy_start_locations[0], 2), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 5)
                                    .towards(self.enemy_start_locations[0], 5), True)
                        scout.build(UnitTypeId.PHOTONCANNON, self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 7)
                                    .towards(self.mineral_field.closest_to(self.enemy_main_base_ramp.bottom_center), 7), True)
                        scout.build(UnitTypeId.PHOTONCANNON, self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 7)
                                    .towards(self.mineral_field.closest_to(self.enemy_main_base_ramp.bottom_center), 6), True)
                        scout.build(UnitTypeId.PHOTONCANNON, self.enemy_main_base_ramp.bottom_center
                                    .towards(self.enemy_natural, 7)
                                    .towards(self.mineral_field.closest_to(self.enemy_main_base_ramp.bottom_center), 5), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.structures(UnitTypeId.PYLON).closest_to(scout).position
                                    .towards(self.enemy_start_locations[0], 5), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.structures(UnitTypeId.PYLON).closest_to(scout).position
                                    .towards(self.enemy_start_locations[0], 4), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.structures(UnitTypeId.PYLON).closest_to(scout).position
                                    .towards(self.enemy_start_locations[0], 3), True)
                        scout.build(UnitTypeId.PHOTONCANNON,
                                    self.structures(UnitTypeId.PYLON).closest_to(scout).position
                                    .towards(self.enemy_start_locations[0], 2), True)
                        scout.move(self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 1), True)
                        scout(AbilityId.PATROL, self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 3), True)
                    else:
                        if scout.is_idle:
                            scout.move(self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 1))
                            scout(AbilityId.PATROL, self.enemy_main_base_ramp.top_center.towards(self.enemy_main_base_ramp.bottom_center, 3), True)
                            return
                if self.proxy_pylon and not self.second_proxy and (self.iteration - self.last_highground_pylon_timing) > 30 and \
                        self.structures(UnitTypeId.PHOTONCANNON).amount > 1 and \
                        self.can_afford(UnitTypeId.PYLON):
                    self.iteration = self.last_highground_pylon_timing
                    scout.build(UnitTypeId.PYLON, self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 2))
                    scout.build(UnitTypeId.PYLON, position.Point2((self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 2)[0],
                                                                   self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 2)[1] + 1)), True)
                    scout.build(UnitTypeId.PYLON, position.Point2((self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 2)[0],
                                                                   self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 2)[1] - 1)), True)
                    scout.build(UnitTypeId.PYLON, self.enemy_main_base_ramp.top_center.towards(self.enemy_start_locations[0], 3), True)
                    return
                if self.proxy_pylon and self.can_afford(UnitTypeId.PHOTONCANNON) and \
                        self.structures(UnitTypeId.FORGE).ready.exists and not self.already_pending(UnitTypeId.PHOTONCANNON) > 0 and \
                        not self.structures(UnitTypeId.PHOTONCANNON).exists > 0:
                    pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(scout)
                    if pylon and pylon.distance_to(scout) < 40:
                        scout.build(UnitTypeId.PHOTONCANNON, pylon.position.towards(self.game_info.map_center, 2))
                        scout.build(UnitTypeId.PHOTONCANNON, pylon.position.towards(self.enemy_natural.position, 2), True)
                        scout.move(self.enemy_natural.position, True)
                        scout(AbilityId.PATROL, self.enemy_natural.position.towards(self.enemy_main_base_ramp.bottom_center, 2), True)
                    else:
                        return

    async def select_scout_zerg(self):
        if self.scout_tag is not None:
            if not self.units.find_by_tag(self.scout_tag):
                self.scout_tag = None
                return
        workers = self.units(UnitTypeId.PROBE)
        if not workers.exists:
            return
        """if self.scouted and self.scout_tag and (self.iteration - self.lastAttack) > 80:
            scout = self.units.find_by_tag(self.scout_tag)
            if scout and self.second_proxy and self.third_proxy and self.iteration > 1200 and self.structures(UnitTypeId.PHOTONCANNON).ready.amount > 1:
                if self.can_afford(UnitTypeId.PYLON):
                    scout.build(UnitTypeId.PYLON, self.ordered_enemy_expands_locations[4])
                scout.move(self.ordered_enemy_expands_locations[4], True)
                scout(AbilityId.PATROL, position.Point2(self.ordered_enemy_expands_locations[4]).towards(self.start_location, 1), True)
            if scout and self.can_afford(UnitTypeId.PYLON) and self.second_proxy and not self.third_proxy and self.iteration > 1200 and \
                    self.structures(UnitTypeId.PHOTONCANNON).ready.amount > 1:
                scout.build(UnitTypeId.PYLON, self.ordered_enemy_expands_locations[3])
                self.third_proxy = True
                self.lastAttack = self.iteration
                return
        """
        if self.scouted and self.scout_tag:
            scout = self.units.find_by_tag(self.scout_tag)
            if scout and self.iteration < 1500 and self.enemy_units(UnitTypeId.ZERGLING).amount > 18 and self.rushDetected is False:
                self.rushDetected = True
            if scout and self.iteration < 600 and self.enemy_units(UnitTypeId.ZERGLING).amount > 1 and self.rushDetected is False:
                self.rushDetected = True
                self.scout_tag = None
                scout.move(self.main_base_ramp.top_center)
                if self.structures(UnitTypeId.ASSIMILATOR).not_ready.exists:
                    for assimilator in self.structures(UnitTypeId.ASSIMILATOR).not_ready:
                        assimilator(AbilityId.CANCEL_BUILDINPROGRESS)
                if self.structures(UnitTypeId.PHOTONCANNON).not_ready.exists:
                    for canon in self.structures(UnitTypeId.PHOTONCANNON).not_ready:
                        canon(AbilityId.CANCEL_BUILDINPROGRESS)
                await self.build(UnitTypeId.PYLON, near=self.townhalls.ready.first)
            if scout and scout.shield_percentage < 0.4 and (self.iteration - self.lastAttack) > 50:
                if self.enemy_natural == self.ordered_enemy_expands_locations[1]:
                    scout(AbilityId.SMART, self.mineral_field.closest_to(self.ordered_enemy_expands_locations[2].position))
                    scout.move(self.enemy_natural.position.towards(self.game_info.map_center, 12), True)
                    self.lastAttack = self.iteration
                else:
                    scout(AbilityId.SMART, self.mineral_field.closest_to(self.ordered_enemy_expands_locations[1].position))
                    scout.move(self.enemy_natural.position.towards(self.game_info.map_center, 12), True)
                    self.lastAttack = self.iteration
        """for player in self.game_info.players:
            print(player.id)
            print(player.name)
            
            if player.name.__contains__("QueenBot"):
                closest_worker = workers.closest_to(position.Point2(position.Pointlike(self.enemy_start_locations[0])))
                self.scout_tag = closest_worker.tag
                #closest_worker.move(self.computeScoutLocation(1, self.enemy_start_locations[0]))
                #closest_worker.move(self.computeScoutLocation(2, self.enemy_start_locations[0]), True)
                #closest_worker.move(self.computeScoutLocation(3, self.enemy_start_locations[0]), True)
                closest_worker.move(self.enemy_natural.position.towards(self.game_info.map_center, 15))
                closest_worker(AbilityId.PATROL, position.Point2((self.enemy_natural.position.towards(self.game_info.map_center, 15)[0] + 1,
                                                                 self.enemy_natural.position.towards(self.game_info.map_center, 15)[1] + 1)), True)"""

        if not self.scouted and self.scout_tag is None and self.ordered_enemy_expands_locations:
            self.scouted = True
            closest_worker = workers.closest_to(position.Point2(position.Pointlike(self.enemy_start_locations[0])))
            self.scout_tag = closest_worker.tag
            # closest_worker.move(self.computeScoutLocation(1, self.enemy_start_locations[0]))
            # closest_worker.move(self.computeScoutLocation(2, self.enemy_start_locations[0]), True)
            # closest_worker.move(self.computeScoutLocation(3, self.enemy_start_locations[0]), True)
            closest_worker.move(self.enemy_natural.position.towards(self.game_info.map_center, 10))
            closest_worker(AbilityId.PATROL, position.Point2((self.enemy_natural.position.towards(self.game_info.map_center, 10)[0] + 1,
                                                              self.enemy_natural.position.towards(self.game_info.map_center, 10)[1] + 1)), True)
        """if not self.scouted and self.scout_tag and self.ordered_enemy_expands_locations:
            scout = self.units.find_by_tag(self.scout_tag)
            if scout:
                scout.move(self.enemy_natural.position)
                scout(AbilityId.PATROL, position.Point2((self.enemy_natural.position[0] + 1, self.enemy_natural.position[1] + 1)), True)
                self.scouted = True"""

    def get_mineral_workers(self) -> Units:
        def miner_filter(unit: Unit) -> bool:
            if unit.is_carrying_vespene:
                return False
            if unit.order_target is not None and isinstance(unit.order_target, int):
                try:
                    target_unit = self.mineral_field.by_tag(unit.order_target)
                except:
                    target_unit = None
                if target_unit is not None and target_unit.has_vespene:
                    return False
            return True

        units = self.units.gathering.filter(miner_filter)
        return units

    def speedmine(self, workers: Units):
        for worker in workers:
            self.speedmine_single(worker)

    def speedmine_single(self, worker: Unit):
        if self.townhalls.exists:
            townhall = self.townhalls.closest_to(worker)

            if worker.is_returning and len(worker.orders) == 1:
                target: Point2 = townhall.position
                target = target.towards(worker, townhall.radius + worker.radius)
                if 0.75 < worker.distance_to(target) < 2:
                    worker.move(target)
                    worker(AbilityId.SMART, townhall, True)
                    return

            if (
                    not worker.is_returning
                    and len(worker.orders) == 1
                    and isinstance(worker.order_target, int)
            ):
                try:
                    mf = self.mineral_field.by_tag(worker.order_target)
                except:
                    mf = None
                if mf is not None and mf.is_mineral_field:
                    target = self.mineral_target_dict.get(mf.position)
                    if 0.75 < worker.distance_to(target) < 2:
                        worker.move(target)
                        worker(AbilityId.SMART, mf, True)

    async def build_defensive_structures(self):
        try:
            if self.roach_rush and not self.structures(UnitTypeId.SHIELDBATTERY).ready.exists and not self.already_pending(UnitTypeId.SHIELDBATTERY) > 0 and \
                    self.can_afford(UnitTypeId.SHIELDBATTERY):
                builder = self.units.find_by_tag(self.builder_tag)
                if builder:
                    builder.build(UnitTypeId.SHIELDBATTERY, self.main_base_ramp.top_center.towards(self.start_location, 7))
            if self.iteration > 2000 and self.structures(UnitTypeId.FORGE).ready.exists:
                if self.already_pending(UnitTypeId.PHOTONCANNON) or self.already_pending(UnitTypeId.SHIELDBATTERY):
                    ongoing_structures = self.structures.of_type({UnitTypeId.PHOTONCANNON, UnitTypeId.SHIELDBATTERY}).not_ready
                    if ongoing_structures:
                        for structure in ongoing_structures:
                            if self.mineral_field.closest_to(structure) and self.mineral_field.closest_to(structure).distance_to(structure) < 2:
                                structure(AbilityId.CANCEL_BUILDINPROGRESS)
                if not self.already_pending(UnitTypeId.PYLON) > 0 and \
                        not self.already_pending(UnitTypeId.PHOTONCANNON) > 0 and \
                        not self.already_pending(UnitTypeId.SHIELDBATTERY) > 0:

                    for nexus in self.townhalls.ready:
                        if not nexus.distance_to(self.start_location) < 3:
                            if not self.structures(UnitTypeId.PYLON).closer_than(7, nexus).ready.exists:
                                mineral = self.mineral_field.closer_than(7, nexus).random
                                target = self.calculatePylonPos(5, nexus, mineral)
                                await self.build(UnitTypeId.PYLON, near=target)
                            elif not self.structures(UnitTypeId.PHOTONCANNON).closer_than(7, nexus).ready.exists:
                                await self.build(UnitTypeId.PHOTONCANNON, near=nexus)
                                return
                            elif not self.structures(UnitTypeId.SHIELDBATTERY).closer_than(7, nexus).ready.exists:
                                await self.build(UnitTypeId.SHIELDBATTERY, near=nexus)
                                return
            elif not self.structures(UnitTypeId.FORGE).ready.exists and not self.already_pending(UnitTypeId.FORGE) > 0 and self.iteration > 2000:
                if self.townhalls.exists:
                    await self.build(UnitTypeId.FORGE, near=self.townhalls.random)
        except:
            return

    async def manageChronoboost(self):
        try:
            for nexus in self.townhalls.ready:
                # Chrono nexus if cybercore is not ready, else chrono cybercore
                if self.iteration > 150 and not (self.enemy_race == Race.Protoss and self.structures(UnitTypeId.GATEWAY).ready.amount == 1) \
                        and not (self.enemy_race == Race.Terran and self.structures(UnitTypeId.GATEWAY).ready.amount == 1) and \
                        not (self.rushDetected is True and self.iteration < 600) and not 0 < self.already_pending_upgrade(
                    UpgradeId.WARPGATERESEARCH) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.BLINKTECH) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.CHARGE) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PSISTORMTECH) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.EXTENDEDTHERMALLANCE) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL1) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL2) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL3) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL1) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL2) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL3) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL1) < 1 and not 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL2) < 1 and not 0 < self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) < 1:
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not nexus.is_idle:
                        if nexus.energy >= 50:
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                elif self.enemy_race != Race.Zerg and \
                        self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.exists and \
                        self.already_pending(UnitTypeId.STALKER) and self.iteration < 1000:
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.GATEWAY).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.GATEWAY).ready.first)
                elif self.rushDetected is True and self.structures(UnitTypeId.GATEWAY).ready.exists and self.already_pending(UnitTypeId.ZEALOT) and self.iteration < 500:
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.GATEWAY).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.GATEWAY).ready.first)
                elif self.structures(UnitTypeId.STARGATE).ready.exists and self.already_pending(UnitTypeId.VOIDRAY):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.STARGATE).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.STARGATE).ready.first)
                elif self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and (0 < self.already_pending_upgrade(UpgradeId.BLINKTECH) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first)
                elif self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and (0 < self.already_pending_upgrade(UpgradeId.CHARGE) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first)
                elif self.structures(UnitTypeId.ROBOTICSBAY).ready.exists and (0 < self.already_pending_upgrade(UpgradeId.EXTENDEDTHERMALLANCE) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.ROBOTICSBAY).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.ROBOTICSBAY).ready.first)
                elif self.structures(UnitTypeId.TEMPLARARCHIVE).ready.exists and (0 < self.already_pending_upgrade(UpgradeId.PSISTORMTECH) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.TEMPLARARCHIVE).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.TEMPLARARCHIVE).ready.first)
                elif self.structures(UnitTypeId.FORGE).ready.exists and (
                        0 < self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) < 1 or 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) < 1 or 0 < self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.FORGE).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.FORGE).ready.first)
                elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and (
                        0 < self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) < 1 or 0 < self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL2) < 1 or 0 < self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) < 1):
                    if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50 and not self.structures(UnitTypeId.FORGE).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.CYBERNETICSCORE).ready.first)
        except:
            return

    def calculateOffensiveBlinkDest(self, stalker, enemy):
        enemy_position = enemy.position
        stalker_position = stalker.position
        offensive_location = stalker_position.towards(enemy_position, 6)
        return position.Point2(position.Pointlike(offensive_location))

    def calculateBlinkDest(self, stalker, enemy):
        enemy_position = enemy.position
        stalker_position = stalker.position
        targetBlinkPosition = (stalker_position[0] + stalker_position[0] - enemy_position[0], stalker_position[1] + stalker_position[1] - enemy_position[1])
        escape_location = stalker.position.towards(position.Point2(position.Pointlike(targetBlinkPosition)), 6)
        return position.Point2(position.Pointlike(escape_location))

    def calculateDodgeDest(self, unit, enemy):
        enemy_position = enemy.position
        unit_position = unit.position
        targetDodgePosition = (unit_position[0] + unit_position[0] - enemy_position[0], unit_position[1] + unit_position[1] - enemy_position[1])
        escape_location = unit.position.towards(position.Point2(position.Pointlike(targetDodgePosition)), 3)
        return position.Point2(position.Pointlike(escape_location))

    def calculatePylonPos(self, dist, nexus, mineral):
        mineral_position = mineral.position
        nexus_position = nexus.position
        targetPylonPosition = (nexus_position[0] + nexus_position[0] - mineral_position[0], nexus_position[1] + nexus_position[1] - mineral_position[1])
        targetPylonPosition = nexus.position.towards(position.Point2(position.Pointlike(targetPylonPosition)), dist)
        return position.Point2(position.Pointlike(targetPylonPosition))

    def computeScoutLocation(self, positionNumber, targetPosition):
        if positionNumber == 1:
            return position.Point2(position.Pointlike((targetPosition[0] - 7, targetPosition[1] - 7)))
        if positionNumber == 2:
            return position.Point2(position.Pointlike((targetPosition[0] + 8, targetPosition[1] - 8)))
        if positionNumber == 3:
            return position.Point2(position.Pointlike((targetPosition[0], targetPosition[1] + 7)))

    def calculateScoutEscape(self, scoutingObs, enemy):
        enemy_position = enemy.position
        scout_position = scoutingObs.position
        targetEscapePosition = (scoutingObs.position[0] + scout_position[0] - enemy_position[0], scout_position[1] + scout_position[1] - enemy_position[1])
        escape_location = scoutingObs.position.towards(position.Point2(position.Pointlike(targetEscapePosition)), 2)
        return position.Point2(position.Pointlike(escape_location))

    async def handleBlink(self):
        for stalker in self.units(UnitTypeId.STALKER).ready:
            if stalker.tag in self.availableUnitsAbilities.keys():
                if AbilityId.EFFECT_BLINK_STALKER in self.availableUnitsAbilities.get(stalker.tag, set()):
                    enemy = self.enemy_units.closest_to(stalker) if self.enemy_units.exists else None
                    if enemy:
                        if enemy.is_flying and enemy.distance_to(stalker) < 12 and self.units(UnitTypeId.STALKER).closer_than(5, stalker.position).exists:
                            if self.units(UnitTypeId.STALKER).closer_than(5, stalker.position).amount > 4:
                                for stalker_in_area in self.units(UnitTypeId.STALKER).closer_than(5, stalker.position):
                                    targetOffensiveBlinkPosition = self.calculateOffensiveBlinkDest(stalker_in_area, enemy)
                                    if stalker_in_area.in_ability_cast_range(AbilityId.EFFECT_BLINK_STALKER, targetOffensiveBlinkPosition) \
                                            and self.in_pathing_grid(targetOffensiveBlinkPosition):
                                        stalker_in_area(AbilityId.EFFECT_BLINK_STALKER, targetOffensiveBlinkPosition)
                                        stalker_in_area.attack(enemy, True)
                                break
                        if stalker.shield_percentage < 0.1 and stalker.is_attacking:
                            targetBlinkPosition = self.calculateBlinkDest(stalker, enemy)
                            if stalker.in_ability_cast_range(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition) and self.in_pathing_grid(targetBlinkPosition):
                                stalker(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition)

    def calculateStormCastPosition(self, enemy):
        enemy_position = enemy.position
        return position.Point2(position.Pointlike(enemy_position))

    async def handleHighTemplar(self):
        for ht in self.units(UnitTypeId.HIGHTEMPLAR).ready:
            if ht.tag in self.availableUnitsAbilities.keys():
                if AbilityId.PSISTORM_PSISTORM in self.availableUnitsAbilities.get(ht.tag, set()):
                    enemy = self.enemy_units.of_type({UnitTypeId.MARINE, UnitTypeId.MARAUDER}).closest_to(ht) if self.enemy_units.of_type(
                        {UnitTypeId.MARINE, UnitTypeId.MARAUDER}).exists else None
                    if enemy:
                        targetStormPosition = self.calculateStormCastPosition(enemy)
                        if ht.in_ability_cast_range(AbilityId.PSISTORM_PSISTORM, targetStormPosition):
                            ht(AbilityId.PSISTORM_PSISTORM, targetStormPosition)

    async def handleKiting(self):
        for effect in self.state.effects:
            if effect.id == EffectId.RAVAGERCORROSIVEBILECP or effect.id == EffectId.PSISTORMPERSISTENT:
                positions = effect.positions
                for unit in self.units:
                    for pos in positions:
                        if unit.position.distance_to(pos) < 3:
                            if self.enemy_units.exists:
                                enemy = self.enemy_units.closest_to(unit)
                                if enemy:
                                    unit.move(self.calculateDodgeDest(unit, enemy))
                                    break
                            else:
                                unit.move(unit.position.towards(position.Point2(position.Pointlike(self.start_location)), 3))
        for stalker in self.units(UnitTypeId.STALKER).ready:
            if self.iteration % 4 == 0:
                if not stalker.tag in self.no_kiting_delay_map:
                    self.no_kiting_delay_map[stalker.tag] = 0
                enemy_structure = self.enemy_structures.of_type({UnitTypeId.PLANETARYFORTRESS,
                                                                 UnitTypeId.PHOTONCANNON,
                                                                 UnitTypeId.SPINECRAWLER,
                                                                 UnitTypeId.BUNKER}).closest_to(stalker) if self.enemy_structures.of_type({UnitTypeId.PLANETARYFORTRESS,
                                                                                                                                           UnitTypeId.PHOTONCANNON,
                                                                                                                                           UnitTypeId.SPINECRAWLER,
                                                                                                                                           UnitTypeId.BUNKER}).exists else None
                if enemy_structure and stalker.shield_percentage < 0.5 and enemy_structure.distance_to(stalker) < 10:
                    self.pf_build = True
                    kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 6)
                    stalker.move(kite_location)
                offensive_unit_types = {UnitTypeId.ZEALOT,
                                        UnitTypeId.PROBE,
                                        UnitTypeId.STALKER,
                                        UnitTypeId.IMMORTAL,
                                        UnitTypeId.VOIDRAY,
                                        UnitTypeId.ARCHON,
                                        UnitTypeId.ADEPT,
                                        UnitTypeId.SCV,
                                        UnitTypeId.MARINE,
                                        UnitTypeId.MARAUDER,
                                        UnitTypeId.GHOST,
                                        UnitTypeId.SIEGETANK,
                                        UnitTypeId.BATTLECRUISER,
                                        UnitTypeId.THOR,
                                        UnitTypeId.DRONE,
                                        UnitTypeId.ROACH,
                                        UnitTypeId.HYDRALISK,
                                        UnitTypeId.ULTRALISK,
                                        UnitTypeId.BANELING,
                                        UnitTypeId.MUTALISK}
                enemy = self.enemy_units.of_type(offensive_unit_types).closest_to(stalker) if self.enemy_units.of_type(offensive_unit_types).exists else None
                if enemy and (self.iteration - self.no_kiting_delay_map[stalker.tag]) > 20 and not ((enemy.type_id == UnitTypeId.PROBE or
                                                                                                     enemy.type_id == UnitTypeId.SCV or
                                                                                                     enemy.type_id == UnitTypeId.DRONE) and
                                                                                                    self.enemy_units.closer_than(4, enemy).amount < 3):
                    if self.enemy_race == Race.Protoss:
                        if enemy.type_id == UnitTypeId.ZEALOT:
                            if enemy.distance_to(stalker) < 4 or (enemy.distance_to(stalker) < 6 and stalker.shield_percentage < 0.2) and \
                                    stalker.distance_to(self.start_location) > 6 and self.townhalls.ready.exists and \
                                    not stalker.is_facing(self.townhalls.ready.closest_to(self.start_location), 0.1):
                                kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 2)
                                second_kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 5)
                                if stalker.health_percentage < 0.3 and stalker.distance_to(self.main_base_ramp.top_center) > 6:
                                    stalker.move(self.main_base_ramp.top_center)
                                    self.no_kiting_delay_map[stalker.tag] = self.iteration
                                    continue
                                if self.in_pathing_grid(kite_location):
                                    stalker.move(kite_location)
                                    if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                        stalker.attack(enemy.position, True)
                                elif self.in_pathing_grid(second_kite_location):
                                    stalker.move(second_kite_location)
                                    if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                        stalker.attack(enemy.position, True)
                                elif stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                    self.no_kiting_delay_map[stalker.tag] = self.iteration
                                    stalker.move(self.main_base_ramp.top_center)
                        elif enemy.distance_to(stalker) < 6 or (enemy.distance_to(stalker) < 7 and stalker.shield_percentage < 0.2) and \
                                stalker.distance_to(self.start_location) > 6 and self.townhalls.ready.exists and \
                                not stalker.is_facing(self.townhalls.ready.closest_to(self.start_location), 0.1):
                            if stalker.health_percentage < 0.3 and stalker.shield_percentage < 0.1 and stalker.distance_to(self.main_base_ramp.top_center) > 6:
                                stalker.move(self.main_base_ramp.top_center)
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                continue
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 2)
                            second_kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 5)
                            if self.in_pathing_grid(kite_location):
                                stalker.move(kite_location)
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif self.in_pathing_grid(second_kite_location):
                                stalker.move(second_kite_location)
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                stalker.move(self.main_base_ramp.top_center)
                    if self.enemy_race == Race.Terran:
                        if enemy.distance_to(stalker) < 6 < stalker.distance_to(self.start_location) or \
                                (enemy.distance_to(stalker) < 6 and stalker.shield_percentage < 0.2) and \
                                not enemy.type_id == UnitTypeId.LIBERATOR and self.townhalls.ready.exists and \
                                not stalker.is_facing(self.townhalls.ready.closest_to(self.start_location), 0.1):
                            if stalker.health_percentage < 0.3 and stalker.shield_percentage < 0.1 and stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                stalker.move(self.main_base_ramp.top_center)
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                continue
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 2)
                            second_kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 5)
                            if self.in_pathing_grid(kite_location):
                                stalker.move(kite_location)
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif self.in_pathing_grid(second_kite_location):
                                stalker.move(stalker.position.towards(self.start_location, 6))
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                stalker.move(self.main_base_ramp.top_center)

                    if self.enemy_race == Race.Zerg:
                        if 1 < enemy.distance_to(stalker) < 7 and \
                                stalker.distance_to(self.start_location) > 6 and \
                                stalker.shield_percentage < 0.5 and self.townhalls.ready.exists and \
                                not stalker.is_facing(self.townhalls.ready.closest_to(self.start_location), 0.1):
                            if stalker.health_percentage < 0.3 and stalker.shield_percentage < 0.1 and stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                stalker.move(self.main_base_ramp.top_center)
                                continue
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 2)
                            second_kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 5)
                            if self.in_pathing_grid(kite_location):
                                stalker.move(kite_location)
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif self.in_pathing_grid(second_kite_location):
                                stalker.move(second_kite_location)
                                if stalker.health_percentage > 0.4 and stalker.shield_percentage > 0.1:
                                    stalker.attack(enemy.position, True)
                            elif stalker.distance_to(self.main_base_ramp.top_center) > 15:
                                self.no_kiting_delay_map[stalker.tag] = self.iteration
                                stalker.move(self.main_base_ramp.top_center)

                    if self.rushDetected is True and self.iteration < 2000:
                        if enemy.distance_to(stalker) < 3 < stalker.distance_to(self.main_base_ramp.protoss_wall_pylon):
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.main_base_ramp.protoss_wall_pylon)), 2)
                            stalker.move(kite_location)

        for supportUnit in self.units.of_type({UnitTypeId.SENTRY, UnitTypeId.IMMORTAL}).ready:
            if not supportUnit.tag in self.no_kiting_delay_map:
                self.no_kiting_delay_map[supportUnit.tag] = 0
            enemy = self.enemy_units.closest_to(supportUnit) if self.enemy_units.exists else None
            if enemy and (self.iteration - self.no_kiting_delay_map[supportUnit.tag]) > 20 and supportUnit.distance_to(self.start_location) > 6:
                if enemy.distance_to(supportUnit) < 6 < supportUnit.distance_to(self.start_location) and supportUnit.shield_percentage < 0.8:
                    kite_location = supportUnit.position.towards(self.start_location, 2)
                    second_kite_location = supportUnit.position.towards(self.start_location, 5)
                    if self.in_pathing_grid(kite_location):
                        supportUnit.move(kite_location)
                        if supportUnit.health_percentage > 0.4 and supportUnit.shield_percentage > 0.1:
                            supportUnit.attack(enemy.position, True)
                    elif self.in_pathing_grid(second_kite_location):
                        supportUnit.move(supportUnit.position.towards(self.start_location, 6))
                        if supportUnit.health_percentage > 0.4 and supportUnit.shield_percentage > 0.1:
                            supportUnit.attack(enemy.position, True)
                    elif supportUnit.distance_to(self.main_base_ramp.top_center) > 15:
                        self.no_kiting_delay_map[supportUnit.tag] = self.iteration
                        supportUnit.move(self.main_base_ramp.top_center)

        for colossy in self.units.of_type({UnitTypeId.COLOSSUS}).ready:
            if not colossy.tag in self.no_kiting_delay_map:
                self.no_kiting_delay_map[colossy.tag] = 0
            enemy = self.enemy_units.closest_to(colossy) if self.enemy_units.exists else None
            if enemy and (self.iteration - self.no_kiting_delay_map[colossy.tag]) > 20:
                if enemy.distance_to(colossy) < 7 and colossy.distance_to(self.start_location) > 4:
                    kite_location = colossy.position.towards(position.Point2(position.Pointlike(self.start_location)), 4)
                    if self.in_pathing_grid(kite_location):
                        colossy.move(kite_location)
                        if colossy.health_percentage > 0.4 and colossy.shield_percentage > 0.1:
                            colossy.attack(enemy.position, True)
                    elif colossy.distance_to(self.main_base_ramp.top_center) > 15:
                        self.no_kiting_delay_map[colossy.tag] = self.iteration
                        colossy.move(self.main_base_ramp.top_center)
                    else:
                        colossy.move(self.start_location)

        for casterUnit in self.units(UnitTypeId.HIGHTEMPLAR).ready:
            enemy = self.enemy_units.closest_to(casterUnit) if self.enemy_units.exists else None
            if enemy:
                if enemy.distance_to(casterUnit) < 7:
                    if casterUnit.health_percentage < 0.3:
                        casterUnit.move(self.main_base_ramp.top_center)
                        continue
                    kite_location = casterUnit.position.towards(position.Point2(position.Pointlike(self.start_location)), 6)
                    casterUnit.move(kite_location)
            for templar in self.units.of_type({UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR}):
                if templar.distance_to(casterUnit) < 4 and templar.energy_percentage < 0.2 and casterUnit.energy_percentage < 0.2:
                    templar(AbilityId.MORPH_ARCHON)

    async def handleSentry(self):
        if self.units(UnitTypeId.SENTRY).ready.exists:
            for sentry in self.units(UnitTypeId.SENTRY).ready:
                if self.enemy_units.amount > 0:
                    enemy = self.enemy_units.closest_to(sentry)
                    if enemy:
                        if not sentry.is_using_ability(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD) and sentry.distance_to(enemy) < 8 and \
                                not enemy.is_detector and not enemy.type_id == UnitTypeId.PROBE and \
                                not enemy.type_id == UnitTypeId.SCV and \
                                not enemy.type_id == UnitTypeId.DRONE and \
                                AbilityId.GUARDIANSHIELD_GUARDIANSHIELD in self.availableUnitsAbilities.get(sentry.tag, set()):
                            sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)

    async def handleVoidray(self):
        if self.units(UnitTypeId.VOIDRAY).ready.exists:
            voidrays = self.units(UnitTypeId.VOIDRAY).ready
            for voidray in voidrays:
                if self.enemy_units.amount > 0:
                    enemy = self.enemy_units.closest_to(voidray)
                    if enemy:
                        if not voidray.is_using_ability(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT) and voidray.distance_to(enemy) < 6 and enemy.is_armored and \
                                not enemy.is_detector and AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT in self.availableUnitsAbilities.get(voidray.tag, set()):
                            voidray(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT)
                if self.enemy_structures(UnitTypeId.PHOTONCANNON).closer_than(30, self.start_location).exists:
                    if voidray.shield_percentage > 0.9:
                        voidray.attack(self.enemy_structures(UnitTypeId.PHOTONCANNON).closest_to(self.main_base_ramp.top_center))
                    elif voidray.shield_percentage < 0.1:
                        voidray.move(self.start_location)
                        voidray.attack(self.main_base_ramp.top_center, True)

    async def handleScout(self):
        scoutingObs = None
        if self.scoutingObsTag:
            scoutingObs = self.units.find_by_tag(self.scoutingObsTag)
        if scoutingObs:
            enemies = self.enemy_units.closer_than(12, scoutingObs)
            enemy_structures = self.enemy_structures.closer_than(12, scoutingObs)
            if enemies.amount > 0:
                for enemy in enemies:
                    if enemy.is_detector:
                        targetEscapePosition = self.calculateScoutEscape(scoutingObs, enemy)
                        scoutingObs.move(targetEscapePosition)
            if enemy_structures.amount > 0:
                for enemy_structure in enemy_structures:
                    if enemy_structure.is_detector:
                        targetEscapePosition = self.calculateScoutEscape(scoutingObs, enemy_structure)
                        scoutingObs.move(targetEscapePosition)

    async def getAvailableAbilities(self):
        if self.units.ready:
            try:
                self.units_abilities = await self.get_available_abilities(self.units.ready)
                for abilities in self.units_abilities:
                    if abilities.__contains__(AbilityId.EFFECT_BLINK_STALKER) or \
                            abilities.__contains__(AbilityId.PSISTORM_PSISTORM) or \
                            abilities.__contains__(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD) or \
                            abilities.__contains__(AbilityId.BEHAVIOR_PULSARBEAMON) or \
                            abilities.__contains__(AbilityId.BEHAVIOR_PULSARBEAMOFF) or \
                            abilities.__contains__(AbilityId.ORACLEREVELATION_ORACLEREVELATION) or \
                            abilities.__contains__(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT):
                        self.availableUnitsAbilities = await self.client.query_available_abilities_with_tag(
                            self.units.of_type({UnitTypeId.IMMORTAL,
                                                UnitTypeId.VOIDRAY,
                                                UnitTypeId.STALKER,
                                                UnitTypeId.SENTRY,
                                                UnitTypeId.ORACLE,
                                                UnitTypeId.HIGHTEMPLAR}).ready)
            except:
                return

    async def computeTradeEfficiency(self):
        currentArmyValue = self.units.of_type({UnitTypeId.IMMORTAL,
                                               UnitTypeId.VOIDRAY,
                                               UnitTypeId.HIGHTEMPLAR,
                                               UnitTypeId.ARCHON,
                                               UnitTypeId.COLOSSUS,
                                               UnitTypeId.ZEALOT,
                                               UnitTypeId.STALKER,
                                               UnitTypeId.SENTRY,
                                               UnitTypeId.ORACLE,
                                               UnitTypeId.DARKTEMPLAR}).ready.amount
        if currentArmyValue == 0 or self.armyValue == 0:
            self.tradeEfficiency = 100
            self.stopTradeTime = self.iteration
        else:
            self.tradeEfficiency = (currentArmyValue * self.tradeEfficiency) / self.armyValue
            if self.tradeEfficiency > 100:
                self.tradeEfficiency = 100

        if self.tradeEfficiency < 60 and not self.defensiveBehavior:
            self.defensiveBehavior = True
            self.stopTradeTime = self.iteration
        elif (self.iteration - self.stopTradeTime) > 200 and self.defensiveBehavior:
            self.stopTradeTime = self.iteration
            self.tradeEfficiency = 100
            self.defensiveBehavior = False
        self.armyValue = currentArmyValue

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        #  FIXED THIS
        x += ((random.randrange(-20, 20)) / 100) * self.game_info.map_size[0]
        y += ((random.randrange(-20, 20)) / 100) * self.game_info.map_size[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))

        return go_to

    def send_observer_to_corner(self, observer):
        corners = [Point2((20, 20)),
                   Point2((self.game_info.map_size.x - 20, 20)),
                   Point2((20, self.game_info.map_size.y - 20)),
                   Point2((self.game_info.map_size.x - 20, self.game_info.map_size.y - 20)),
                   self.game_info.map_center]

        # Find the closest corner to the observer
        corner = corners[self.visited_corner_index]
        self.visited_corner_index += 1
        if self.visited_corner_index == 4:
            self.visited_corner_index = 0
        # Move the observer to the corner
        observer.move(corner)

    async def scout(self):
        scoutingObs = None
        if self.scoutingObsTag:
            scoutingObs = self.units.find_by_tag(self.scoutingObsTag)
        if self.units(UnitTypeId.OBSERVER).ready.amount > 0:
            if not scoutingObs:
                scoutingObs = self.units(UnitTypeId.OBSERVER).ready[0]
                self.scoutingObsTag = scoutingObs.tag
                self.scout_time = self.iteration
            if scoutingObs.is_idle and (self.iteration - self.scout_time) < 3000:
                enemy_location = self.enemy_start_locations[random.randrange(len(self.enemy_start_locations))]
                move_to = self.random_location_variance(enemy_location).position
                scoutingObs.move(move_to)
            elif scoutingObs.is_idle and self.enemy_units.amount == 0 and (self.iteration - self.scout_time) > 3000:
                self.send_observer_to_corner(scoutingObs)

    async def followingObserver(self):
        if self.units(UnitTypeId.OBSERVER).ready.amount == 2:
            followingObs = None
            if self.followingObsTag == self.scoutingObsTag:
                self.followingObsTag = None
            if self.followingObsTag:
                followingObs = self.units.find_by_tag(self.followingObsTag)
                if not followingObs:
                    self.followingObsTag = None
            if not self.followingObsTag:
                observers = self.units(UnitTypeId.OBSERVER).ready.filter(lambda unit: unit.tag != self.scoutingObsTag)
                self.followingObsTag = observers[0].tag
                followingObs = observers[0]
            if followingObs:
                if followingObs.is_idle and self.units(self.mainUnit).amount > 0:
                    if self.squadLeaderTag:
                        squadLeader = self.units.find_by_tag(self.squadLeaderTag)
                        friendly_unit = squadLeader.position if squadLeader else None
                        if not friendly_unit and self.units(self.mainUnit).ready.amount > 0:
                            friendly_unit = self.units(self.mainUnit).ready.closest_to(followingObs)
                        elif not friendly_unit:
                            friendly_unit = self.townhalls.closest_to(followingObs).position
                        move_to = friendly_unit
                        if followingObs.distance_to(friendly_unit) > 2:
                            followingObs.move(move_to)
        else:
            try:
                for rf in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0 and not self.already_pending(UnitTypeId.OBSERVER) > 0:
                        if not self.units(UnitTypeId.OBSERVER).ready:
                            rf.train(UnitTypeId.OBSERVER)
                        elif self.units(UnitTypeId.OBSERVER).ready.amount < 2:
                            rf.train(UnitTypeId.OBSERVER)
            except:
                return

    async def build_workers(self):
        if self.MAX_WORKERS > self.units(UnitTypeId.PROBE).amount and self.units(UnitTypeId.PROBE).amount < (24 * self.townhalls.amount):
            for nexus in self.townhalls.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT:[SIZE,(BGR COLOR)]
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.ROBOTICSFACILITY: [3, (200, 100, 0)],
            UnitTypeId.ROBOTICSBAY: [2, (200, 100, 0)],
            UnitTypeId.TWILIGHTCOUNCIL: [2, (200, 100, 0)],
            UnitTypeId.DARKSHRINE: [2, (200, 100, 0)],
            UnitTypeId.WARPGATE: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [4, (255, 100, 0)],
            UnitTypeId.IMMORTAL: [3, (120, 100, 0)],
            UnitTypeId.STALKER: [3, (120, 100, 50)],
            UnitTypeId.SENTRY: [2, (255, 192, 203)],
            UnitTypeId.ARCHON: [3, (255, 194, 50)],
            UnitTypeId.DARKTEMPLAR: [2, (241, 194, 50)],
            UnitTypeId.HIGHTEMPLAR: [2, (255, 192, 203)],
            UnitTypeId.ZEALOT: [2, (120, 100, 255)],
            UnitTypeId.ORACLE: [2, (241, 194, 50)]
        }
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv.circle(game_data, (int(pos[0]), int(pos[1])),
                          draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.enemy_units:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.enemy_units:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(UnitTypeId.OBSERVER).ready:
            pos = obs.position
            cv.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0
        if self.supply_cap == 0:
            population_ratio = 0
        else:
            population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0
        military_weight = 0
        if self.supply_cap - self.supply_left == 0:
            military_weight = 0
        else:
            military_weight = len(
                list(
                    set(chain.from_iterable(
                        [self.units(UnitTypeId.VOIDRAY).idle,
                         self.units(UnitTypeId.ARCHON).idle,
                         self.units(UnitTypeId.HIGHTEMPLAR).idle,
                         self.units(UnitTypeId.COLOSSUS).idle,
                         self.units(UnitTypeId.ZEALOT).idle,
                         self.units(UnitTypeId.IMMORTAL).idle,
                         self.units(UnitTypeId.STALKER).idle,
                         self.units(UnitTypeId.DARKTEMPLAR).idle,
                         self.units(UnitTypeId.ORACLE).idle])))) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv.flip(game_data, 0)

        if not HEADLESS:
            resized = cv.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv.imshow('Intel', resized)
            cv.waitKey(1)  # 1ms

    async def build_pylons(self):
        try:
            if self.structures(UnitTypeId.PYLON).ready.amount == 0 and not self.already_pending(UnitTypeId.PYLON) > 0:
                nexus = self.townhalls.random
                if nexus:
                    if self.can_afford(UnitTypeId.PYLON):
                        mineral = self.mineral_field.closer_than(7, nexus).random
                        target = self.calculatePylonPos(8, nexus, mineral)
                        await self.build(UnitTypeId.PYLON, near=target)
            elif not self.supply_used >= 194 and self.structures(UnitTypeId.CYBERNETICSCORE).amount > 0 and not self.can_feed(UnitTypeId.COLOSSUS) and \
                    (not self.already_pending(UnitTypeId.PYLON) > 0 or (self.iteration > 1500 and
                                                                        self.already_pending(UnitTypeId.PYLON) < 3)):
                nexus = self.townhalls.random
                if nexus:
                    if self.can_afford(UnitTypeId.PYLON):
                        mineral = self.mineral_field.closer_than(7, nexus).random
                        target = self.calculatePylonPos(8, nexus, mineral)
                        await self.build(UnitTypeId.PYLON, near=target)
        except:
            return

    async def build_assimilators(self):
        try:
            if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).exists and (self.enemy_race != Race.Terran or self.iteration > 400):
                for nexus in self.townhalls.ready:
                    if self.units(UnitTypeId.PROBE).closer_than(10, nexus).amount > 8:
                        vespenes = self.vespene_geyser.closer_than(15.0, nexus)
                        if vespenes:
                            for vespene in vespenes:
                                if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.already_pending(
                                        UnitTypeId.ASSIMILATOR) > 0:
                                    worker = self.select_build_worker(vespene.position, True)
                                    if worker is None:
                                        return
                                    if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                                        worker.build(UnitTypeId.ASSIMILATOR, vespene, True)
                                        worker.gather(self.mineral_field.closest_to(worker), True)
                                        return
        except:
            return

    async def expand(self):
        try:
            if self.already_pending(UnitTypeId.NEXUS) > 0.3:
                if self.townhalls.not_ready.first.health_percentage < 0.1:
                    self.townhalls.not_ready.first(AbilityId.CANCEL_BUILDINPROGRESS)
            if self.townhalls.ready.amount + 2 < (self.iteration / self.ITERATIONS_PER_MINUTE) * 2 and self.can_afford(
                    UnitTypeId.NEXUS) and self.townhalls.ready.amount < 15 and not self.already_pending(UnitTypeId.NEXUS) > 0 and \
                    (self.iteration - self.expand_time) > 400:
                await self.expand_now()
                self.expand_time = self.iteration
        except:
            return

    async def offensive_force_buildings(self):
        if self.structures(UnitTypeId.PYLON).ready.exists and self.townhalls.ready.exists:
            pylons = self.structures(UnitTypeId.PYLON).ready.filter(lambda p: p.distance_to(self.townhalls.ready.closest_to(p)) < 20)
            pylon = None
            if pylons.amount > 0:
                pylon = pylons.random
            if pylon:
                try:
                    if self.armyComp == ArmyComp.AIR:
                        if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.exists and \
                                not self.already_pending(UnitTypeId.CYBERNETICSCORE) > 0 and \
                                not self.structures(UnitTypeId.CYBERNETICSCORE).exists and \
                                not (self.rushDetected is True and not self.structures(UnitTypeId.PHOTONCANNON).exists):
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        elif self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.amount < 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0 and \
                                    not self.enemy_race == Race.Zerg and not (self.enemy_race == Race.Terran and self.iteration < 400):
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.STARGATE).amount + 3 < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.structures(
                                    UnitTypeId.STARGATE).ready.amount < 10 and self.structures(UnitTypeId.STARGATE).ready.idle.amount == 0:
                                if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE) > 0 or \
                                        (self.iteration > 3000 and self.already_pending(UnitTypeId.STARGATE) > 0 and
                                         self.structures(UnitTypeId.STARGATE).not_ready.amount < 3):
                                    await self.build(UnitTypeId.STARGATE, near=pylon)
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount < 1 and self.roach_rush is False:
                                if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(UnitTypeId.ROBOTICSFACILITY) > 0:
                                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
                    if self.armyComp == ArmyComp.GROUND:
                        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and \
                                not self.structures(UnitTypeId.TEMPLARARCHIVE).ready.exists and \
                                self.enemy_race == Race.Terran and \
                                self.iteration > 2000 and not self.already_pending(UnitTypeId.TEMPLARARCHIVE) > 0:
                            if self.structures(UnitTypeId.TEMPLARARCHIVE).ready.amount < 1:
                                if self.can_afford(UnitTypeId.TEMPLARARCHIVE):
                                    await self.build(UnitTypeId.TEMPLARARCHIVE, near=pylon)
                        elif self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and \
                                not self.structures(UnitTypeId.DARKSHRINE).ready.exists and \
                                self.enemy_race == Race.Protoss and \
                                self.iteration > 2000 and not self.already_pending(UnitTypeId.DARKSHRINE) > 0:
                            if self.structures(UnitTypeId.DARKSHRINE).ready.amount < 1:
                                if self.can_afford(UnitTypeId.DARKSHRINE):
                                    await self.build(UnitTypeId.DARKSHRINE, near=pylon)
                        elif self.structures(UnitTypeId.ROBOTICSFACILITY).ready.exists and \
                                not self.structures(UnitTypeId.ROBOTICSBAY).ready.exists and \
                                self.iteration > 3000 and not self.already_pending(UnitTypeId.ROBOTICSBAY) > 0 and self.units(UnitTypeId.PROBE).amount > 60:
                            if self.can_afford(UnitTypeId.ROBOTICSBAY) and self.structures(UnitTypeId.ROBOTICSBAY).amount == 0:
                                await self.build(UnitTypeId.ROBOTICSBAY, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and not self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and \
                                not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL) > 0 and self.units(UnitTypeId.STALKER).exists:
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.amount < 1:
                                if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL):
                                    await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and not self.structures(UnitTypeId.ROBOTICSFACILITY).ready.exists and \
                                not self.already_pending(UnitTypeId.ROBOTICSFACILITY) > 0:
                            if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount < 1:
                                if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(UnitTypeId.ROBOTICSFACILITY) > 0:
                                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and self.structures(
                                UnitTypeId.FORGE).ready.amount < 1 and self.iteration > 1400 and not self.already_pending(UnitTypeId.FORGE) > 0:
                            if self.structures(UnitTypeId.FORGE).ready.amount < 1:
                                if self.can_afford(UnitTypeId.FORGE):
                                    await self.build(UnitTypeId.FORGE, near=pylon)
                        elif self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE).exists and \
                                not self.already_pending(UnitTypeId.CYBERNETICSCORE) > 0 and \
                                not (self.rushDetected is True and not self.structures(UnitTypeId.PHOTONCANNON).exists):
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        if (self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).amount < (self.townhalls.ready.amount + 2) and
                            self.iteration > 1500 and
                            self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists) or \
                                (self.enemy_race == Race.Terran and
                                 self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).amount < (self.townhalls.ready.amount + 2) and
                                 self.iteration > 800):
                            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        elif self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.amount < 1 and not self.enemy_race == Race.Zerg and \
                                not (self.enemy_race == Race.Terran and self.iteration < 400):
                            if self.can_afford(UnitTypeId.GATEWAY) and \
                                    not self.already_pending(UnitTypeId.GATEWAY) > 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        else:
                            if self.iteration > 1500 and self.can_afford(UnitTypeId.PYLON) and not self.pylonAtRamp:
                                try:
                                    await self.build(UnitTypeId.PYLON, near=self.main_base_ramp.protoss_wall_pylon)
                                    self.pylonAtRamp = True
                                except:
                                    self.pylonAtRamp = False
                                    return
                except:
                    return

    async def build_offensive_force(self):
        try:
            if self.armyComp == ArmyComp.AIR:
                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.VOIDRAY) and sg.is_powered and self.supply_left > 0:
                        if self.iteration > 2500 and not self.units(UnitTypeId.ORACLE).exists and \
                                not self.already_pending(UnitTypeId.ORACLE) > 0 and self.units(UnitTypeId.ORACLE).amount < 2:
                            sg.train(UnitTypeId.ORACLE)
                        else:
                            sg.train(UnitTypeId.VOIDRAY)
                for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.COLOSSUS) and rb.is_powered and self.supply_left > 4 and self.units(UnitTypeId.IMMORTAL).amount > 4 and \
                            self.research(UpgradeId.EXTENDEDTHERMALLANCE) and self.units(UnitTypeId.COLOSSUS).ready.amount < 4:
                        rb.train(UnitTypeId.COLOSSUS)
                    elif self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 4:
                        rb.train(UnitTypeId.IMMORTAL)
                for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                    if self.structures(UnitTypeId.PYLON).ready.exists:
                        pylon = self.structures(UnitTypeId.PYLON).ready.random
                        if pylon:
                            pos = pylon.position.to2.random_on_distance(4)
                            placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=2)
                            if placement is None:
                                # return ActionResult.CantFindPlacementLocation
                                logger.info("can't place")
                                break
                            if self.can_afford(UnitTypeId.ZEALOT) and (self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY}).amount > 10 or \
                                                                       self.already_pending(UnitTypeId.VOIDRAY) > 4) and self.iteration > 2500:
                                warpgate.warp_in(UnitTypeId.ZEALOT, placement)
            if self.armyComp == ArmyComp.GROUND:
                cyberList = self.structures(UnitTypeId.CYBERNETICSCORE).ready
                if cyberList.idle.amount > 0:
                    cyber = cyberList.ready.idle.random
                    if cyber and self.can_afford(UpgradeId.WARPGATERESEARCH) and self.already_pending_upgrade(
                            UpgradeId.WARPGATERESEARCH) == 0 and cyber.is_powered and self.units(UnitTypeId.STALKER).ready.amount > 1:
                        cyber.research(UpgradeId.WARPGATERESEARCH)
                for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
                    if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 1:
                        gateway(AbilityId.MORPH_WARPGATE)
                    if self.can_afford(UnitTypeId.STALKER) and gateway.is_powered and cyberList.amount > 0 and not self.research(
                            UpgradeId.WARPGATERESEARCH) and self.supply_left > 0:
                        if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) < 1:
                            gateway.train(UnitTypeId.STALKER)
                for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                    if self.structures(UnitTypeId.PYLON).ready.exists:
                        pylon = self.structures(UnitTypeId.PYLON).ready.random
                        if pylon:
                            pos = pylon.position.to2.random_on_distance(4)
                            placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=2)
                            if placement is None:
                                # return ActionResult.CantFindPlacementLocation
                                logger.info("can't place")
                                break
                            if self.research(UpgradeId.PSISTORMTECH) and self.units(UnitTypeId.HIGHTEMPLAR).ready.amount < 3 and \
                                    self.units(self.mainUnit).ready.amount > 10 and self.can_afford(UnitTypeId.HIGHTEMPLAR):
                                warpgate.warp_in(UnitTypeId.HIGHTEMPLAR, placement)
                            if not self.units(UnitTypeId.DARKTEMPLAR).exists and self.structures(UnitTypeId.DARKSHRINE).ready.exists and \
                                    self.units(UnitTypeId.DARKTEMPLAR).amount < 2 and self.can_afford(UnitTypeId.DARKTEMPLAR) and \
                                    self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY}).amount > 10:
                                warpgate.warp_in(UnitTypeId.DARKTEMPLAR, placement)
                            if not self.units(UnitTypeId.SENTRY).exists and self.can_afford(UnitTypeId.SENTRY) \
                                    and self.units(UnitTypeId.SENTRY).amount < 1 and self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY}).amount > 10:
                                warpgate.warp_in(UnitTypeId.SENTRY, placement)
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and self.can_afford(UnitTypeId.STALKER):
                                warpgate.warp_in(UnitTypeId.STALKER, placement)
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and self.can_afford(UnitTypeId.ZEALOT) and \
                                    self.iteration > 2500 and self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY}).amount > 10:
                                warpgate.warp_in(UnitTypeId.ZEALOT, placement)
                for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.COLOSSUS) and rb.is_powered and self.supply_left > 4 and \
                            self.research(UpgradeId.EXTENDEDTHERMALLANCE) and self.units(UnitTypeId.COLOSSUS).ready.amount < 4 and \
                            self.units(UnitTypeId.IMMORTAL).ready.amount > 3:
                        rb.train(UnitTypeId.COLOSSUS)
                    elif self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 4 and self.units(UnitTypeId.IMMORTAL).ready.amount < 4:
                        rb.train(UnitTypeId.IMMORTAL)
                if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                    if self.already_pending_upgrade(UpgradeId.BLINKTECH) == 0 and self.iteration > 1000:
                        self.research(UpgradeId.BLINKTECH)
                    if self.already_pending_upgrade(UpgradeId.CHARGE) == 0 and self.iteration > 2500:
                        self.research(UpgradeId.CHARGE)
                if self.structures(UnitTypeId.TEMPLARARCHIVE).exists:
                    if self.already_pending_upgrade(UpgradeId.PSISTORMTECH) == 0:
                        self.research(UpgradeId.PSISTORMTECH)
                if self.structures(UnitTypeId.ROBOTICSBAY).exists:
                    if self.already_pending_upgrade(UpgradeId.EXTENDEDTHERMALLANCE) == 0:
                        self.research(UpgradeId.EXTENDEDTHERMALLANCE)
        except:
            return

    async def handleHarass(self):
        if self.units.of_type({UnitTypeId.DARKTEMPLAR, UnitTypeId.ORACLE}).ready.exists:
            oracles = self.units(UnitTypeId.ORACLE).ready.idle
            if oracles:
                for oracle in oracles:
                    if self.enemy_units.exists and self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).ready.amount > 0 and \
                            self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).closer_than(7, oracle).exists:
                        enemyWorkers = self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).closer_than(7, oracle)
                        if oracle.tag in self.availableUnitsAbilities.keys():
                            if AbilityId.BEHAVIOR_PULSARBEAMON in self.availableUnitsAbilities.get(oracle.tag, set()):
                                oracle(AbilityId.BEHAVIOR_PULSARBEAMON)
                                if enemyWorkers and enemyWorkers.amount > 0:
                                    first = None
                                    for enemyWorker in enemyWorkers:
                                        if not first:
                                            first = enemyWorker
                                            oracle.smart(enemyWorker)
                                        oracle.smart(enemyWorker, True)
                    else:
                        if AbilityId.BEHAVIOR_PULSARBEAMOFF in self.availableUnitsAbilities.get(oracle.tag, set()):
                            oracle(AbilityId.BEHAVIOR_PULSARBEAMOFF)

            dts = self.units(UnitTypeId.DARKTEMPLAR).ready.idle
            if dts:
                for dt in dts:
                    if self.enemy_units.exists and self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).ready.amount > 0 and \
                            self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).closer_than(5, dt).exists:
                        enemyWorkers = self.enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).closer_than(7, dt)
                        if enemyWorkers and enemyWorkers.amount > 0:
                            first = None
                            for enemyWorker in enemyWorkers:
                                if not first:
                                    first = enemyWorker
                                    dt.smart(enemyWorker)
                                dt.smart(enemyWorker, True)

    def harass(self):
        ordered_enemy_expands_locations = sorted(self.expansion_locations_list, key=lambda expansion: expansion.distance_to(self.enemy_start_locations[0]))
        if self.units.of_type({UnitTypeId.DARKTEMPLAR, UnitTypeId.ORACLE}).ready.exists:
            oracles = self.units(UnitTypeId.ORACLE).ready.idle
            if oracles:
                for oracle in oracles:
                    oracle.move(Point2((self.enemy_start_locations[0].x, self.start_location.y)), True)
                    if ordered_enemy_expands_locations[1].distance_to(Point2((self.enemy_start_locations[0].x, self.start_location.y))) < \
                            ordered_enemy_expands_locations[2].distance_to(Point2((self.enemy_start_locations[0].x, self.start_location.y))):
                        oracle.move(ordered_enemy_expands_locations[1], True)
                    else:
                        oracle.move(ordered_enemy_expands_locations[2], True)
            dts = self.units(UnitTypeId.DARKTEMPLAR).ready.idle
            if dts:
                for dt in dts:
                    dt.move(Point2((self.enemy_start_locations[0].x, self.start_location.y)), True)
                    if ordered_enemy_expands_locations[1].distance_to(Point2((self.enemy_start_locations[0].x, self.start_location.y))) < \
                            ordered_enemy_expands_locations[2].distance_to(Point2((self.enemy_start_locations[0].x, self.start_location.y))):
                        dt.move(ordered_enemy_expands_locations[1], True)
                    else:
                        dt.move(ordered_enemy_expands_locations[2], True)

    async def detect_zerg_burrow(self):
        if self.enemy_race == Race.Zerg:
            if self.enemy_units.exists:
                for enemy_unit in self.enemy_units:
                    if enemy_unit.is_burrowed:
                        self.burrow_detected = True

    def find_target(self):
        if self.enemy_units.amount > 0:
            if self.squadLeaderTag:
                squadLeader = self.units.find_by_tag(self.squadLeaderTag)
                if squadLeader:
                    return self.enemy_units.closest_to(squadLeader).position
                else:
                    return self.enemy_units.random.position
            else:
                return self.enemy_units.random.position
        elif self.enemy_structures.amount > 0 and self.iteration > 4000:
            return self.enemy_structures.random.position
        else:
            return self.enemy_start_locations[0]

    async def upgrades(self):
        try:
            if self.armyComp == ArmyComp.AIR:
                if self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0 and self.iteration > 2000:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) > 0.4 and len(self.structures(UnitTypeId.FLEETBEACON).ready) < 1:
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    if pylon and self.can_afford(UnitTypeId.FLEETBEACON) and not self.already_pending(UnitTypeId.FLEETBEACON) > 0:
                        await self.build(UnitTypeId.FLEETBEACON, near=pylon)
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0 and self.upgrade_time != 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL1) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRARMORSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL2) == 0 and (self.iteration - self.upgrade_time) > 200:
                    if self.research(UpgradeId.PROTOSSAIRARMORSLEVEL2):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRARMORSLEVEL3):
                        self.upgrade_time = self.iteration
            if self.armyComp == ArmyComp.GROUND:
                if self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0 and self.iteration > 2000:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) > 0.4 and len(self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready) < 1:
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    if pylon and self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL) > 0:
                        await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0 and self.upgrade_time != 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDARMORSLEVEL1) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDARMORSLEVEL2) == 0 and (self.iteration - self.upgrade_time) > 200:
                    if self.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL2):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDARMORSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 200:
                    if self.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL3):
                        self.upgrade_time = self.iteration
        except:
            return

    async def probeEscape(self):
        offensive_units = {UnitTypeId.ZEALOT,
                           UnitTypeId.ZERGLING,
                           UnitTypeId.ROACH,
                           UnitTypeId.STALKER,
                           UnitTypeId.IMMORTAL,
                           UnitTypeId.HYDRALISK,
                           UnitTypeId.ADEPT,
                           UnitTypeId.DARKTEMPLAR,
                           UnitTypeId.SENTRY,
                           UnitTypeId.MARAUDER,
                           UnitTypeId.MARINE}
        if self.rushDetected is True:
            if self.enemy_units.of_type({UnitTypeId.ZERGLING, UnitTypeId.ROACH}).exists:
                for probe in self.units(UnitTypeId.PROBE).ready:
                    if probe.distance_to(self.start_location) > 9:
                        enemy_units = self.enemy_units.of_type({UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.ROACH, UnitTypeId.LURKER}).closer_than(5, probe)
                        if enemy_units and not probe.is_attacking:
                            probe(AbilityId.SMART, self.mineral_field.closest_to(self.start_location))
        if self.enemy_units.of_type(offensive_units).exists:
            for probe in self.units(UnitTypeId.PROBE).ready:
                if probe.distance_to(self.start_location) > 9:
                    enemy_units = self.enemy_units.of_type(offensive_units).closer_than(4, probe)
                    if enemy_units and not probe.is_attacking:
                        probe(AbilityId.SMART, self.mineral_field.closest_to(self.start_location))

        for nexus in self.townhalls.ready:
            if (self.iteration - self.lastAttack) > 80 and not self.enemy_units.closer_than(10, nexus).amount > 0:
                if self.enemy_structures.of_type({UnitTypeId.PYLON, UnitTypeId.PHOTONCANNON}).exists:
                    for proxy in self.enemy_structures.of_type({UnitTypeId.PYLON, UnitTypeId.PHOTONCANNON}):
                        if nexus.distance_to(proxy) < 30 and self.iteration < 1000:
                            self.lastAttack = self.iteration
                            return
                for probe in self.units(UnitTypeId.PROBE):
                    if probe.is_attacking or probe.is_idle:
                        probe(AbilityId.SMART, self.mineral_field.closest_to(self.townhalls.ready.closest_to(probe)))
            elif nexus.shield_percentage < 0.5 and (self.iteration - self.lastAttack) > 50 and self.enemy_units.closer_than(10, nexus):
                self.lastAttack = self.iteration
                self.attacked_nexus = nexus.tag
                for probe in self.units(UnitTypeId.PROBE).closer_than(7, nexus):
                    try:
                        if self.enemy_units.exists:
                            probe.attack(self.enemy_units.closest_to(nexus).position)
                            break
                    except:
                        return
            elif nexus.distance_to(self.start_location) < 4 \
                    and (self.iteration - self.lastAttack) > 50 and \
                    self.enemy_units.of_type(offensive_units).exists and \
                    self.enemy_units.of_type(offensive_units).closer_than(8, nexus) and \
                    0 < self.enemy_units.closer_than(8, nexus).of_type(offensive_units).amount < 5:
                self.lastAttack = self.iteration
                self.attacked_nexus = nexus.tag
                for probe in self.units(UnitTypeId.PROBE).closer_than(7, nexus):
                    try:
                        if self.enemy_units.exists:
                            probe.attack(self.enemy_units.closest_to(nexus).position)
                            break
                    except:
                        return
            elif self.iteration < 2000 and (self.iteration - self.lastAttack) > 50 and self.enemy_units.exists and (
                    6 < self.enemy_units.closer_than(10, nexus).of_type({UnitTypeId.DRONE,
                                                                         UnitTypeId.PROBE,
                                                                         UnitTypeId.SCV}).amount or (
                            0 < self.enemy_units.closer_than(10, nexus).of_type(offensive_units).amount < 8 and
                            0 < self.enemy_units.closer_than(7, nexus).of_type(offensive_units).amount)):
                self.lastAttack = self.iteration
                self.attacked_nexus = nexus.tag
                total_probes = self.units(UnitTypeId.PROBE).ready.amount
                num_enemies = self.enemy_units.closer_than(9, nexus).amount
                max_probes_to_select = min(total_probes, num_enemies * 6)
                for i in range(max_probes_to_select):
                    probe = self.units(UnitTypeId.PROBE).ready[i]
                    if self.townhalls.ready.exists:
                        probe.attack(self.enemy_units.closest_to(nexus).position)
                break
            elif self.iteration < 2000 and (self.iteration - self.lastAttack) > 50 and self.enemy_units.exists and \
                    self.enemy_units.closer_than(10, nexus).amount > 7:
                self.lastAttack = self.iteration
                if not self.attacked_nexus or nexus.tag == self.attacked_nexus:
                    self.attacked_nexus = None
                    for probe in self.units(UnitTypeId.PROBE).closer_than(10, nexus):
                        try:
                            if self.townhalls.ready.exists and probe.distance_to(self.start_location) > 9:
                                probe.move(self.townhalls.closest_to(self.start_location).position)
                                probe(AbilityId.SMART, self.mineral_field.closest_to(self.townhalls.closest_to(self.start_location)), True)
                                break
                        except:
                            return

    async def get_next_exp(self):
        self.next_exp = await self.get_next_expansion()

    async def idle_defense_behavior(self):
        if self.iteration < self.do_something_after:
            if self.burrow_detected:
                for oracle in self.units.of_type({UnitTypeId.ORACLE}).idle:
                    if self.enemy_units.exists:
                        for enemy_unit in self.enemy_units:
                            if enemy_unit.distance_to(oracle) < 8:
                                if oracle.tag in self.availableUnitsAbilities.keys():
                                    if AbilityId.BEHAVIOR_PULSARBEAMON in self.availableUnitsAbilities.get(oracle.tag, set()):
                                        oracle(AbilityId.BEHAVIOR_PULSARBEAMON)
                                        oracle.smart(enemy_unit, True)
                            else:
                                if AbilityId.BEHAVIOR_PULSARBEAMOFF in self.availableUnitsAbilities.get(oracle.tag, set()):
                                    oracle(AbilityId.BEHAVIOR_PULSARBEAMOFF)

                    if (self.iteration - self.last_check) > 100 and self.next_exp and oracle.energy_percentage > 0.4 and \
                            (100 < (self.iteration - self.expand_time) < 400 and not self.already_pending(UnitTypeId.NEXUS) > 0) and \
                            AbilityId.ORACLEREVELATION_ORACLEREVELATION in self.availableUnitsAbilities.get(oracle.tag, set()):
                        self.last_check = self.iteration
                        oracle(AbilityId.ORACLEREVELATION_ORACLEREVELATION, self.next_exp)
            if (self.iteration - self.lastAttack) > 40:
                for unit in self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY, UnitTypeId.SENTRY, UnitTypeId.IMMORTAL, UnitTypeId.ZEALOT}).idle:
                    if self.enemy_units.exists:
                        first = None
                        for nexus in self.townhalls:
                            enemy_targets = self.enemy_units.closer_than(15, nexus) \
                                .filter(lambda e: e.type_id != UnitTypeId.OVERLORD and e.type_id != UnitTypeId.OVERSEER)
                            if enemy_targets.amount > 0:
                                for enemy_unit in enemy_targets:
                                    if not first:
                                        unit.attack(enemy_unit.position)
                                    else:
                                        unit.attack(enemy_unit.position, True)
            if self.enemy_race == Race.Zerg and self.townhalls.amount > 1 and \
                    self.iteration < self.do_something_after and (self.iteration - self.last_check) > 100 and \
                    (100 < (self.iteration - self.expand_time) < 400 and not self.already_pending(UnitTypeId.NEXUS) > 0):
                leader = self.units.find_by_tag(self.squadLeaderTag)
                if leader and self.next_exp:
                    leader.attack(self.next_exp)
                    self.last_check = self.iteration

    def manageArmy(self):
        if self.units(self.mainUnit).ready.amount > 0:
            squadLeader = None
            if self.squadLeaderTag:
                squadLeader = self.units.find_by_tag(self.squadLeaderTag)
            if not squadLeader:
                squadLeader = self.units(self.mainUnit).ready.first
                self.squadLeaderTag = squadLeader.tag
            for u in list(set(chain.from_iterable([self.units(UnitTypeId.VOIDRAY).idle,
                                                   self.units(UnitTypeId.IMMORTAL).idle,
                                                   self.units(UnitTypeId.STALKER).idle,
                                                   self.units(UnitTypeId.HIGHTEMPLAR).idle,
                                                   self.units(UnitTypeId.ZEALOT).idle,
                                                   self.units(UnitTypeId.ARCHON).idle,
                                                   self.units(UnitTypeId.SENTRY).idle,
                                                   self.units(UnitTypeId.COLOSSUS).idle]))):
                if squadLeader and u.tag != self.squadLeaderTag and u.tag != self.zealot_tag:
                    u.attack(squadLeader.position)

    async def attack(self):
        choice_dict = {0: "No Attack!",
                       1: "Attack close to our nexus !",
                       2: "Attack Enemy Structure !",
                       3: "Attack Enemy Start !",
                       4: "Attack specific target in vision !"}
        attackedNexus = None
        build_choices_made = 0
        if self.iteration < 100:
            return
        # Choices for build orders (can change only after 3000 iterations)
        if not self.armyComp or (self.iteration - self.lastBuildChoice) > 3000:
            if self.selected_strategy_sequence and build_choices_made < 3:
                self.lastBuildChoice = self.iteration
                self.armyComp = ArmyComp.GROUND if self.selected_strategy_sequence[build_choices_made] == ArmyComp.GROUND.value else ArmyComp.AIR
                self.mainUnit = UnitTypeId.STALKER if self.armyComp == ArmyComp.GROUND else UnitTypeId.VOIDRAY
                build_choices_made += 1
            else:
                self.lastBuildChoice = self.iteration
                buildChoice = random.randrange(0, 2)
                self.armyComp = ArmyComp.GROUND if buildChoice == ArmyComp.GROUND.value else ArmyComp.AIR
                self.mainUnit = UnitTypeId.STALKER if self.armyComp == ArmyComp.GROUND else UnitTypeId.VOIDRAY

        if self.armyComp == ArmyComp.AIR:
            self.mainUnit = UnitTypeId.VOIDRAY
        elif self.armyComp == ArmyComp.GROUND:
            self.mainUnit = UnitTypeId.STALKER
        else:
            self.mainUnit = UnitTypeId.VOIDRAY
        if self.units.of_type({UnitTypeId.STALKER, UnitTypeId.VOIDRAY}).idle.amount > 0 and self.units(UnitTypeId.IMMORTAL).idle.amount >= 0:
            if self.iteration > self.do_something_after:
                for nexus in self.townhalls.ready:
                    if nexus.shield_percentage < 0.6 and self.enemy_units.closer_than(8, nexus):
                        attackedNexus = nexus
                if self.use_model:
                    resized_image = cv.resize(self.flipped, (168, 168))
                    pad_with = ((0, 0), (0, 24), (0, 0))
                    padded_image = np.pad(resized_image, pad_with, mode='constant')
                    prediction = self.model.predict([padded_image.reshape([-1, 168, 192, 3])])
                    choice = np.argmax(prediction[0])
                    # print('prediction: ',choice)
                    logger.info("Choice #{}:{}".format(choice, choice_dict[choice]))
                else:
                    choice = random.randrange(0, 5)

                target = False
                logger.info(f"Resolved Choice #{choice}:{choice_dict[choice]}")
                if choice == 0:
                    # no attack
                    self.manageArmy()
                    wait = 190
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    try:
                        # attack_unit_closest_nexus
                        if self.enemy_units.amount > 0 and attackedNexus:
                            target = self.enemy_units.closest_to(attackedNexus).position
                            if self.iteration < 2000 and self.rushDetected is False:
                                for probe in self.units(UnitTypeId.PROBE).ready:
                                    probe.attack(self.enemy_units.closest_to(attackedNexus).position)
                        elif self.enemy_units.amount > 0:
                            for nexus in self.townhalls.ready:
                                if self.enemy_units.closer_than(15, nexus):
                                    target = self.enemy_units.closest_to(self.structures(UnitTypeId.NEXUS).random).position
                        else:
                            self.manageArmy()
                    except:
                        target = self.enemy_units.closest_to(random.choice(self.structures(UnitTypeId.NEXUS).ready)).position

                elif choice == 2:
                    # attack enemy structures
                    if len(self.enemy_structures) > 0:
                        target = random.choice(self.enemy_structures).position

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0].position

                elif choice == 4:
                    self.harass()
                    target = self.find_target()  # possibility to pass state
                if target:
                    leader = self.units.find_by_tag(self.squadLeaderTag)
                    units_close_to_leader = self.units(self.mainUnit).idle
                    if leader and self.units.of_type({UnitTypeId.VOIDRAY,
                                                      UnitTypeId.IMMORTAL,
                                                      UnitTypeId.STALKER,
                                                      UnitTypeId.SENTRY,
                                                      UnitTypeId.ARCHON,
                                                      UnitTypeId.HIGHTEMPLAR,
                                                      UnitTypeId.ZEALOT,
                                                      UnitTypeId.COLOSSUS}).closer_than(10, leader).amount > 0:
                        units_close_to_leader = self.units.of_type({UnitTypeId.VOIDRAY,
                                                                    UnitTypeId.IMMORTAL,
                                                                    UnitTypeId.STALKER,
                                                                    UnitTypeId.SENTRY,
                                                                    UnitTypeId.ARCHON,
                                                                    UnitTypeId.HIGHTEMPLAR,
                                                                    UnitTypeId.ZEALOT,
                                                                    UnitTypeId.COLOSSUS}).closer_than(10, leader)
                    for u in list(
                            set(chain.from_iterable([units_close_to_leader,
                                                     self.units(UnitTypeId.VOIDRAY).idle,
                                                     self.units(UnitTypeId.IMMORTAL).idle,
                                                     self.units(UnitTypeId.STALKER).idle,
                                                     self.units(UnitTypeId.SENTRY).idle,
                                                     self.units(UnitTypeId.ARCHON).idle,
                                                     self.units(UnitTypeId.HIGHTEMPLAR).idle,
                                                     self.units(UnitTypeId.ZEALOT).idle,
                                                     self.units(UnitTypeId.COLOSSUS).idle]))):
                        u.attack(target.position, True)
                y = np.zeros(5)
                y[choice] = 1
                self.train_data.append([y, self.flipped])


def main():
    while True:
        run_multiple_games(
            [
                GameMatch(
                    maps.get("GresvanAIE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Terran, Difficulty.Harder, ai_build=AIBuild.Rush)], realtime=True),
                GameMatch(
                    maps.get("StargazersAIE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Zerg, Difficulty.Harder, ai_build=AIBuild.Rush)], realtime=True),
                GameMatch(
                    maps.get("GresvanLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Zerg, Difficulty.Harder, ai_build=AIBuild.Rush)], realtime=True)
            ]
        )


if __name__ == '__main__':
    main()

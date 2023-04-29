from typing import Optional

from sc2.bot_ai import BotAI
from sc2.main import GameMatch, run_multiple_games
from loguru import logger
from sc2 import maps, position
from sc2.data import Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import *
import random
import cv2 as cv
import sys
import numpy as np
import time
from itertools import chain

from sc2.position import Point2

HEADLESS = True


class RaidenBot(BotAI):
    iteration = 0
    starting_time = 0
    scout_time = 0
    visited_corner_index = 0

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 300
        self.MAX_WORKERS = 80
        self.do_something_after = 0
        self.train_data = []
        self.flipped = []

    async def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)
        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration: int):
        self.iteration = iteration
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()
        await self.intel()

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

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

    async def scout(self):
        if len(self.units(UnitTypeId.OBSERVER)) > 0:
            scout = self.units(UnitTypeId.OBSERVER)[0]
            if scout.is_idle and not scout.target_in_range(self.enemy_units) and (time.perf_counter() - self.scout_time) < 80:
                enemy_location = self.enemy_start_locations[random.randrange(len(self.enemy_start_locations))]
                move_to = enemy_location
                scout.move(move_to)
            elif scout.is_idle and not scout.target_in_range(self.enemy_units) and (time.perf_counter() - self.scout_time) > 80:
                self.send_observer_to_corner(scout)
        else:
            for rf in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    rf.train(UnitTypeId.OBSERVER)
                    self.scout_time = time.perf_counter()

    def send_observer_to_corner(self, observer):
        corners = [Point2((10, 10)),
                   Point2((self.game_info.map_size.x - 10, 10)),
                   Point2((10, self.game_info.map_size.y - 10)),
                   Point2((self.game_info.map_size.x - 10, self.game_info.map_size.y - 10)),
                   self.game_info.map_center]

        # Find the closest corner to the observer
        corner = corners[self.visited_corner_index]
        self.visited_corner_index += 1
        if self.visited_corner_index == 4:
            self.visited_corner_index = 0
            self.scout_time = time.perf_counter()
        # Move the observer to the corner
        observer.move(corner)

    async def build_workers(self):
        if len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.townhalls.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)

    async def intel(self):
        # print('dir:', dir(self))  # 你总是可以使用dir命令来获取帮助，也可以直接看源码
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)  # 反转图片像素

        # UNIT:[SIZE,(BGR COLOR)]
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [3, (255, 100, 0)],
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
        if self.supply_cap == 0 :
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
            military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),
                 3)  # plausible supply (supply/200.0)
        cv.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),
                 3)  # population ratio (supply_left/supply)
        cv.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv.flip(game_data, 0)

        if not HEADLESS:
            resized = cv.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv.imshow('Intel', resized)
            cv.waitKey(1)  # 1ms

    async def build_pylons(self):
        try:
            if not self.supply_used >= 194 and not self.can_feed(UnitTypeId.COLOSSUS) and int(self.already_pending(UnitTypeId.PYLON)) == 0:
                nexus = self.townhalls.random
                if nexus:
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=nexus)
        except:
            logger.error("Error occured while trying to build a pylone.")

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position, True)
                if worker is None:
                    break
                if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                    worker.build(UnitTypeId.ASSIMILATOR, vespene, True)

    async def expand(self):
        try:
            if self.structures(UnitTypeId.NEXUS).ready.amount < (self.iteration / self.ITERATIONS_PER_MINUTE * 2) and self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
        except:
            logger.error("Error occured: Couldn't expand")

    async def offensive_force_buildings(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if pylon:
                try:
                    if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                        if self.can_afford(UnitTypeId.CYBERNETICSCORE) and int(self.already_pending(UnitTypeId.CYBERNETICSCORE)) == 0:
                            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                    elif self.structures(UnitTypeId.GATEWAY).ready.amount < 1:
                        if self.can_afford(UnitTypeId.GATEWAY) and int(self.already_pending(UnitTypeId.GATEWAY)) == 0:
                            await self.build(UnitTypeId.GATEWAY, near=pylon)
                    if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                        if self.structures(UnitTypeId.STARGATE).ready.amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.structures(
                                UnitTypeId.STARGATE).ready.amount < 6:
                            if self.can_afford(UnitTypeId.STARGATE) and int(self.already_pending(UnitTypeId.STARGATE)) == 0:
                                await self.build(UnitTypeId.STARGATE, near=pylon)
                    if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                        if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount < 1:
                            if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and int(self.already_pending(UnitTypeId.ROBOTICSFACILITY)) == 0:
                                await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
                except:
                    logger.error("Error occured when trying to build something")
                    return
            else:
                nexus = self.townhalls.random
                if nexus:
                    await self.build(UnitTypeId.PYLON, near=nexus)

    async def build_offensive_force(self):
        for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
            if self.can_afford(UnitTypeId.VOIDRAY) and sg.is_powered and self.supply_left > 0:
                sg.train(UnitTypeId.VOIDRAY)
        for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
            if self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 0:
                rb.train(UnitTypeId.IMMORTAL)

    def find_target(self):
        if len(self.enemy_units) > 0:
            return random.choice(self.enemy_units)
        elif len(self.enemy_structures) > 0:
            return random.choice(self.enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        if self.units(UnitTypeId.VOIDRAY).idle.amount > 0 and self.units(UnitTypeId.IMMORTAL).idle.amount >= 0:
            choice = random.randrange(0, 6)
            target = False
            if self.iteration > self.do_something_after:
                if choice == 0 or choice == 5 or choice == 6 or (time.perf_counter() - self.starting_time) < 600:
                    # no attack
                    if (time.perf_counter() - self.starting_time) < 600 :
                        choice = 0
                    wait = random.randrange(40, 250)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    # attack_unit_closest_nexus
                    if len(self.enemy_units) > 0:
                        target = self.enemy_units.closest_to(random.choice(self.structures(UnitTypeId.NEXUS)))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.enemy_units) > 0:
                        target = random.choice(self.enemy_units)

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                elif choice == 4:
                    target = self.find_target()  # possibility to pass state
                ground_enemies = self.enemy_units.filter(
                    lambda unit: not unit.is_flying and unit.type_id not in {UnitTypeId.LARVA, UnitTypeId.EGG}
                )
                if not ground_enemies:
                    for unit in self.units(UnitTypeId.VOIDRAY).idle:
                        # clear found structures
                        if self.enemy_structures:
                            # focus down low hp structures first
                            in_range_structures = self.enemy_structures.in_attack_range_of(unit)
                            if in_range_structures:
                                lowest_hp = min(in_range_structures, key=lambda e: (e.health + e.shield, e.tag))
                                if unit.weapon_cooldown == 0:
                                    unit.attack(lowest_hp)
                                else:
                                    # dont go closer than 1 with roaches to use ranged attack
                                    if unit.ground_range > 1:
                                        unit.move(lowest_hp.position.towards(unit, 1 + lowest_hp.radius))
                                    else:
                                        unit.move(lowest_hp.position)
                            else:
                                unit.move(self.enemy_structures.closest_to(unit))
                        # check bases to find new structures
                        else:
                            unit.move(self.find_target())
                    return
                if target:
                    for vr in list(set(chain.from_iterable([self.units(UnitTypeId.VOIDRAY).idle, self.units(UnitTypeId.IMMORTAL).idle]))):
                        vr.attack(target)
                y = np.zeros(6)
                y[choice] = 1
                self.train_data.append([y, self.flipped])


def main():
    while(True):
        run_multiple_games(
            [
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Hard)], realtime=False)
            ]
        )


if __name__ == '__main__':
    main()

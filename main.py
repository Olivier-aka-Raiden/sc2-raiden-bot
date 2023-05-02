from sc2.bot_ai import BotAI
from sc2.main import GameMatch, run_multiple_games
from loguru import logger
from sc2 import maps, position
from sc2.data import Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import *
from sc2.ids.upgrade_id import *
from sc2.ids.ability_id import *
import random
import cv2 as cv
import numpy as np
import time
from itertools import chain

from sc2.position import Point2

HEADLESS = False


class RaidenBot(BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 190
        self.MAX_WORKERS = 80
        self.tradeEfficiency = 100
        self.defensiveBehavior = True
        self.lastAttack = 0
        self.armyValue = 0
        self.stopTradeTime = 0
        self.upgrade_time = 0
        self.scout_time = 0
        self.visited_corner_index = 0
        self.armyValue = 0
        self.do_something_after = 0
        self.armyComp = "GROUND"
        self.mainUnit = UnitTypeId.STALKER if self.armyComp == "GROUND" else UnitTypeId.VOIDRAY
        self.scoutingObs = None
        self.followingObs = None
        self.squadLeader = None
        self.blinkAvailable = False
        self.units_abilities = []
        self.train_data = []
        self.flipped = []
        self.stalkers = {}

    async def on_end(self, game_result):
        logger.info('--- on_end called ---')
        logger.info(game_result)
        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data, dtype=object))

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
        await self.probeEscape()
        await self.upgrades()
        await self.computeTradeEfficiency()
        await self.followingObserver()
        await self.getAvailableAbilities()
        await self.handleBlink()

    def calculateBlinkDest(self, stalker, enemy):
        enemy_position = enemy.position
        stalker_position = stalker.position
        targetBlinkPosition = (stalker.position[0] + stalker_position[0] - enemy_position[0], stalker_position[1] + stalker_position[1] - enemy_position[1])
        escape_location = stalker.position.towards(position.Point2(position.Pointlike(targetBlinkPosition)), 6)
        return position.Point2(position.Pointlike(escape_location))

    async def handleBlink(self):
        for stalker in self.units(UnitTypeId.STALKER).owned.ready:
            if stalker.tag in self.stalkers.keys():
                if stalker.shield_percentage < 0.1 and AbilityId.EFFECT_BLINK_STALKER in self.stalkers.get(stalker.tag, set()):
                    enemy = self.enemy_units.closest_to(stalker) if self.enemy_units.exists else None
                    if enemy:
                        targetBlinkPosition = self.calculateBlinkDest(stalker, enemy)
                        if stalker.in_ability_cast_range(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition):
                            stalker(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition)

    async def getAvailableAbilities(self):
        self.units_abilities = await self.get_available_abilities(self.units.owned)
        for abilities in self.units_abilities:
            if abilities.__contains__(AbilityId.EFFECT_BLINK_STALKER):
                self.blinkAvailable = True
                self.stalkers = await self.client.query_available_abilities_with_tag(self.units(UnitTypeId.STALKER).owned.ready)

    async def computeTradeEfficiency(self):
        currentArmyValue = self.units.of_type({UnitTypeId.IMMORTAL, UnitTypeId.VOIDRAY, UnitTypeId.STALKER}).owned.amount
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
        elif self.tradeEfficiency == 100 and (self.iteration - self.stopTradeTime) > 400 and self.defensiveBehavior:
            self.stopTradeTime = self.iteration
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

    async def scout(self):
        if self.units(UnitTypeId.OBSERVER).ready.amount > 0:
            if not self.scoutingObs in self.units(UnitTypeId.OBSERVER).ready:
                if self.followingObs:
                    self.scoutingObs = self.followingObs
                    self.followingObs = None
                else:
                    self.scoutingObs = self.units(UnitTypeId.OBSERVER).ready[0]
            if self.scoutingObs.is_idle and not self.scoutingObs.target_in_range(self.enemy_units):
                enemy_location = self.enemy_start_locations[random.randrange(len(self.enemy_start_locations))]
                move_to = self.random_location_variance(enemy_location)
                self.scoutingObs.move(move_to, True)
            elif self.scoutingObs.is_idle and not self.scoutingObs.target_in_range(self.enemy_units) and self.iteration > 5000:
                self.send_observer_to_corner(self.scoutingObs)
        else:
            for rf in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    rf.train(UnitTypeId.OBSERVER)

    async def followingObserver(self):
        if self.units(UnitTypeId.OBSERVER).owned.ready.amount == 2:
            if not self.followingObs:
                observers = self.units(UnitTypeId.OBSERVER).owned.ready.filter(lambda unit: unit.tag != self.scoutingObs.tag)
                self.followingObs = observers[0]
            if self.followingObs.is_idle and not self.followingObs.target_in_range(self.enemy_units) and self.units(self.mainUnit).owned.amount > 0:
                friendly_unit = self.squadLeader if self.squadLeader else self.units(self.mainUnit).owned.closest_to(self.followingObs)
                move_to = friendly_unit
                self.followingObs.move(move_to, True)
            elif self.enemy_units.amount > 0 and self.followingObs.target_in_range(self.enemy_units):
                enemy_location = self.enemy_units.closest_to(self.followingObs).position
                move_to = enemy_location
                self.followingObs.move(move_to, True)
        else:
            for rf in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0 and not self.followingObs:
                    rf.train(UnitTypeId.OBSERVER)

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
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

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
            UnitTypeId.IMMORTAL: [3, (255, 100, 0)],
            UnitTypeId.STALKER: [3, (255, 100, 0)],
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
            military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap - self.supply_left)
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
            if not self.supply_used >= 194 and not self.can_feed(UnitTypeId.COLOSSUS) and int(self.already_pending(UnitTypeId.PYLON)) == 0:
                nexus = self.townhalls.random
                if nexus:
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=nexus)
        except:
            logger.error("Error occured while trying to build a pylone.")

    async def build_assimilators(self):
        try:
            for nexus in self.townhalls.ready:
                vespenes = self.vespene_geyser.closer_than(15.0, nexus)
                for vespene in vespenes:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR) or self.already_pending(UnitTypeId.ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vespene.position, True)
                    if worker is None:
                        break
                    if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists and not self.already_pending(UnitTypeId.ASSIMILATOR):
                        worker.build(UnitTypeId.ASSIMILATOR, vespene, True)
        except:
            logger.error("Error while trying to build an assimilator")

    async def expand(self):
        try:
            if self.structures(UnitTypeId.NEXUS).ready.amount + 1 < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(
                    UnitTypeId.NEXUS) and self.townhalls.ready.amount < 12 and not self.already_pending(UnitTypeId.NEXUS):
                await self.expand_now()
        except:
            logger.error("Error occured: Couldn't expand")

    async def offensive_force_buildings(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if pylon:
                try:
                    if self.armyComp == "AIR":
                        if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and int(self.already_pending(UnitTypeId.CYBERNETICSCORE)) == 0:
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        elif self.structures(UnitTypeId.GATEWAY).ready.amount < 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and int(self.already_pending(UnitTypeId.GATEWAY)) == 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.STARGATE).ready.amount + 1 < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.structures(
                                    UnitTypeId.STARGATE).ready.amount < 6 and self.structures(UnitTypeId.STARGATE).ready.idle.amount == 0:
                                if self.can_afford(UnitTypeId.STARGATE) and int(self.already_pending(UnitTypeId.STARGATE)) == 0:
                                    await self.build(UnitTypeId.STARGATE, near=pylon)
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount < 1:
                                if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and int(self.already_pending(UnitTypeId.ROBOTICSFACILITY)) == 0:
                                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
                    if self.armyComp == "GROUND":
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and self.iteration > 1400:
                            if self.structures(UnitTypeId.FORGE).ready.amount < 1:
                                if self.can_afford(UnitTypeId.FORGE) and int(self.already_pending(UnitTypeId.FORGE)) == 0:
                                    await self.build(UnitTypeId.FORGE, near=pylon)
                        if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and int(self.already_pending(UnitTypeId.CYBERNETICSCORE)) == 0:
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        elif self.structures(UnitTypeId.GATEWAY).ready.amount < 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and int(self.already_pending(UnitTypeId.GATEWAY)) == 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        elif self.structures(UnitTypeId.GATEWAY).ready.amount > 0 and self.structures(UnitTypeId.GATEWAY).ready.amount + 2 < (
                                self.iteration / self.ITERATIONS_PER_MINUTE) < self.structures(UnitTypeId.NEXUS).ready.amount + 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and int(self.already_pending(UnitTypeId.GATEWAY)) == 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.amount < 1:
                                if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and int(self.already_pending(UnitTypeId.TWILIGHTCOUNCIL)) == 0:
                                    await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
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
        if self.armyComp == "AIR":
            for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                if self.can_afford(UnitTypeId.VOIDRAY) and sg.is_powered and self.supply_left > 0:
                    sg.train(UnitTypeId.VOIDRAY)
            for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 0:
                    rb.train(UnitTypeId.IMMORTAL)
        if self.armyComp == "GROUND":
            cyberList = self.structures(UnitTypeId.CYBERNETICSCORE).ready
            if cyberList.idle.amount > 0:
                cyber = cyberList.ready.idle.random
                if cyber and self.can_afford(UpgradeId.WARPGATERESEARCH) and not self.already_pending_upgrade(
                        UpgradeId.WARPGATERESEARCH) and cyber.is_powered and self.units(UnitTypeId.STALKER).ready.amount > 1:
                    cyber.research(UpgradeId.WARPGATERESEARCH)
            for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
                if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 1:
                    gateway(AbilityId.MORPH_WARPGATE)
                if self.can_afford(UnitTypeId.STALKER) and gateway.is_powered and cyberList.amount > 0 and not self.research(
                        UpgradeId.WARPGATERESEARCH) and self.supply_left > 0:
                    if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) < 1:
                        gateway.train(UnitTypeId.STALKER)
            for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                if pylon:
                    pos = pylon.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        break
                    warpgate.warp_in(UnitTypeId.STALKER, placement)
            for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 0:
                    rb.train(UnitTypeId.IMMORTAL)
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                if self.already_pending_upgrade(UpgradeId.BLINKTECH) == 0 and self.iteration > 1000:
                    self.research(UpgradeId.BLINKTECH)

    def find_target(self):
        if self.enemy_units.amount > 0:
            if self.squadLeader:
                return self.enemy_units.closest_to(self.squadLeader)
            else:
                return self.enemy_units.random
        elif self.enemy_structures.amount > 0 and self.iteration > 4000:
            return self.enemy_structures.random
        else:
            return self.enemy_start_locations[0]

    async def upgrades(self):
        try:
            if self.armyComp == "AIR":
                if self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0 and self.iteration > 2000:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) > 0.4 and len(self.structures(UnitTypeId.FLEETBEACON).ready) < 1:
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    if pylon and self.can_afford(UnitTypeId.FLEETBEACON) and int(self.already_pending(UnitTypeId.FLEETBEACON)) == 0:
                        await self.build(UnitTypeId.FLEETBEACON, near=pylon)
                if self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0 and self.upgrade_time != 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2):
                        self.upgrade_time = self.iteration
                if self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3):
                        self.upgrade_time = 0
            if self.armyComp == "GROUND":
                if self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0 and self.iteration > 2000:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1):
                        self.upgrade_time = self.iteration
                elif self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) > 0.4 and len(self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready) < 1:
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    if pylon and self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and int(self.already_pending(UnitTypeId.TWILIGHTCOUNCIL)) == 0:
                        await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
                if self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0 and self.upgrade_time != 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2):
                        self.upgrade_time = self.iteration
                if self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0 and (self.iteration - self.upgrade_time) > 300:
                    if self.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3):
                        self.upgrade_time = 0

        except:
            logger.error("Error occured while trying to upgrade")

    async def probeEscape(self):
        for nexus in self.townhalls.ready:
            if nexus.shield_percentage < 0.1 and (self.iteration - self.lastAttack) > 100 and self.enemy_units.closer_than(20, nexus):
                self.attackedNexus = nexus
                self.lastAttack = self.iteration
                for probe in self.units(UnitTypeId.PROBE).closer_than(15, self.attackedNexus.position):
                    try:
                        probe.move(self.mineral_field.closer_than(10, self.townhalls.n_closest_to_distance(self.attackedNexus.position, 40, 15).random).random)
                        probe.gather(self.mineral_field.closer_than(4, probe).random, True)
                    except:
                        logger.error("erreur lors de la fuite des probes...")

    def manageArmy(self):
        if self.units(self.mainUnit).owned.ready.amount > 0:
            if self.squadLeader:
                if self.squadLeader.tag not in self.units(self.mainUnit).ready.tags:
                    self.squadLeader = self.units(self.mainUnit).ready[0]
                else:
                    self.squadLeader = self.units(self.mainUnit).ready[0]
            else:
                self.squadLeader = self.units(self.mainUnit).ready[0]
            for u in list(set(chain.from_iterable([self.units(UnitTypeId.VOIDRAY).idle, self.units(UnitTypeId.IMMORTAL).idle, self.units(UnitTypeId.STALKER).idle]))):
                if self.squadLeader and u.tag != self.squadLeader.tag:
                    u.attack(self.squadLeader.position)

    async def attack(self):
        if self.armyComp == "AIR":
            self.mainUnit = UnitTypeId.VOIDRAY
        elif self.armyComp == "GROUND":
            self.mainUnit = UnitTypeId.STALKER
        else:
            self.mainUnit = UnitTypeId.VOIDRAY
        if self.units(self.mainUnit).idle.amount > 0 and self.units(UnitTypeId.IMMORTAL).idle.amount >= 0:
            choice = random.randrange(0, 6)
            if not self.blinkAvailable or self.defensiveBehavior:
                if choice == 3 or choice == 2 or choice == 4:
                    choice = 0
                if self.attackedNexus:
                    choice = 1

            target = False

            if self.iteration > self.do_something_after:
                if choice == 0 or choice == 5 or choice == 6:
                    # no attack
                    self.manageArmy()
                    wait = 100
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    try:
                        # attack_unit_closest_nexus
                        if len(self.enemy_units) > 0 and self.attackedNexus:
                            target = self.enemy_units.closest_to(self.attackedNexus)
                            del self.attackedNexus
                        elif len(self.enemy_units) > 0:
                            for nexus in self.townhalls:
                                if self.enemy_units.closer_than(20, nexus):
                                    target = self.enemy_units.closest_to(self.structures(UnitTypeId.NEXUS).random)
                    except:
                        target = self.enemy_units.closest_to(random.choice(self.structures(UnitTypeId.NEXUS)))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.enemy_units) > 0:
                        target = random.choice(self.enemy_units)
                        wait = random.randrange(10, 15)
                        self.do_something_after = self.iteration + wait

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0].position

                elif choice == 4:
                    target = self.find_target()  # possibility to pass state
                if target:
                    for u in list(
                            set(chain.from_iterable([self.units(UnitTypeId.VOIDRAY).idle, self.units(UnitTypeId.IMMORTAL).idle, self.units(UnitTypeId.STALKER).idle]))):
                        u.attack(target, True)
                y = np.zeros(6)
                y[choice] = 1
                self.train_data.append([y, self.flipped])


def main():
    while True:
        run_multiple_games(
            [
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("ThunderbirdLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False)
            ]
        )


if __name__ == '__main__':
    main()

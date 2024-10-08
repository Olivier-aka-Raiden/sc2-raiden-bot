from enum import Enum

import tensorflow as tf
from sc2.bot_ai import BotAI
from sc2.main import GameMatch, run_multiple_games
from loguru import logger
from sc2 import maps
from sc2 import position
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Result
from sc2.ids.unit_typeid import *
from sc2.ids.upgrade_id import *
from sc2.ids.ability_id import *
from sc2.ids.buff_id import *
import random
import cv2 as cv
import numpy as np

from sc2.player import Bot, Computer
from sc2.position import Point2
from itertools import chain
from sc2 import client

HEADLESS = True


class ArmyComp(Enum):
    GROUND = 0
    AIR = 1


class RaidenBot(BotAI):

    def __init__(self):
        self.lastBuildChoice = None
        self.ITERATIONS_PER_MINUTE = 190
        self.MAX_WORKERS = 80
        self.tradeEfficiency = 100
        self.defensiveBehavior = True
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
        self.scoutingObsTag = None
        self.followingObsTag = None
        self.squadLeaderTag = None
        self.pylonAtRamp = False
        self.units_abilities = []
        self.train_data = []
        self.flipped = []
        self.availableUnitsAbilities = {}
        self.use_model = True
        if self.use_model:
            self.model = tf.keras.models.load_model("etc/BasicCNN-10-epochs-0.0001-LR-STAGE2")

    async def on_end(self, game_result):
        logger.info("--- on_end called ---")
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
        await self.getAvailableAbilities()
        await self.computeTradeEfficiency()
        await self.handleScout()
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
        await self.handleBlink()
        await self.handleHighTemplar()
        await self.manageChronoboost()
        await self.handleSentry()
        await self.handleKiting()
        await self.handleHarass()
        await self.handleVoidray()
        await self.build_defensive_structures()

    async def build_defensive_structures(self):
        try:
            if self.iteration > 2000 and self.structures(UnitTypeId.FORGE).ready.exists:
                if not self.already_pending(UnitTypeId.PYLON) > 0 and \
                        not self.already_pending(UnitTypeId.PHOTONCANNON) > 0 and \
                        not self.already_pending(UnitTypeId.SHIELDBATTERY) > 0:

                    for nexus in self.townhalls.ready:
                        if not self.structures(UnitTypeId.PYLON).closer_than(7, nexus).ready.exists:
                            await self.build(UnitTypeId.PYLON, near=nexus)
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
        for nexus in self.townhalls.ready:
            # Chrono nexus if cybercore is not ready, else chrono cybercore
            if self.iteration > 150 and not 0 < self.already_pending_upgrade(
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
            elif self.structures(UnitTypeId.FORGE).ready.exists and (
                    0 < self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) < 1 or 0 < self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRWEAPONSLEVEL2) < 1 or 0 < self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) < 1):
                if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    if nexus.energy >= 50 and not self.structures(UnitTypeId.FORGE).ready.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.structures(UnitTypeId.CYBERNETICSCORE).ready.first)
            else:
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                    ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
                    if ccore:
                        if not ccore.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not ccore.is_idle:
                            if nexus.energy >= 50:
                                nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore)

    def calculateBlinkDest(self, stalker, enemy):
        enemy_position = enemy.position
        stalker_position = stalker.position
        targetBlinkPosition = (stalker_position[0] + stalker_position[0] - enemy_position[0], stalker_position[1] + stalker_position[1] - enemy_position[1])
        escape_location = stalker.position.towards(position.Point2(position.Pointlike(targetBlinkPosition)), 6)
        return position.Point2(position.Pointlike(escape_location))

    def calculatePylonPos(self, nexus, mineral):
        mineral_position = mineral.position
        nexus_position = nexus.position
        targetPylonPosition = (nexus_position[0] + nexus_position[0] - mineral_position[0], nexus_position[1] + nexus_position[1] - mineral_position[1])
        return position.Point2(position.Pointlike(targetPylonPosition))
    def calculateScoutEscape(self, scoutingObs, enemy):
        enemy_position = enemy.position
        scout_position = scoutingObs.position
        targetEscapePosition = (scoutingObs.position[0] + scout_position[0] - enemy_position[0], scout_position[1] + scout_position[1] - enemy_position[1])
        escape_location = scoutingObs.position.towards(position.Point2(position.Pointlike(targetEscapePosition)), 2)
        return position.Point2(position.Pointlike(escape_location))

    async def handleBlink(self):
        for stalker in self.units(UnitTypeId.STALKER).ready:
            if stalker.tag in self.availableUnitsAbilities.keys():
                if stalker.shield_percentage < 0.1 and AbilityId.EFFECT_BLINK_STALKER in self.availableUnitsAbilities.get(stalker.tag, set()):
                    enemy = self.enemy_units.closest_to(stalker) if self.enemy_units.exists else None
                    if enemy:
                        targetBlinkPosition = self.calculateBlinkDest(stalker, enemy)
                        if stalker.in_ability_cast_range(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition):
                            stalker(AbilityId.EFFECT_BLINK_STALKER, targetBlinkPosition)

    def calculateStormCastPosition(self, enemy):
        enemy_position = enemy.position
        return position.Point2(position.Pointlike(enemy_position))

    async def handleHighTemplar(self):
        for ht in self.units(UnitTypeId.HIGHTEMPLAR).ready:
            if ht.tag in self.availableUnitsAbilities.keys():
                if AbilityId.PSISTORM_PSISTORM in self.availableUnitsAbilities.get(ht.tag, set()):
                    enemy = self.enemy_units.of_type({UnitTypeId.MARINE, UnitTypeId.MARAUDER}).closest_to(ht) if self.enemy_units.of_type({UnitTypeId.MARINE, UnitTypeId.MARAUDER}).exists else None
                    if enemy:
                        targetStormPosition = self.calculateStormCastPosition(enemy)
                        if ht.in_ability_cast_range(AbilityId.PSISTORM_PSISTORM, targetStormPosition):
                            ht(AbilityId.PSISTORM_PSISTORM, targetStormPosition)

    async def handleKiting(self):
        for stalker in self.units(UnitTypeId.STALKER).ready:
            enemy = self.enemy_units.closest_to(stalker) if self.enemy_units.exists else None
            if enemy:
                if self.enemy_race != Race.Zerg:
                    if self.enemy_race == Race.Protoss:
                        if enemy.distance_to(stalker) < 3:
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 3)
                            stalker.move(kite_location)
                    if self.enemy_race == Race.Terran:
                        if enemy.distance_to(stalker) < 5.2:
                            kite_location = stalker.position.towards(position.Point2(position.Pointlike(self.start_location)), 2)
                            stalker.move(kite_location)
        for supportUnit in self.units.of_type({UnitTypeId.SENTRY, UnitTypeId.IMMORTAL}).ready:
            enemy = self.enemy_units.closest_to(supportUnit) if self.enemy_units.exists else None
            if enemy:
                if self.enemy_race != Race.Zerg:
                    if enemy.distance_to(supportUnit) < 3:
                        kite_location = supportUnit.position.towards(position.Point2(position.Pointlike(self.start_location)), 3)
                        supportUnit.move(kite_location)

        for casterUnit in self.units(UnitTypeId.HIGHTEMPLAR).ready:
            enemy = self.enemy_units.closest_to(casterUnit) if self.enemy_units.exists else None
            if enemy:
                if enemy.distance_to(casterUnit) < 5.5:
                    kite_location = casterUnit.position.towards(position.Point2(position.Pointlike(self.start_location)), 6)
                    casterUnit.move(kite_location)

    async def handleSentry(self):
        if self.units(UnitTypeId.SENTRY).ready.exists:
            sentry = self.units(UnitTypeId.SENTRY).ready.first
            if self.enemy_units.amount > 0:
                enemy = self.enemy_units.closest_to(sentry)
                if enemy:
                    if not sentry.is_using_ability(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD) and sentry.distance_to(enemy) < 6 and \
                            not enemy.is_detector and not enemy.type_id == UnitTypeId.PROBE and \
                            not enemy.type_id == UnitTypeId.SCV and \
                            not enemy.type_id == UnitTypeId.DRONE and \
                            AbilityId.GUARDIANSHIELD_GUARDIANSHIELD in self.availableUnitsAbilities.get(sentry.tag, set()):
                        sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)

    async def handleVoidray(self):
        if self.units(UnitTypeId.VOIDRAY).ready.exists:
            voidray = self.units(UnitTypeId.VOIDRAY).ready.first
            if self.enemy_units.amount > 0:
                enemy = self.enemy_units.closest_to(voidray)
                if enemy:
                    if not voidray.is_using_ability(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT) and voidray.distance_to(enemy) < 6 and enemy.is_armored and \
                            not enemy.is_detector and AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT in self.availableUnitsAbilities.get(voidray.tag, set()):
                        voidray(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT)

    async def handleScout(self):
        scoutingObs = None
        if self.scoutingObsTag:
            scoutingObs = self.units.find_by_tag(self.scoutingObsTag)
        if scoutingObs:
            enemies = self.enemy_units.closer_than(11, scoutingObs)
            enemy_structures = self.enemy_structures.closer_than(11, scoutingObs)
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
        elif (self.iteration - self.stopTradeTime) > 400 and self.defensiveBehavior:
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
                        friendly_unit = squadLeader if squadLeader else None
                        if not friendly_unit and self.units(self.mainUnit).ready.amount > 0:
                            friendly_unit = self.units(self.mainUnit).ready.closest_to(followingObs)
                        elif not friendly_unit:
                            friendly_unit = self.townhalls.closest_to(followingObs).position
                        move_to = friendly_unit
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
            UnitTypeId.WARPGATE: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [4, (255, 100, 0)],
            UnitTypeId.IMMORTAL: [3, (120, 100, 0)],
            UnitTypeId.STALKER: [3, (120, 100, 50)],
            UnitTypeId.SENTRY: [2, (255, 192, 203)],
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
            if not self.supply_used >= 194 and not self.can_feed(UnitTypeId.COLOSSUS) \
                    and (not self.already_pending(UnitTypeId.PYLON) > 0 or (self.iteration > 1500 and
                                                                              self.already_pending(UnitTypeId.PYLON) < 2)):
                nexus = self.townhalls.random
                if nexus:
                    if self.can_afford(UnitTypeId.PYLON):
                        mineral = self.mineral_field.closer_than(7, nexus).random
                        target = self.calculatePylonPos(nexus, mineral)
                        await self.build(UnitTypeId.PYLON, near=target)
        except:
            return

    async def build_assimilators(self):
        try:
            if self.iteration > 150:
                for nexus in self.townhalls.ready:
                    vespenes = self.vespene_geyser.closer_than(15.0, nexus)
                    if vespenes:
                        if not self.already_pending(UnitTypeId.ASSIMILATOR) > 0:
                            for vespene in vespenes:
                                if self.can_afford(UnitTypeId.ASSIMILATOR):
                                    worker = self.select_build_worker(vespene.position, True)
                                    if worker is None:
                                        return
                                    if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                                        worker.build(UnitTypeId.ASSIMILATOR, vespene, True)
                                        worker.gather(self.mineral_field.closest_to(worker), True)
                                        break
        except:
            return

    async def expand(self):
        try:
            if self.townhalls.ready.amount + 1 < (self.iteration / self.ITERATIONS_PER_MINUTE) * 2 and self.can_afford(
                    UnitTypeId.NEXUS) and self.townhalls.ready.amount < 15 and not self.already_pending(UnitTypeId.NEXUS) > 0 and \
                    (self.iteration - self.expand_time) > 400:
                await self.expand_now()
                self.expand_time = self.iteration
            if self.already_pending(UnitTypeId.NEXUS) > 0.3:
                if self.townhalls.not_ready.first.health_percentage < 0.09:
                    self.townhalls.not_ready.first(AbilityId.CANCEL)
        except:
            return

    async def offensive_force_buildings(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if pylon:
                try:
                    if self.armyComp == ArmyComp.AIR:
                        if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE).ready:
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE) > 0:
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        elif self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.amount < 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.STARGATE).amount + 2 < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.structures(
                                    UnitTypeId.STARGATE).ready.amount < 10 and self.structures(UnitTypeId.STARGATE).ready.idle.amount == 0:
                                if self.can_afford(UnitTypeId.STARGATE) and (not self.already_pending(UnitTypeId.STARGATE) > 0 or
                                                                             not (self.iteration > 2500 and self.already_pending(UnitTypeId.STARGATE) > 2)):
                                    await self.build(UnitTypeId.STARGATE, near=pylon)
                        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount < 1:
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
                                self.iteration > 3000 and not self.already_pending(UnitTypeId.ROBOTICSBAY) > 0:
                                if self.can_afford(UnitTypeId.ROBOTICSBAY) and self.structures(UnitTypeId.ROBOTICSBAY).amount < 1:
                                    await self.build(UnitTypeId.ROBOTICSBAY, near=pylon)
                        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and not self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and \
                                not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL) > 0:
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
                                self.structures(UnitTypeId.NEXUS).amount > 1 and not self.already_pending(UnitTypeId.CYBERNETICSCORE) > 0:
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
                        if self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).amount < (self.townhalls.ready.amount + 2) and \
                                self.iteration > 1500 and \
                                self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0:
                                await self.build(UnitTypeId.GATEWAY, near=pylon)
                        elif self.structures.of_type({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE}).ready.amount < 1:
                            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY) > 0:
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
                    logger.error("Error occured when trying to build something")
                    return
            else:
                nexus = self.townhalls.random
                if nexus:
                    await self.build(UnitTypeId.PYLON, near=nexus)

    async def build_offensive_force(self):
        try:
            if self.armyComp == ArmyComp.AIR:
                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.VOIDRAY) and sg.is_powered and self.supply_left > 0:
                        if self.iteration > 2500 and not self.units(UnitTypeId.ORACLE).exists and \
                                self.units(UnitTypeId.ORACLE).amount < 2:
                            sg.train(UnitTypeId.ORACLE)
                        else:
                            sg.train(UnitTypeId.VOIDRAY)
                for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 0:
                        rb.train(UnitTypeId.IMMORTAL)
                for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                    if self.structures(UnitTypeId.PYLON).ready.exists:
                        pylon = self.structures(UnitTypeId.PYLON).ready.random
                        if pylon:
                            pos = pylon.position.to2.random_on_distance(4)
                            placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                            if placement is None:
                                # return ActionResult.CantFindPlacementLocation
                                logger.info("can't place")
                                break
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and self.can_afford(UnitTypeId.ZEALOT) and \
                                    self.iteration > 2500:
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
                            placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                            if placement is None:
                                # return ActionResult.CantFindPlacementLocation
                                logger.info("can't place")
                                break
                            if self.research(UpgradeId.PSISTORMTECH) and self.units(UnitTypeId.HIGHTEMPLAR).ready.amount < 3 and \
                                    self.units(self.mainUnit).ready.amount > 10 and self.can_afford(UnitTypeId.HIGHTEMPLAR):
                                warpgate.warp_in(UnitTypeId.HIGHTEMPLAR, placement)
                            if not self.units(UnitTypeId.DARKTEMPLAR).exists and self.structures(UnitTypeId.DARKSHRINE).ready.exists and \
                                    self.units(UnitTypeId.DARKTEMPLAR).amount < 2 and self.can_afford(UnitTypeId.DARKTEMPLAR):
                                warpgate.warp_in(UnitTypeId.DARKTEMPLAR, placement)
                            if not self.units(UnitTypeId.SENTRY).exists and self.can_afford(UnitTypeId.SENTRY) and self.units(UnitTypeId.SENTRY).amount < 1:
                                warpgate.warp_in(UnitTypeId.SENTRY, placement)
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and self.can_afford(UnitTypeId.STALKER):
                                warpgate.warp_in(UnitTypeId.STALKER, placement)
                            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists and self.can_afford(UnitTypeId.ZEALOT) and \
                                    self.iteration > 2500:
                                warpgate.warp_in(UnitTypeId.ZEALOT, placement)
                for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.COLOSSUS) and rb.is_powered and self.supply_left > 0 :
                        if self.research(UpgradeId.EXTENDEDTHERMALLANCE) and self.units(UnitTypeId.COLOSSUS).ready.amount < 4:
                            rb.train(UnitTypeId.COLOSSUS)
                    if self.can_afford(UnitTypeId.IMMORTAL) and rb.is_powered and self.supply_left > 0:
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
                                    for enemyWorker in enemyWorkers:
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
                            for enemyWorker in enemyWorkers:
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
        for nexus in self.townhalls.ready:
            if nexus.shield_percentage < 0.5 and (self.iteration - self.lastAttack) > 150 and self.enemy_units.closer_than(10, nexus):
                self.lastAttack = self.iteration
                for probe in self.units(UnitTypeId.PROBE).closer_than(6, nexus):
                    try:
                        if self.enemy_units.exists:
                            probe.attack(self.enemy_units.closest_to(nexus).position)
                    except:
                        return
            if self.iteration < 2000 and (self.iteration - self.lastAttack) > 150 and self.enemy_units.exists and \
                    self.enemy_units.closer_than(10, nexus).of_type({UnitTypeId.ZEALOT,
                                                                     UnitTypeId.STALKER,
                                                                     UnitTypeId.ZERGLING,
                                                                     UnitTypeId.MARINE,
                                                                     UnitTypeId.MARAUDER}).amount > 0 and \
                    self.enemy_units.closer_than(10, nexus).of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}).amount == 0:
                self.lastAttack = self.iteration
                for probe in self.units(UnitTypeId.PROBE).closer_than(7, nexus):
                    try:
                        probe.attack(self.enemy_units.closest_to(nexus).position)
                    except:
                        return

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
                                                   self.units(UnitTypeId.SENTRY).idle,
                                                   self.units(UnitTypeId.COLOSSUS).idle]))):
                if squadLeader and u.tag != self.squadLeaderTag:
                    u.attack(squadLeader.position)

    async def attack(self):
        choice_dict = {0: "No Attack!",
                       1: "Attack close to our nexus !",
                       2: "Attack Enemy Structure !",
                       3: "Attack Enemy Start !",
                       4: "Attack specific target in vision !"}
        attackedNexus = None
        # Choices for build orders (can change only after 3000 iterations)
        if not self.armyComp or (self.iteration - self.lastBuildChoice) > 2500:
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
                    try:
                        # attack_unit_closest_nexus
                        if self.enemy_units.amount > 0 and attackedNexus:
                            target = self.enemy_units.closest_to(attackedNexus).position
                        elif self.enemy_units.amount > 0:
                            for nexus in self.townhalls:
                                if self.enemy_units.closer_than(20, nexus):
                                    target = self.enemy_units.closest_to(self.structures(UnitTypeId.NEXUS).random).position
                    except:
                        logger.error("Couldn't target enemy close to nexus")

                elif choice == 1:
                    try:
                        # attack_unit_closest_nexus
                        if self.enemy_units.amount > 0 and attackedNexus:
                            target = self.enemy_units.closest_to(attackedNexus).position
                        elif self.enemy_units.amount > 0:
                            for nexus in self.townhalls:
                                if self.enemy_units.closer_than(20, nexus):
                                    target = self.enemy_units.closest_to(self.structures(UnitTypeId.NEXUS).random).position
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
                    for u in list(
                            set(chain.from_iterable([self.units(UnitTypeId.VOIDRAY).idle,
                                                     self.units(UnitTypeId.IMMORTAL).idle,
                                                     self.units(UnitTypeId.STALKER).idle,
                                                     self.units(UnitTypeId.SENTRY).idle,
                                                     self.units(UnitTypeId.HIGHTEMPLAR),
                                                     self.units(UnitTypeId.ZEALOT),
                                                     self.units(UnitTypeId.COLOSSUS)]))):
                        u.attack(target, True)
                y = np.zeros(5)
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
                    maps.get("AltitudeLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False),
                GameMatch(
                    maps.get("GresvanLE"), [
                        Bot(Race.Protoss, RaidenBot()),
                        Computer(Race.Protoss, Difficulty.Harder)], realtime=False)
            ]
        )


if __name__ == '__main__':
    main()

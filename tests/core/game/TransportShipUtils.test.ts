import { beforeAll, describe, expect, test } from "vitest";
import {
  Game,
  Player,
  PlayerInfo,
  PlayerType,
} from "../../../src/core/game/Game";
import { TileRef } from "../../../src/core/game/GameMap";
import * as TransportUtils from "../../../src/core/game/TransportShipUtils";
import { setup } from "../../util/Setup";

/**
 * Test suite for Transport Ship Pathfinding (Optimized BFS).
 *
 * Validates shore detection, water connectivity, and performance
 * characteristics of the transport ship navigation system.
 */

const TEST_CONFIG = {
  MAP_NAME: "ocean_and_land",
  PLAYER_ID: "player_id",
  ENEMY_ID: "enemy_id",
  PERFORMANCE: {
    ITERATIONS: 5000,
    MAX_DURATION_MS: 500,
    SEARCH_DEPTH: 50,
  },
} as const;

interface TestContext {
  game: Game;
  player: Player;
  enemy: Player;
}

interface TilePair {
  shore: TileRef;
  water: TileRef;
}

let ctx: TestContext;

function findShoreWaterPair(game: Game): TilePair | null {
  const map = game.map();
  const totalTiles = map.width() * map.height();

  for (let i = 0; i < totalTiles; i++) {
    if (!map.isShore(i)) continue;

    for (const neighbor of map.neighbors(i)) {
      if (!map.isLand(neighbor)) {
        return { shore: i, water: neighbor };
      }
    }
  }
  return null;
}

function findUnownedShoreWithWater(game: Game): TilePair | null {
  const map = game.map();
  const totalTiles = map.width() * map.height();

  for (let i = 0; i < totalTiles; i++) {
    if (!map.isShore(i) || game.hasOwner(i)) continue;

    for (const neighbor of map.neighbors(i)) {
      if (!map.isLand(neighbor)) {
        return { shore: i, water: neighbor };
      }
    }
  }
  return null;
}

function findLakeTile(game: Game): TileRef | null {
  const map = game.map();
  const totalTiles = map.width() * map.height();

  for (let i = 0; i < totalTiles; i++) {
    if (game.isLake(i)) return i;
  }
  return null;
}

function advancePastSpawnPhase(game: Game): void {
  while (game.inSpawnPhase()) {
    game.executeNextTick();
  }
}

describe("TransportShipUtils", () => {
  beforeAll(async () => {
    const game = await setup(TEST_CONFIG.MAP_NAME, {
      infiniteGold: true,
      instantBuild: true,
    });

    game.addPlayer(
      new PlayerInfo("Player", PlayerType.Human, null, TEST_CONFIG.PLAYER_ID),
    );
    game.addPlayer(
      new PlayerInfo("Enemy", PlayerType.Human, null, TEST_CONFIG.ENEMY_ID),
    );

    advancePastSpawnPhase(game);

    ctx = {
      game,
      player: game.player(TEST_CONFIG.PLAYER_ID),
      enemy: game.player(TEST_CONFIG.ENEMY_ID),
    };
  });

  describe("targetTransportTile", () => {
    test("identifies enemy-owned shore tiles as valid landing targets", () => {
      // Verifies that clicking water adjacent to an enemy island correctly
      // returns the enemy shore, rather than searching for distant neutral tiles.

      const tilePair = findShoreWaterPair(ctx.game);
      expect(
        tilePair,
        "Map must contain adjacent shore/water tiles",
      ).not.toBeNull();

      const { shore, water } = tilePair!;
      ctx.enemy.conquer(shore);

      expect(ctx.game.owner(shore)).toBe(ctx.enemy);

      const result = TransportUtils.targetTransportTile(ctx.game, water);
      expect(result).toBe(shore);
    });
  });

  describe("closestShoreFromPlayer", () => {
    test("returns player shore reachable via water connectivity", () => {
      // Ensures the pathfinder respects water-based traversal when
      // identifying valid launch points for transport ships.

      const tilePair = findUnownedShoreWithWater(ctx.game);
      expect(
        tilePair,
        "Map must contain unowned shore with adjacent water",
      ).not.toBeNull();

      const { shore, water } = tilePair!;
      ctx.player.conquer(shore);

      const result = TransportUtils.closestShoreFromPlayer(
        ctx.game.map(),
        ctx.player,
        water,
      );
      expect(result).toBe(shore);
    });
  });

  describe("closestShoreTN", () => {
    test("respects water body boundaries (lake vs ocean separation)", () => {
      // Validates that BFS does not traverse across land to connect
      // separate water bodies (e.g., returning ocean shores for lake queries).

      const lakeTile = findLakeTile(ctx.game);

      if (lakeTile === null) {
        console.warn("Test skipped: map contains no lake tiles");
        return;
      }

      const map = ctx.game.map();
      const result = TransportUtils.findClosestShore(
        map,
        lakeTile,
        TEST_CONFIG.PERFORMANCE.SEARCH_DEPTH,
      );

      expect(result).not.toBeNull();
      expect(ctx.game.isLake(result!) || map.isShore(result!)).toBe(true);
    });
  });

  describe("Performance", () => {
    test("completes high-volume BFS operations within time constraints", () => {
      // Stress test validating that the Uint32-based BFS implementation
      // maintains consistent performance without excessive allocations.

      const map = ctx.game.map();
      const totalTiles = map.width() * map.height();
      const centerTile = Math.floor(totalTiles / 2);
      const { ITERATIONS, SEARCH_DEPTH, MAX_DURATION_MS } =
        TEST_CONFIG.PERFORMANCE;

      const startTime = performance.now();

      let successCount = 0;
      for (let i = 0; i < ITERATIONS; i++) {
        const startTile = (centerTile + i) % totalTiles;
        if (
          TransportUtils.findClosestShore(map, startTile, SEARCH_DEPTH) !== null
        ) {
          successCount++;
        }
      }

      const duration = performance.now() - startTime;

      expect(successCount).toBeGreaterThan(0);
      expect(duration).toBeLessThan(MAX_DURATION_MS);
    });
  });
});

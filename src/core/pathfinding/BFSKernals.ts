/**
 * @fileoverview Performance-optimized BFS kernels for tile-based pathfinding.
 *
 * Provides specialized breadth-first search implementations optimized for
 * maximum JIT compiler efficiency. Key design decisions:
 *
 * - **Generation-based visited tracking**: Uses a persistent Uint32Array with
 *   incrementing generation counters, eliminating O(N) clearing between searches.
 *
 * - **Inlined neighbor traversal**: All neighbor iteration is manually inlined
 *   to avoid function call overhead in hot loops. This is intentional and
 *   critical for performance.
 *
 * - **Cached map dimensions**: Grid dimensions are cached to avoid repeated
 *   virtual method calls during traversal.
 *
 * Memory: ~4 bytes per tile (~32MB for an 8M tile map).
 *
 * @warning These functions are NOT reentrant. BFS operations must be sequential.
 *
 * @module core/pathfinding/BFSKernels
 */

import { Game, Player } from "../game/Game";
import { GameMap, TileRef } from "../game/GameMap";

/** Maximum generation value before overflow reset. */
const MAX_GENERATION = 0xffffffff;

/** Initial generation value (0 represents "never visited"). */
const INITIAL_GENERATION = 1;

// ---------------------------------------------------------------------------
// Global BFS State
// ---------------------------------------------------------------------------
// Shared state enables zero-allocation searches by reusing the visited buffer
// across all BFS operations. The generation counter allows O(1) "clearing"
// by simply incrementing the threshold for what constitutes "visited".
// ---------------------------------------------------------------------------

let generation: number = INITIAL_GENERATION;
let visitedFlags: Uint32Array | null = null;

let cachedWidth: number = 0;
let cachedSize: number = 0;
let cachedLastRowStart: number = 0;

/**
 * Ensures the global visited buffer is allocated and sized for the given map.
 * Reinitializes if the map dimensions have changed.
 *
 * @param map - The game map to prepare for BFS operations.
 */
export function prepareBFSState(map: GameMap): void {
  const width = map.width();
  const height = map.height();
  const size = width * height;

  if (visitedFlags === null || cachedSize !== size) {
    visitedFlags = new Uint32Array(size);
    cachedWidth = width;
    cachedSize = size;
    cachedLastRowStart = (height - 1) * width;
    generation = INITIAL_GENERATION;
  }
}

/**
 * Advances the generation counter for visited tracking.
 *
 * On overflow (after ~4 billion searches), resets the counter and clears
 * the visited buffer. This is the only scenario requiring O(N) work.
 */
export function advanceGeneration(): void {
  generation++;
  if (generation === MAX_GENERATION) {
    generation = INITIAL_GENERATION;
    visitedFlags!.fill(0);
  }
}

// ---------------------------------------------------------------------------
// BFS Kernels
// ---------------------------------------------------------------------------
// Each kernel is a specialized BFS implementation with inlined traversal
// logic. The repetitive structure is intentional—extracting helper functions
// would introduce call overhead in performance-critical inner loops.
// ---------------------------------------------------------------------------

/**
 * Finds the nearest player-owned tile reachable through lake or shore tiles.
 *
 * Traversal is restricted to lake and shore terrain. When multiple targets
 * exist at the same BFS depth, returns the highest tile index for determinism.
 *
 * @param game - The game instance providing ownership and terrain data.
 * @param start - The starting tile reference.
 * @param player - The player whose territory to search for.
 * @param maxDepth - Maximum BFS depth to search.
 * @returns The nearest player-owned tile, or null if none found within depth.
 */
export function bfsFindPlayerOwnedLakeShore(
  game: Game,
  start: TileRef,
  player: Player,
  maxDepth: number,
): TileRef | null {
  prepareBFSState(game);
  advanceGeneration();

  if (game.owner(start) === player) {
    return start;
  }

  const width = cachedWidth;
  const lastRowStart = cachedLastRowStart;
  const flags = visitedFlags!;
  const gen = generation;

  flags[start] = gen;
  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (let i = 0; i < currentLevel.length; i++) {
      const tile = currentLevel[i];
      const col = tile % width;

      // Up neighbor
      if (tile >= width) {
        const neighbor = tile - width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (game.owner(neighbor) === player) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (game.isLake(neighbor) || game.isShore(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Down neighbor
      if (tile < lastRowStart) {
        const neighbor = tile + width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (game.owner(neighbor) === player) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (game.isLake(neighbor) || game.isShore(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Left neighbor
      if (col > 0) {
        const neighbor = tile - 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (game.owner(neighbor) === player) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (game.isLake(neighbor) || game.isShore(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Right neighbor
      if (col < width - 1) {
        const neighbor = tile + 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (game.owner(neighbor) === player) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (game.isLake(neighbor) || game.isShore(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    const temp = currentLevel;
    currentLevel = nextLevel;
    nextLevel = temp;
  }

  return null;
}

/**
 * Finds the nearest shore tile reachable through unowned water tiles.
 *
 * Traversal is restricted to unowned tiles (open water). Shore tiles are
 * valid targets regardless of ownership, enabling navigation to any coast.
 *
 * @param map - The game map.
 * @param start - The starting tile reference.
 * @param maxDepth - Maximum BFS depth to search.
 * @returns The nearest shore tile, or null if none found within depth.
 */
export function bfsFindClosestShore(
  map: GameMap,
  start: TileRef,
  maxDepth: number,
): TileRef | null {
  prepareBFSState(map);
  advanceGeneration();

  if (map.isShore(start)) {
    return start;
  }

  const width = cachedWidth;
  const lastRowStart = cachedLastRowStart;
  const flags = visitedFlags!;
  const gen = generation;

  flags[start] = gen;
  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (let i = 0; i < currentLevel.length; i++) {
      const tile = currentLevel[i];
      const col = tile % width;

      // Up neighbor
      if (tile >= width) {
        const neighbor = tile - width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (map.isShore(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.hasOwner(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Down neighbor
      if (tile < lastRowStart) {
        const neighbor = tile + width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (map.isShore(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.hasOwner(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Left neighbor
      if (col > 0) {
        const neighbor = tile - 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (map.isShore(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.hasOwner(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Right neighbor
      if (col < width - 1) {
        const neighbor = tile + 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (map.isShore(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.hasOwner(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    const temp = currentLevel;
    currentLevel = nextLevel;
    nextLevel = temp;
  }

  return null;
}

/**
 * Finds the nearest tile from a target set reachable via water traversal.
 *
 * Traversal is restricted to non-land tiles (water and coastal terrain).
 * This ensures pathfinding respects water body boundaries and does not
 * cross land masses.
 *
 * @param map - The game map.
 * @param start - The starting tile reference.
 * @param targetSet - Set of valid target tile references.
 * @param maxDepth - Maximum BFS depth to search.
 * @returns The nearest target tile, or null if none found within depth.
 */
export function bfsFindClosestInSet(
  map: GameMap,
  start: TileRef,
  targetSet: ReadonlySet<TileRef>,
  maxDepth: number,
): TileRef | null {
  if (targetSet.size === 0) {
    return null;
  }

  if (targetSet.has(start)) {
    return start;
  }

  prepareBFSState(map);
  advanceGeneration();

  const width = cachedWidth;
  const lastRowStart = cachedLastRowStart;
  const flags = visitedFlags!;
  const gen = generation;

  flags[start] = gen;
  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (let i = 0; i < currentLevel.length; i++) {
      const tile = currentLevel[i];
      const col = tile % width;

      // Up neighbor
      if (tile >= width) {
        const neighbor = tile - width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (targetSet.has(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.isLand(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Down neighbor
      if (tile < lastRowStart) {
        const neighbor = tile + width;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (targetSet.has(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.isLand(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Left neighbor
      if (col > 0) {
        const neighbor = tile - 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (targetSet.has(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.isLand(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }

      // Right neighbor
      if (col < width - 1) {
        const neighbor = tile + 1;
        if (flags[neighbor] !== gen) {
          flags[neighbor] = gen;
          if (targetSet.has(neighbor)) {
            if (bestMatch === null || neighbor > bestMatch)
              bestMatch = neighbor;
          }
          if (!map.isLand(neighbor)) {
            nextLevel.push(neighbor);
          }
        }
      }
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    const temp = currentLevel;
    currentLevel = nextLevel;
    nextLevel = temp;
  }

  return null;
}

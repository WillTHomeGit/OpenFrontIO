/**
 * @fileoverview Performance-optimized BFS kernels for tile-based pathfinding.
 *
 * This module provides specialized breadth-first search implementations that
 * prioritize execution speed over flexibility. Key optimizations include:
 *
 * - **Generation-based visited tracking**: Uses a persistent Uint32Array with
 *   incrementing generation counters, eliminating O(N) clearing between searches.
 *
 * - **Inlined neighbor traversal**: Avoids callback overhead by embedding
 *   traversal and target logic directly in each kernel.
 *
 * - **Cached map dimensions**: Reduces virtual method calls in hot loops.
 *
 * Memory Footprint: ~4 bytes per tile (~32MB for an 8M tile map).
 *
 * @warning These functions are NOT reentrant. BFS operations must be sequential.
 *
 * @module core/pathfinding/BFSOptimizedKernels
 */

import { Game, Player } from "../game/Game";
import { GameMap, TileRef } from "../game/GameMap";

const MAX_GENERATION = 0xffffffff;
const INITIAL_GENERATION = 1;

interface BFSState {
  visitedFlags: Uint32Array | null;
  generation: number;
  width: number;
  height: number;
  size: number;
  lastRowStart: number;
}

const state: BFSState = {
  visitedFlags: null,
  generation: INITIAL_GENERATION,
  width: 0,
  height: 0,
  size: 0,
  lastRowStart: 0,
};

/**
 * Ensures the global visited buffer matches the current map dimensions.
 * Reinitializes the buffer if the map size has changed.
 *
 * @param map - The game map to prepare for BFS operations.
 */
function prepareBFSState(map: GameMap): void {
  const width = map.width();
  const height = map.height();
  const size = width * height;

  if (state.visitedFlags === null || state.size !== size) {
    state.visitedFlags = new Uint32Array(size);
    state.width = width;
    state.height = height;
    state.size = size;
    state.lastRowStart = (height - 1) * width;
    state.generation = INITIAL_GENERATION;
  }
}

/**
 * Advances the generation counter for visited tracking.
 *
 * Wrapping is handled automatically; the buffer is cleared only when
 * the generation counter overflows (~4 billion operations).
 */
function advanceGeneration(): void {
  state.generation++;

  if (state.generation === MAX_GENERATION) {
    state.generation = INITIAL_GENERATION;
    state.visitedFlags!.fill(0);
  }
}

type NeighborCallback = (neighbor: TileRef) => void;

/**
 * Iterates over valid cardinal neighbors of a tile, invoking the callback
 * for each unvisited neighbor.
 *
 * @param tile - The source tile reference.
 * @param generation - The current BFS generation.
 * @param callback - Function to invoke for each unvisited neighbor.
 */
function forEachUnvisitedNeighbor(
  tile: TileRef,
  generation: number,
  callback: NeighborCallback,
): void {
  const { visitedFlags, width, lastRowStart } = state;
  const flags = visitedFlags!;
  const col = tile % width;

  // Up
  if (tile >= width) {
    const neighbor = tile - width;
    if (flags[neighbor] !== generation) {
      flags[neighbor] = generation;
      callback(neighbor);
    }
  }

  // Down
  if (tile < lastRowStart) {
    const neighbor = tile + width;
    if (flags[neighbor] !== generation) {
      flags[neighbor] = generation;
      callback(neighbor);
    }
  }

  // Left
  if (col > 0) {
    const neighbor = tile - 1;
    if (flags[neighbor] !== generation) {
      flags[neighbor] = generation;
      callback(neighbor);
    }
  }

  // Right
  if (col < width - 1) {
    const neighbor = tile + 1;
    if (flags[neighbor] !== generation) {
      flags[neighbor] = generation;
      callback(neighbor);
    }
  }
}

/**
 * Finds the nearest player-owned tile reachable through lake or shore tiles.
 *
 * Traversal is restricted to water-adjacent terrain (lakes and shores).
 * When multiple targets exist at the same depth, returns the highest tile index.
 *
 * @param game - The game instance.
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

  const generation = state.generation;
  state.visitedFlags![start] = generation;

  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (const tile of currentLevel) {
      forEachUnvisitedNeighbor(tile, generation, (neighbor) => {
        if (game.owner(neighbor) === player) {
          if (bestMatch === null || neighbor > bestMatch) {
            bestMatch = neighbor;
          }
        }
        if (game.isLake(neighbor) || game.isShore(neighbor)) {
          nextLevel.push(neighbor);
        }
      });
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    [currentLevel, nextLevel] = [nextLevel, currentLevel];
  }

  return null;
}

/**
 * Finds the nearest shore tile reachable through unowned water tiles.
 *
 * Traversal is restricted to unowned tiles (open water). Shore tiles are
 * valid targets regardless of ownership, enabling navigation to enemy coasts.
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

  const generation = state.generation;
  state.visitedFlags![start] = generation;

  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (const tile of currentLevel) {
      forEachUnvisitedNeighbor(tile, generation, (neighbor) => {
        if (map.isShore(neighbor)) {
          if (bestMatch === null || neighbor > bestMatch) {
            bestMatch = neighbor;
          }
        }
        if (!map.hasOwner(neighbor)) {
          nextLevel.push(neighbor);
        }
      });
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    [currentLevel, nextLevel] = [nextLevel, currentLevel];
  }

  return null;
}

/**
 * Finds the nearest tile contained within a specified target set.
 *
 * Traversal is unrestricted; all tiles are considered passable.
 * Useful for pathfinding to arbitrary collections of goal tiles.
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

  const generation = state.generation;
  state.visitedFlags![start] = generation;

  let currentLevel: TileRef[] = [start];
  let nextLevel: TileRef[] = [];

  for (let depth = 0; depth < maxDepth && currentLevel.length > 0; depth++) {
    nextLevel.length = 0;
    let bestMatch: TileRef | null = null;

    for (const tile of currentLevel) {
      forEachUnvisitedNeighbor(tile, generation, (neighbor) => {
        if (targetSet.has(neighbor)) {
          if (bestMatch === null || neighbor > bestMatch) {
            bestMatch = neighbor;
          }
        }
        nextLevel.push(neighbor);
      });
    }

    if (bestMatch !== null) {
      return bestMatch;
    }

    [currentLevel, nextLevel] = [nextLevel, currentLevel];
  }

  return null;
}

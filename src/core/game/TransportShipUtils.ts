/**
 * @fileoverview Transport ship navigation and deployment utilities.
 *
 * Provides functions for determining valid transport ship routes, spawn points,
 * and target destinations. Handles both ocean and lake navigation scenarios.
 *
 * @module core/game/TransportShipUtils
 */

import * as BFSKernels from "../pathfinding/BFSKernals";
import { Game, Player, UnitType } from "./Game";
import { GameMap, TileRef } from "./GameMap";

/**
 * Maximum BFS depth for lake shore searches.
 * Lakes are typically smaller bodies of water, so a moderate depth suffices.
 */
const LAKE_SEARCH_DEPTH = 300;

/**
 * Maximum BFS depth for terra nullius (unowned territory) searches.
 * Limited depth for performance when targeting neutral shores.
 */
const TERRA_NULLIUS_SEARCH_DEPTH = 50;

/**
 * Maximum BFS depth for player border shore searches.
 * Higher depth to ensure connectivity across large maps.
 */
const BORDER_SHORE_SEARCH_DEPTH = 10000;

/**
 * Sampling interval divisor for candidate shore tile selection.
 * Higher values produce fewer samples from the border shore set.
 */
const SHORE_SAMPLING_DIVISOR = 50;

/**
 * Minimum sampling interval to prevent excessive candidates on small borders.
 */
const MIN_SAMPLING_INTERVAL = 10;

/**
 * Determines if a transport ship can be built to reach the specified tile.
 *
 * Validates unit limits, target accessibility, diplomatic status, and
 * water connectivity between source and destination.
 *
 * @param game - The game instance.
 * @param player - The player attempting to build the transport ship.
 * @param tile - The target destination tile.
 * @returns The spawn tile reference if buildable, or false if not permitted.
 */
export function canBuildTransportShip(
  game: Game,
  player: Player,
  tile: TileRef,
): TileRef | false {
  if (
    player.unitCount(UnitType.TransportShip) >= game.config().boatMaxNumber()
  ) {
    return false;
  }

  const destination = targetTransportTile(game, tile);
  if (destination === null) {
    return false;
  }

  const targetOwner = game.owner(tile);
  if (targetOwner === player) {
    return false;
  }

  if (targetOwner.isPlayer() && !player.canAttackPlayer(targetOwner)) {
    return false;
  }

  if (game.isOceanShore(destination)) {
    return canBuildOceanTransport(
      game,
      player,
      targetOwner as Player,
      tile,
      destination,
    );
  }

  return canBuildLakeTransport(game, player, destination);
}

/**
 * Validates ocean transport ship construction requirements.
 */
function canBuildOceanTransport(
  game: Game,
  player: Player,
  targetOwner: Player,
  tile: TileRef,
  destination: TileRef,
): TileRef | false {
  const playerBordersOcean = hasOceanBorder(game, player);
  const targetBordersOcean = game.hasOwner(tile)
    ? hasOceanBorder(game, targetOwner)
    : true;

  if (playerBordersOcean && targetBordersOcean) {
    return resolveTransportSpawn(game, player, destination);
  }

  return false;
}

/**
 * Validates lake transport ship construction requirements.
 */
function canBuildLakeTransport(
  game: Game,
  player: Player,
  destination: TileRef,
): TileRef | false {
  const spawn = BFSKernels.bfsFindPlayerOwnedLakeShore(
    game,
    destination,
    player,
    LAKE_SEARCH_DEPTH,
  );

  if (spawn === null) {
    return false;
  }

  return resolveTransportSpawn(game, player, spawn);
}

/**
 * Checks if a player has any border tiles adjacent to the ocean.
 */
function hasOceanBorder(game: Game, player: Player): boolean {
  for (const tile of player.borderTiles()) {
    if (game.isOceanShore(tile)) {
      return true;
    }
  }
  return false;
}

/**
 * Resolves the spawn point for a transport ship given a target shore.
 */
function resolveTransportSpawn(
  game: Game,
  player: Player,
  targetTile: TileRef,
): TileRef | false {
  if (!game.isShore(targetTile)) {
    return false;
  }

  const spawn = closestShoreFromPlayer(game, player, targetTile);
  return spawn ?? false;
}

/**
 * Finds source and destination shore tiles for an ocean transport route.
 *
 * @param game - The game instance.
 * @param sourcePlayer - The player initiating the transport.
 * @param tile - The target tile reference.
 * @returns A tuple of [source shore, destination shore], with null for unreachable endpoints.
 */
export function sourceDstOceanShore(
  game: Game,
  sourcePlayer: Player,
  tile: TileRef,
): [TileRef | null, TileRef | null] {
  const targetOwner = game.owner(tile);
  const sourceTile = closestShoreFromPlayer(game, sourcePlayer, tile);

  let destinationTile: TileRef | null;
  if (targetOwner.isPlayer()) {
    destinationTile = closestShoreFromPlayer(game, targetOwner as Player, tile);
  } else {
    destinationTile = findClosestShore(game, tile, TERRA_NULLIUS_SEARCH_DEPTH);
  }

  return [sourceTile, destinationTile];
}

/**
 * Determines the target shore tile for a transport ship given a clicked tile.
 *
 * For player-owned tiles, returns the owner's closest shore.
 * For unowned tiles, searches for the nearest shore via water traversal.
 *
 * @param game - The game instance.
 * @param tile - The clicked tile reference.
 * @returns The target shore tile, or null if no valid target exists.
 */
export function targetTransportTile(game: Game, tile: TileRef): TileRef | null {
  const tileOwner = game.playerBySmallID(game.ownerID(tile));

  if (tileOwner.isPlayer()) {
    return closestShoreFromPlayer(game, tileOwner as Player, tile);
  }

  return findClosestShore(game, tile, TERRA_NULLIUS_SEARCH_DEPTH);
}

/**
 * Finds the closest player-owned shore tile reachable via water from the target.
 *
 * Uses BFS through water tiles to find the nearest shore belonging to the
 * specified player, ensuring proper water connectivity.
 *
 * @param map - The game map.
 * @param player - The player whose shores to search.
 * @param target - The target tile to measure distance from.
 * @returns The closest reachable shore tile, or null if none found.
 */
export function closestShoreFromPlayer(
  map: GameMap,
  player: Player,
  target: TileRef,
): TileRef | null {
  const borderShores = collectPlayerBorderShores(player, map);

  if (borderShores.size === 0) {
    return null;
  }

  return BFSKernels.bfsFindClosestInSet(
    map,
    target,
    borderShores,
    BORDER_SHORE_SEARCH_DEPTH,
  );
}

/**
 * Collects all shore tiles on a player's border.
 */
function collectPlayerBorderShores(player: Player, map: GameMap): Set<TileRef> {
  const shores = new Set<TileRef>();

  for (const tile of player.borderTiles()) {
    if (map.isShore(tile)) {
      shores.add(tile);
    }
  }

  return shores;
}

/**
 * Finds the best shore tile for deploying a transport ship toward a target.
 *
 * @param game - The game instance.
 * @param player - The player deploying the transport.
 * @param target - The target destination tile.
 * @returns The optimal deployment shore tile, or false if none valid.
 */
export function bestShoreDeploymentSource(
  game: Game,
  player: Player,
  target: TileRef,
): TileRef | false {
  const transportTarget = targetTransportTile(game, target);
  if (transportTarget === null) {
    return false;
  }

  const bestShore = closestShoreFromPlayer(game, player, transportTarget);
  if (bestShore === null) {
    return false;
  }

  if (!game.isShore(bestShore) || game.owner(bestShore) !== player) {
    return false;
  }

  return bestShore;
}

/**
 * Generates a set of candidate shore tiles for transport ship deployment.
 *
 * Returns a diverse selection including:
 * - The BFS-closest shore to the target
 * - Extreme positions (min/max X and Y coordinates)
 * - Uniform samples from the border shore set
 *
 * @param game - The game instance.
 * @param player - The player whose shores to consider.
 * @param target - The target tile reference.
 * @returns Array of unique candidate shore tiles.
 */
export function candidateShoreTiles(
  game: Game,
  player: Player,
  target: TileRef,
): TileRef[] {
  const borderShores = collectBorderShoresWithExtremums(player, game);

  if (borderShores.tiles.length === 0) {
    return [];
  }

  const borderShoreSet = new Set(borderShores.tiles);
  const closestByBFS = BFSKernels.bfsFindClosestInSet(
    game,
    target,
    borderShoreSet,
    BORDER_SHORE_SEARCH_DEPTH,
  );

  return buildUniqueCandidateList(borderShores, closestByBFS);
}

interface BorderShoreData {
  tiles: TileRef[];
  minX: TileRef | null;
  minY: TileRef | null;
  maxX: TileRef | null;
  maxY: TileRef | null;
}

/**
 * Collects border shore tiles and identifies extremum positions.
 */
function collectBorderShoresWithExtremums(
  player: Player,
  map: GameMap,
): BorderShoreData {
  const result: BorderShoreData = {
    tiles: [],
    minX: null,
    minY: null,
    maxX: null,
    maxY: null,
  };

  let minXCoord = Infinity;
  let minYCoord = Infinity;
  let maxXCoord = -Infinity;
  let maxYCoord = -Infinity;

  for (const tile of player.borderTiles()) {
    if (!map.isShore(tile)) continue;

    result.tiles.push(tile);

    const x = map.x(tile);
    const y = map.y(tile);

    if (x < minXCoord || (x === minXCoord && tile > (result.minX ?? -1))) {
      minXCoord = x;
      result.minX = tile;
    }

    if (y < minYCoord || (y === minYCoord && tile > (result.minY ?? -1))) {
      minYCoord = y;
      result.minY = tile;
    }

    if (x > maxXCoord || (x === maxXCoord && tile > (result.maxX ?? -1))) {
      maxXCoord = x;
      result.maxX = tile;
    }

    if (y > maxYCoord || (y === maxYCoord && tile > (result.maxY ?? -1))) {
      maxYCoord = y;
      result.maxY = tile;
    }
  }

  return result;
}

/**
 * Builds a deduplicated list of candidate tiles from extremums and samples.
 */
function buildUniqueCandidateList(
  borderShores: BorderShoreData,
  closestTile: TileRef | null,
): TileRef[] {
  const seen = new Set<TileRef>();
  const candidates: TileRef[] = [];

  const addUnique = (tile: TileRef | null): void => {
    if (tile !== null && !seen.has(tile)) {
      seen.add(tile);
      candidates.push(tile);
    }
  };

  addUnique(closestTile);
  addUnique(borderShores.minX);
  addUnique(borderShores.minY);
  addUnique(borderShores.maxX);
  addUnique(borderShores.maxY);

  const { tiles } = borderShores;
  const samplingInterval = Math.max(
    MIN_SAMPLING_INTERVAL,
    Math.ceil(tiles.length / SHORE_SAMPLING_DIVISOR),
  );

  for (let i = 0; i < tiles.length; i += samplingInterval) {
    addUnique(tiles[i]);
  }

  return candidates;
}

/**
 * Finds the nearest shore tile reachable via water from a given position.
 *
 * Used for targeting unowned (terra nullius) coastal areas.
 *
 * @param map - The game map.
 * @param tile - The starting tile reference.
 * @param searchDepth - Maximum BFS search depth.
 * @returns The nearest shore tile, or null if none found within depth.
 */
export function findClosestShore(
  map: GameMap,
  tile: TileRef,
  searchDepth: number,
): TileRef | null {
  return BFSKernels.bfsFindClosestShore(map, tile, searchDepth);
}

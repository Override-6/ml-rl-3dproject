use crate::map::{ComponentId, ComponentType, MapComponent, ObstacleFace, ObstacleFaceSet};
use crate::player::{LASER_LENGTH, PLAYER_LASER_COUNT, Player, PlayerId};
use crate::simulation::SimulationState;
use bevy::color::palettes::basic::{BLUE, GREEN};
use bevy::color::palettes::css::{GRAY, ORANGE, YELLOW};
use bevy::prelude::{
    BevyError, Children, Component, Entity, Gizmos, GlobalTransform, Query, Res, ResMut, With,
};
use bevy::utils::Parallel;
use bevy_math::Vec3;
use bevy_rapier3d::plugin::ReadRapierContext;
use bevy_rapier3d::prelude::QueryFilter;
use std::collections::HashMap;
use crate::component::player_character::PLAYER_HEIGHT;

/// Collection of lasers sensors that allows the player to understand where are the elements of the scene, ant which distance, and which kind of element it is.
#[derive(Component)]
pub struct PlayerVibrissae {
    /// The lasers directions of the player's sensor
    pub lasers: [LaserSensor; PLAYER_LASER_COUNT],
}

impl From<[Vec3; PLAYER_LASER_COUNT]> for PlayerVibrissae {
    fn from(value: [Vec3; PLAYER_LASER_COUNT]) -> Self {
        let lasers = value
            .into_iter()
            .map(|direction| LaserSensor {
                direction,
                hit: None,
            })
            .collect::<Vec<LaserSensor>>();
        Self {
            lasers: lasers.try_into().unwrap(),
        }
    }
}

impl PlayerVibrissae {
    pub fn get_directional_laser(&mut self) -> &mut LaserSensor {
        &mut self.lasers[0]
    }
}

#[derive(Debug, Clone)]
pub struct LaserSensor {
    pub direction: Vec3,
    pub hit: Option<LaserHit>,
}

#[derive(Debug, Clone)]
pub struct LaserHit {
    pub comp_type: ComponentType,
    pub distance: f32,
    pub point: Vec3,
}

pub(crate) fn collect_descendants(
    entity: Entity,
    children_query: &Query<&Children>,
    collected: &mut Vec<Entity>,
) {
    collected.push(entity);
    if let Ok(children) = children_query.get(entity) {
        for &child in children.iter() {
            collect_descendants(child, children_query, collected);
        }
    }
}

pub fn update_all_vibrissae_lasers(
    mut query: Query<(Entity, &mut PlayerVibrissae, &GlobalTransform, &Player), With<Player>>,
    entity_type_query: Query<(&ComponentType, Option<&MapComponent>)>,
    children_query: Query<&Children>,
    player_query: Query<&Player>,
    rapier_ctx: ReadRapierContext,
    mut sim: ResMut<SimulationState>,
) -> bevy::prelude::Result<(), BevyError> {
    let mut queue: Parallel<Vec<(ComponentId, PlayerId, ObstacleFaceSet)>> = Parallel::default();

    query.par_iter_mut().for_each_init(
        || queue.borrow_local_mut(),
        |results, (entity, mut vibrissae, player_gt, player)| {
            let map = update_vibrissae_lasers(
                entity,
                vibrissae.as_mut(),
                player_gt,
                entity_type_query,
                player_query,
                children_query,
                &rapier_ctx,
            )
            .unwrap();

            results.extend(map.into_iter().map(|(comp, set)| (comp, player.id, set)));
        },
    );

    for (component_id, player_id, set) in queue.iter_mut().flat_map(|x| x) {
        sim.current_step_rt_info_mut().discovered_obstacles[*player_id].faces[*component_id] |= *set;
    }

    Ok(())
}

pub fn update_vibrissae_lasers(
    entity: Entity,
    vibrissae: &mut PlayerVibrissae,
    player_gt: &GlobalTransform,
    entity_type_query: Query<(&ComponentType, Option<&MapComponent>)>,
    player_query: Query<&Player>,
    children_query: Query<&Children>,
    rapier_ctx: &ReadRapierContext,
) -> Result<HashMap<ComponentId, ObstacleFaceSet>, BevyError> {
    let mut origin = player_gt.translation();

    origin.y += PLAYER_HEIGHT;

    let rotation = player_gt.rotation();

    let mut excluded_entities = Vec::new();
    collect_descendants(entity, &children_query, &mut excluded_entities);

    let filter_predicate = |e| player_query.get(e).is_err() && !excluded_entities.contains(&e);
    let filter = QueryFilter::default().predicate(&filter_predicate);

    let rapier_ctx = rapier_ctx.single()?;

    let mut discovered_faces: HashMap<ComponentId, ObstacleFaceSet> = HashMap::default();

    for laser in &mut vibrissae.lasers {
        let direction = rotation * laser.direction;

        let result = rapier_ctx.cast_ray_and_get_normal(
            origin,
            direction,
            LASER_LENGTH.into(),
            true,
            filter,
        );

        if let Some((entity, intersection)) = result &&
            let Ok((&component_type, component)) = entity_type_query.get(entity) {

            let normal = intersection.normal;

            let face = if normal.z.abs() > normal.x.abs() {
                if normal.z > 0.0 {
                    ObstacleFace::South // +Z
                } else {
                    ObstacleFace::North // -Z
                }
            } else if normal.x > 0.0 {
                ObstacleFace::East // +X
            } else {
                ObstacleFace::West // -X
            };

            if let Some(component) = component {
                discovered_faces
                    .entry(component.0)
                    .and_modify(|set| *set |= face)
                    .or_insert(face as ObstacleFaceSet);
            }

            let toi = f32::from(intersection.time_of_impact);
            let point = origin + direction * toi;

            laser.hit = Some(LaserHit {
                distance: toi,
                comp_type: component_type,
                point,
            })
        } else {
            laser.hit = None;
        }
    }

    Ok(discovered_faces)
}

pub fn debug_render_lasers(
    mut gizmos: Gizmos,
    query: Query<(&PlayerVibrissae, &GlobalTransform, &Player)>,
    sim: Res<SimulationState>,
) {
    for (vibrissae, player_gt, player) in query.iter() {
        let origin = player_gt.translation();
        let direction = player_gt.rotation();

        if player.freeze {
            continue;
        }

        for laser in vibrissae.lasers.iter() {
            let Some(hit) = &laser.hit else {
                gizmos.line(
                    origin,
                    origin + (direction * laser.direction) * LASER_LENGTH,
                    BLUE,
                );
                continue;
            };

            let hit_color = match hit.comp_type {
                ComponentType::Ground => YELLOW,
                ComponentType::Obstacle => ORANGE,
                ComponentType::Objective => GREEN,
                ComponentType::Wall => GRAY,
                ComponentType::None => unreachable!(),
            };

            gizmos.line(
                origin,
                origin + (direction * laser.direction) * LASER_LENGTH,
                hit_color,
            );

            gizmos.sphere(hit.point, 3.0, hit_color);
        }
        if !sim.debug.print_all_lasers {
            // if we dont want to print all lasers, print only the lasers of the first player (which is the observed player)
            return;
        }
    }
}

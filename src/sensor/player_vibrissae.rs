use crate::map::ComponentType;
use crate::player::Player;
use crate::simulation::GameLayer;
use avian3d::prelude::{DebugRender, RayHits};
use avian3d::spatial_query::{RayCaster, SpatialQueryFilter};
use bevy::color::palettes::basic::RED;
use bevy::prelude::{BevyError, Children, Commands, Component, Entity, Gizmos, Query, With};
use bevy_math::{Dir3, Vec3};
use std::iter::once;

pub const LASER_LENGTH: f32 = 3000.0;

/// Collection of lasers sensors that allows the player to understand where are the elements of the scene, ant which distance, and which kind of element it is.
#[derive(Component)]
pub struct PlayerVibrissae {
    /// The other lasers directions of the player's sensor.
    /// First laser will always be the ground sensor (automatically added)
    lasers: Vec<LaserSensor>,
}

impl PlayerVibrissae {
    pub fn from_vec(set: Vec<Dir3>) -> Self {
        let lasers = once(Dir3::NEG_Y)
            .chain(set.into_iter())
            .map(|direction| LaserSensor {
                direction,
                hit: None,
            })
            .collect();
        Self { lasers }
    }

    pub fn lasers(&self) -> &[LaserSensor] {
        &self.lasers
    }

    pub fn ground_sensor(&self) -> &LaserSensor {
        &self.lasers[0]
    }
}

#[derive(Component, Debug)]
pub struct LaserId(usize);

pub struct LaserSensor {
    pub direction: Dir3,
    pub hit: Option<LaserHit>,
}

#[derive(Debug, Clone)]
pub struct LaserHit {
    pub entity_type: ComponentType,
    pub distance: f32,
    // may not be given to AI Model, more for debug purposes
    pub point: Vec3,
}

fn collect_descendants(
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

pub fn spawn_all_lasers(
    mut query: Query<(Entity, &mut PlayerVibrissae), With<Player>>,
    mut commands: Commands,
) {
    let filter = SpatialQueryFilter::from_mask(GameLayer::World);

    for (entity, player_vibrissae) in query.iter_mut() {
        let mut entity = commands.entity(entity);
        for (id, laser) in player_vibrissae.lasers.iter().enumerate() {
            entity.with_children(|parent| {
                parent.spawn((
                    LaserId(id),
                    RayCaster::new(Vec3::ZERO, laser.direction)
                        .with_max_hits(1)
                        .with_max_distance(LASER_LENGTH)
                        .with_query_filter(filter.clone()),
                    DebugRender::none()
                ));
            });
        }
    }
}

pub fn update_all_vibrissae_lasers(
    mut query: Query<(&mut PlayerVibrissae, &Children), With<Player>>,
    entity_type_query: Query<&ComponentType>,
    ray_cast_query: Query<(&RayCaster, &RayHits, &LaserId)>,
) -> bevy::prelude::Result<(), BevyError> {
    query
        .par_iter_mut()
        .for_each(|(mut vibrissae, children)| {
            update_vibrissae_lasers(
                children,
                vibrissae.as_mut(),
                entity_type_query,
                ray_cast_query,
            )
            .unwrap()
        });

    Ok(())
}

pub fn update_vibrissae_lasers(
    children: &Children,
    vibrissae: &mut PlayerVibrissae,
    entity_type_query: Query<&ComponentType>,
    ray_cast_query: Query<(&RayCaster, &RayHits, &LaserId)>,
) -> Result<(), BevyError> {
    for (ray, hits, id) in children
        .into_iter()
        .filter_map(|x| ray_cast_query.get(*x).ok())
    {
        let laser = &mut vibrissae.lasers[id.0];

        if let Some(hit) = hits.iter().next() {
            let toi = f32::from(hit.distance);
            let point = (ray.global_origin()) + (ray.global_direction()) * toi;

            let component_type = entity_type_query
                .get(hit.entity)
                .copied()
                .unwrap_or(ComponentType::Unknown);

            laser.hit = Some(LaserHit {
                distance: toi,
                entity_type: component_type,
                point,
            })
        } else {
            laser.hit = None;
        }
    }

    Ok(())
}

pub fn debug_render_lasers(mut gizmos: Gizmos, query: Query<&PlayerVibrissae, With<Player>>) {
    // Only debug on controlled player (which is first player)
    let vibrissae = query.iter().next().unwrap();

    for laser in vibrissae.lasers.iter() {
        let Some(hit) = &laser.hit else {
            continue;
        };

        gizmos.sphere(hit.point, 3.0, RED);
    }
}

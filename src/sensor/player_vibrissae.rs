use crate::component::player_character::PlayerCharacter;
use crate::map::ComponentType;
use bevy::color::palettes::basic::{BLUE, RED};
use bevy::prelude::{BevyError, ChildOf, Children, Component, Entity, Gizmos, GlobalTransform, Query, With};
use bevy_math::Vec3;
use bevy_rapier3d::plugin::ReadRapierContext;
use bevy_rapier3d::prelude::QueryFilter;

pub const LASER_LENGTH: f32 = 3000.0;


/// Collection of lasers sensors that allows the player to understand where are the elements of the scene, ant which distance, and which kind of element it is.
#[derive(Component)]
pub struct PlayerVibrissae {
    /// The lasers directions of the player's sensor
    pub lasers: Vec<LaserSensor>,
}

impl PlayerVibrissae {
    pub fn from_vec(set: Vec<Vec3>) -> Self {
        let lasers = set
            .into_iter()
            .map(|direction| LaserSensor {
                direction,
                hit: None,
            })
            .collect();
        Self { lasers }
    }
}

pub struct LaserSensor {
    pub direction: Vec3,
    pub hit: Option<LaserHit>,
}

pub struct LaserHit {
    entity_type: ComponentType,
    distance: f32,
    // may not be given to AI Model, more for debug purposes
    point: Vec3,
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

pub fn update_vibrissae_lasers(
    mut query: Query<(&mut PlayerVibrissae, &GlobalTransform, &ChildOf), With<PlayerCharacter>>,
    entity_type_query: Query<&ComponentType>,
    children_query: Query<&Children>,
    rapier_ctx: ReadRapierContext,
) -> bevy::prelude::Result<(), BevyError> {
    let (mut vibrissae, player_gt, co) = query.single_mut()?;

    let origin = player_gt.translation();
    let rotation = player_gt.rotation();

    let mut excluded_entities = Vec::new();
    collect_descendants(co.parent(), &children_query, &mut excluded_entities);


    let filter_predicate = |e| !excluded_entities.contains(&e);
    let filter = QueryFilter::default().predicate(&filter_predicate);

    let rapier_ctx = rapier_ctx.single()?;

    for laser in &mut vibrissae.lasers {
        let direction = rotation * laser.direction;

        let result = rapier_ctx.cast_ray(origin, direction, LASER_LENGTH.into(), false, filter);

        if let Some((entity, toi)) = result {
            let toi = f32::from(toi);
            let point = origin + direction * toi;

            let component_type = entity_type_query
                .get(entity)
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

pub fn debug_render_lasers(
    mut gizmos: Gizmos,
    query: Query<(&PlayerVibrissae, &GlobalTransform), With<PlayerCharacter>>,
) {
    let (vibrissae, player_gt) = query.single().unwrap();

    let origin = player_gt.translation();
    let direction = player_gt.rotation();

    for laser in vibrissae.lasers.iter() {
        let Some(hit) = &laser.hit else {
            gizmos.line(
                origin,
                origin + (direction * laser.direction) * LASER_LENGTH,
                BLUE,
            );
            continue;
        };


        gizmos.line(
            origin,
            origin + (direction * laser.direction) * LASER_LENGTH,
            RED,
        );

        gizmos.sphere(hit.point, 3.0, RED);
    }
}

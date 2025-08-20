use crate::map::ComponentType;
use crate::player::{Player, LASER_LENGTH, PLAYER_LASER_COUNT};
use crate::simulation::SimulationState;
use bevy::color::palettes::basic::{BLUE, GREEN};
use bevy::color::palettes::css::{GRAY, ORANGE, YELLOW};
use bevy::prelude::{BevyError, Children, Component, Entity, Gizmos, GlobalTransform, Query, Res, With};
use bevy_math::Vec3;
use bevy_rapier3d::plugin::ReadRapierContext;
use bevy_rapier3d::prelude::QueryFilter;


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
                direction: direction.clone(),
                hit: None,
            })
            .collect::<Vec<LaserSensor>>();
        Self {
            lasers: lasers.try_into().unwrap(),
        }
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

pub fn update_all_vibrissae_lasers(
    mut query: Query<(Entity, &mut PlayerVibrissae, &GlobalTransform), With<Player>>,
    entity_type_query: Query<&ComponentType>,
    children_query: Query<&Children>,
    player_query: Query<&Player>,
    rapier_ctx: ReadRapierContext,
) -> bevy::prelude::Result<(), BevyError> {
    query
        .par_iter_mut()
        .for_each(|(entity, mut vibrissae, player_gt)| {
            update_vibrissae_lasers(
                entity,
                vibrissae.as_mut(),
                player_gt,
                entity_type_query,
                player_query,
                children_query,
                &rapier_ctx,
            )
            .unwrap()
        });

    Ok(())
}

pub fn update_vibrissae_lasers(
    entity: Entity,
    vibrissae: &mut PlayerVibrissae,
    player_gt: &GlobalTransform,
    entity_type_query: Query<&ComponentType>,
    player_query: Query<&Player>,
    children_query: Query<&Children>,
    rapier_ctx: &ReadRapierContext,
) -> Result<(), BevyError> {
    let origin = player_gt.translation();
    let rotation = player_gt.rotation();

    let mut excluded_entities = Vec::new();
    collect_descendants(entity, &children_query, &mut excluded_entities);

    let filter_predicate = |e| player_query.get(e).is_err() && !excluded_entities.contains(&e);
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
                comp_type: component_type,
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
    query: Query<(&PlayerVibrissae, &GlobalTransform, &Player)>,
    sim: Res<SimulationState>
) {
    for (vibrissae, player_gt, player) in query.iter()  {
        let origin = player_gt.translation();
        let direction = player_gt.rotation();

        if player.freeze {
            continue
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
                ComponentType::Object => ORANGE,
                ComponentType::Objective => GREEN,
                ComponentType::Unknown => GRAY,
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
            return
        }
    }


}

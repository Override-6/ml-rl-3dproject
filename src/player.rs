use crate::map::{ComponentType, MAP_SQUARE_SIZE};
use bevy::prelude::{Component, EventReader, Query, With, Without};
use bevy_math::Vec3;
use bevy_rapier3d::dynamics::{RapierRigidBodyHandle, Velocity};
use bevy_rapier3d::na::UnitQuaternion;
use bevy_rapier3d::pipeline::CollisionEvent;
use bevy_rapier3d::plugin::WriteRapierContext;
use bevy_rapier3d::rapier::prelude as rapier;
use rand::Rng;
use std::f32::consts::SQRT_2;

pub type PlayerId = usize;

#[derive(Component)]
pub struct Player {
    pub id: PlayerId,
    pub freeze: bool,
    pub objective_reached_at_timestep: i32,
    pub touching_obstacle: bool,
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            freeze: false,
            objective_reached_at_timestep: -1,
            touching_obstacle: false,
        }
    }
}

pub const PLAYER_SPEED: f32 = 200.0;

pub const PLAYER_JUMP_SPEED: f32 = 400.0;

pub const PLAYER_TURN_SPEED: f32 = 10.0;

pub const PLAYER_LASERS: [Vec3; 5] = [Vec3::NEG_Y, Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::NEG_X];

pub const PLAYER_LASER_COUNT: usize = PLAYER_LASERS.len();
pub const LASER_LENGTH: f32 = (MAP_SQUARE_SIZE * 2.0) * SQRT_2;

pub const PLAYER_SPAWN_SAFE_DISTANCE: f32 = 50.0;

pub fn get_player_position() -> Vec3 {
    Vec3::new(MAP_SQUARE_SIZE - (PLAYER_SPAWN_SAFE_DISTANCE / 2.0), 10.0f32, MAP_SQUARE_SIZE - (PLAYER_SPAWN_SAFE_DISTANCE / 2.0))
}

pub fn reset_players(
    mut players: Query<(&RapierRigidBodyHandle, &mut Player, Option<&mut Velocity>), With<Player>>,
    mut rapier_ctx: WriteRapierContext,
) -> bevy::prelude::Result<()> {
    let mut context = rapier_ctx.single_mut()?;
    let mut rng = rand::rng();

    for (handle, mut player, vel_opt) in players.iter_mut() {
        let handle = handle.0;

        if let Some(rb) = context.rigidbody_set.bodies.get_mut(handle) {

            rb.set_translation(
                // rapier::Vector::new(
                //     rng.random_range(-MAP_SQUARE_SIZE..MAP_SQUARE_SIZE),
                //     10.0f32,
                //     rng.random_range(-MAP_SQUARE_SIZE..MAP_SQUARE_SIZE),
                // ),
                rapier::Vector::from(get_player_position()),
                false,
            );

            let yaw_deg = rng.random_range(0.0f32..360.0f32);
            let yaw = yaw_deg.to_radians();
            let unit_q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
            rb.set_rotation(unit_q, false);

            rb.set_linvel(rapier::Vector::zeros(), false);
            rb.set_angvel(rapier::Vector::zeros(), false);

            rb.set_body_type(rapier::RigidBodyType::Dynamic, false);
        }

        // Reset Player component state
        player.freeze = false;
        player.objective_reached_at_timestep = -1;
        player.touching_obstacle = false;

        // Reset optional velocity component
        if let Some(mut vel) = vel_opt {
            *vel = Velocity::default();
        }
    }

    Ok(())
}


pub fn player_collision_detection(
    mut collision_events: EventReader<CollisionEvent>,
    mut player_q: Query<&mut Player>,
    component_type_q: Query<&ComponentType, Without<Player>>,
) -> Result<(), bevy::prelude::BevyError> {
    for (event, _) in collision_events.par_read() {
        match event {
            CollisionEvent::Started(e1, e2, _flags) => {
                let component = component_type_q
                    .get(*e1)
                    .or_else(|_| component_type_q.get(*e2));
                if !component.is_ok_and(|&c| c == ComponentType::Obstacle) {
                    continue;
                }
                if let Ok(mut player) = player_q.get_mut(*e1) {
                    player.touching_obstacle = true;
                    continue;
                }
                if let Ok(mut player) = player_q.get_mut(*e2) {
                    player.touching_obstacle = true;
                    continue;
                }
                // update per-entity contact counters or set a 'touching' flag
            }
            CollisionEvent::Stopped(e1, e2, _flags) => {
                let component = component_type_q
                    .get(*e1)
                    .or_else(|_| component_type_q.get(*e2));
                if !component.is_ok_and(|&c| c == ComponentType::Obstacle) {
                    continue;
                }
                if let Ok(mut player) = player_q.get_mut(*e1) {
                    player.touching_obstacle = false;
                    continue;
                }
                if let Ok(mut player) = player_q.get_mut(*e2) {
                    player.touching_obstacle = false;
                    continue;
                }
            }
        }
    }

    Ok(())
}

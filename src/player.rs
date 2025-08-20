use std::f32::consts::SQRT_2;
use crate::simulation::SimulationState;
use bevy::prelude::{Component, Query, ResMut, With};
use bevy_math::Vec3;
use bevy_rapier3d::dynamics::{RapierRigidBodyHandle, Velocity};
use bevy_rapier3d::na::UnitQuaternion;
use bevy_rapier3d::plugin::WriteRapierContext;
use bevy_rapier3d::rapier::prelude as rapier;
use rand::Rng;
use crate::map::MAP_SQUARE_SIZE;

pub type PlayerId = usize;

#[derive(Component)]
pub struct Player {
    pub id: PlayerId,
    pub freeze: bool,
    pub objective_reached_at_timestep: i32,
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            freeze: false,
            objective_reached_at_timestep: -1,
        }
    }
}

pub const PLAYER_SPEED: f32 = 400.0;

pub const PLAYER_JUMP_SPEED: f32 = 400.0;

pub const PLAYER_TURN_SPEED: f32 = 10.0;

pub const PLAYER_LASERS: [Vec3; 5] = [Vec3::NEG_Y, Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::NEG_X];

pub const PLAYER_LASER_COUNT: usize = PLAYER_LASERS.len();
pub const LASER_LENGTH: f32 = (MAP_SQUARE_SIZE * 2.0) * SQRT_2;

pub fn reset_players(
    mut players: Query<(&RapierRigidBodyHandle, &mut Player, Option<&mut Velocity>), With<Player>>,
    mut rapier_ctx: WriteRapierContext,
) -> bevy::prelude::Result<()> {
    let mut context = rapier_ctx.single_mut()?;
    let mut rng = rand::rng();

    for (handle, mut player, vel_opt) in players.iter_mut() {
        let handle = handle.0;

        if let Some(rb) = context.rigidbody_set.bodies.get_mut(handle) {
            rb.set_body_type(rapier::RigidBodyType::Fixed, true);

            rb.set_translation(
                rapier::Vector::new(
                    rng.random_range(-300.0f32..300.0f32),
                    10.0f32,
                    rng.random_range(-300.0f32..300.0f32),
                ),
                /*wake*/ false,
            );

            let yaw_deg = rng.random_range(0.0f32..360.0f32);
            let yaw = yaw_deg.to_radians();
            let unit_q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
            rb.set_rotation(unit_q, /*wake*/ false);

            // 3) Zero rapier velocities (defensive)
            rb.set_linvel(rapier::Vector::zeros(), /*wake*/ false);
            rb.set_angvel(rapier::Vector::zeros(), /*wake*/ false);

            // 4) Restore to Dynamic, but DON'T wake the body (false)
            //    This keeps the body asleep until you explicitly wake a subset per-frame.
            rb.set_body_type(rapier::RigidBodyType::Dynamic, /*wake*/ false);
        }

        // 5) Reset gameplay state
        player.freeze = false;
        player.objective_reached_at_timestep = -1;

        // 6) Also reset the Bevy Velocity component (if present) so ECS view stays consistent
        if let Some(mut vel) = vel_opt {
            *vel = Velocity::default();
        }
    }

    Ok(())
}

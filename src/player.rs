use bevy::prelude::Component;
use bevy_math::Vec3;

pub type PlayerId = usize;

#[derive(Component)]
pub struct Player {
    pub id: PlayerId,
    pub freeze: bool,
    pub objective_reached_at_timestep: i32
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            freeze: false,
            objective_reached_at_timestep: -1
        }
    }
}

pub const PLAYER_SPEED: f32 = 200.0;

pub const PLAYER_JUMP_SPEED: f32 = 400.0;

pub const PLAYER_TURN_SPEED: f32 = 10.0;

pub const PLAYER_LASERS: [Vec3; 5] = [Vec3::NEG_Y, Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::NEG_X];

pub const PLAYER_LASER_COUNT: usize = PLAYER_LASERS.len();

use bevy::prelude::Component;
use bevy_math::Vec3;

pub type PlayerId = usize;

#[derive(Component)]
pub struct Player {
    pub id: PlayerId,
    pub freeze: bool,
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            freeze: false,
        }
    }
}

pub const PLAYER_SPEED: f32 = 200.0;

pub const PLAYER_JUMP_SPEED: f32 = 400.0;

pub const PLAYER_TURN_SPEED: f32 = 10.0;

pub const PLAYER_LASERS: [Vec3; 5] = [Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::NEG_X, Vec3::NEG_Y];

pub const PLAYER_LASER_COUNT: usize = PLAYER_LASERS.len();

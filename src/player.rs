use bevy::prelude::Component;

#[derive(Component)]
pub struct Player;

pub const PLAYER_TURN_SPEED: f32 = 3.0;
pub const PLAYER_JUMP_SPEED: f32 = 400.0;
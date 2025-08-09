use crate::ai::input::{Input, InputSet};
use crate::ai::input_recorder::GameInputRecorder;
use crate::player::Player;
use crate::sensor::ground_sensor::GroundContact;
use crate::simulation::DELTA_TIME;
use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::prelude::{
    BevyError, Commands, Component, KeyCode, Query, Res, ResMut, Transform, Vec3, With,
};
use bevy_rapier3d::dynamics::Velocity;

#[derive(Component)]
pub struct HumanPlayer;

pub const PLAYER_TURN_SPEED: f32 = 10.0;

pub fn move_player(
    kb: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<(&mut Velocity, &Transform, &GroundContact), With<HumanPlayer>>,
    mut input_recorder: ResMut<GameInputRecorder>,
    mut app_exit: EventWriter<AppExit>,
) -> Result<(), BevyError> {
    let (mut velocity, transform, ground_contact) = player_query.single_mut()?;

    let mut move_input = Vec3::ZERO;

    let mut set = InputSet::default();

    // Forward/Backward
    if kb.pressed(KeyCode::KeyW) {
        move_input.z -= 1.0;
        set |= Input::Forward;
    } else if kb.pressed(KeyCode::KeyS) {
        move_input.z += 1.0;
        set |= Input::Backward;
    }

    // Left/Right
    if kb.pressed(KeyCode::KeyA) {
        move_input.x -= 1.0;
        set |= Input::Left;
    } else if kb.pressed(KeyCode::KeyD) {
        move_input.x += 1.0;
        set |= Input::Right;
    }

    // Normalize movement input to prevent diagonal speed boost
    if move_input.length_squared() > 0.0 {
        move_input = move_input.normalize();
    }

    let speed = 200.0;

    // Apply movement relative to player's facing direction
    let rotated_input = transform.rotation * move_input;
    velocity.linvel.x = rotated_input.x * speed;
    velocity.linvel.z = rotated_input.z * speed;

    // Yaw rotation (turn left or right)
    let mut yaw = 0.0;
    if kb.pressed(KeyCode::ArrowLeft) {
        yaw += PLAYER_TURN_SPEED;
        set |= Input::TurnLeft;
    }
    if kb.pressed(KeyCode::ArrowRight) {
        yaw -= PLAYER_TURN_SPEED;
        set |= Input::TurnRight;
    }

    velocity.angvel.y = yaw * DELTA_TIME;

    // Jump
    if kb.pressed(KeyCode::Space) && ground_contact.0 > 0 {
        velocity.linvel.y += 400.0;
        set |= Input::Jump;
    }

    if input_recorder.is_full() {
        app_exit.write(AppExit::Success);
        return Ok(());
    }

    input_recorder.record(set, String::new());

    Ok(())
}

use crate::ai::input::{Input, InputSet};
use crate::ai::input_recorder::{GameInputRecorder, InputRecorder};
use crate::component::player_character::{PlayerCharacter, PLAYER_HEIGHT, PLAYER_WIDTH};
use crate::human::camera_controller::{CameraController, MainCamera};
use crate::player::Player;
use crate::sensor::ground_sensor::GroundContact;
use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::prelude::{
    BevyError, Camera3d, Commands, Component, KeyCode, Query, Res, ResMut, Time, Transform, Vec3,
    With,
};
use bevy_rapier3d::dynamics::{AdditionalMassProperties, CoefficientCombineRule, LockedAxes, RigidBody, Velocity};
use bevy_rapier3d::geometry::{Collider, CollisionGroups, Friction, Group};
use crate::game::DELTA_TIME;

#[derive(Component)]
pub struct HumanPlayer;

pub const PLAYER_TURN_SPEED: f32 = 50.0;


pub fn spawn_human_player(mut commands: Commands) {
    // Spawn player entity
    commands.spawn((
        HumanPlayer,
        Player,
        Transform::default(),
        Velocity::default(),
        InheritedVisibility::VISIBLE,
        GroundContact(0),

        LockedAxes::ROTATION_LOCKED ^ LockedAxes::ROTATION_LOCKED_Y,

        RigidBody::Dynamic,
        AdditionalMassProperties::Mass(200.0),
        Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
        CollisionGroups::new(Group::GROUP_1, Group::ALL),
        Friction {
            coefficient: 0.0,
            combine_rule: CoefficientCombineRule::Min,
        },
    ));

    // Spawn camera
    commands.spawn((
        MainCamera,
        CameraController {
            sensitivity: 0.0005,
            pitch: 30.0f32.to_radians(),
            yaw: 45.0f32.to_radians(),
        },
        Camera3d::default(),
        Transform::from_xyz(-50.0, 250.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

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
        return Ok(())
    }

    input_recorder.record(set, String::new());

    Ok(())
}
